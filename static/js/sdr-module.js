export default class SDRController {
  constructor(apiBaseUrl) {
    this.apiBaseUrl = apiBaseUrl;
    this.wsUrl = apiBaseUrl.replace('http', 'ws') + '/ws/spectrum';
    this.sdrRunning = false;
    this.elements = null;
    this.waterfallData = [];
    this.socket = null;
    this.statusCallbacks = [];
    this.errorCallbacks = [];
    // Display parameters
    this.fftSize = 2048;
    this.spectrumLength = this.fftSize / 2;
    this.dbRange = 100; // dB scale range
    this.refLevel = -20; // Reference level in dB
    this.waterfallHeight = 300;
    
    // Color maps
    this.spectrumGradient = null;
    this.waterfallColors = this.createWaterfallColorMap();
    this.centerFreq = 100; // Default center frequency in MHz
  }

  init(elements) {
    this.elements = elements;
    this.setupEventListeners();
    this.initCanvases();
    this.updateStatus();
  }

  setupEventListeners() {
    if (!this.elements) return;

    // Connection controls
    this.elements.sdrStartBtn.addEventListener('click', () => this.connect());
    this.elements.sdrStopBtn.addEventListener('click', () => this.disconnect());

    // Frequency controls - changed to use 'input' event for real-time updates
    this.elements.sdrFrequency.addEventListener('input', (e) => {
      const freq = parseFloat(e.target.value);
      if (!isNaN(freq)) {
        this.setFrequency(freq);
      }
    });
    
    this.elements.sdrFreqUp.addEventListener('click', () => 
      this.adjustFrequency(0.1));
    this.elements.sdrFreqDown.addEventListener('click', () => 
      this.adjustFrequency(-0.1));

    // Mode selection
    this.elements.sdrMode.addEventListener('change', (e) => 
      this.setMode(e.target.value));

    // Gain controls
    this.elements.sdrLnaGain.addEventListener('input', (e) => {
      const value = parseFloat(e.target.value);
      this.elements.sdrLnaValue.textContent = `${value.toFixed(1)} dB`;
      this.setGain('lna', value);
    });

    // AGC control
    this.elements.sdrAgc.addEventListener('change', (e) => {
      const enabled = e.target.checked;
      this.elements.sdrAgcText.textContent = enabled ? 'Enabled' : 'Disabled';
      this.setAGC(enabled);
    });

    // Volume control
    this.elements.sdrVolume.addEventListener('input', (e) => {
      const value = parseInt(e.target.value);
      this.elements.sdrVolumeValue.textContent = `${value}%`;
      this.setVolume(value);
    });

    // Memory buttons
    this.elements.sdrMemoryBank.addEventListener('click', (e) => {
      if (e.target.classList.contains('memory-btn')) {
        const freq = parseFloat(e.target.dataset.freq);
        this.setFrequency(freq);
        this.elements.sdrFrequency.value = freq.toFixed(3);
      } else if (e.target.id === 'sdrAddMemory') {
        this.addMemoryChannel();
      }
    });

    // Record button
    this.elements.sdrRecordBtn.addEventListener('click', () => 
      this.toggleRecording());

    // Window resize
    window.addEventListener('resize', () => this.handleResize());
  }

createWaterfallColorMap() {
  const colors = [];
  for (let i = 0; i < 256; i++) {
    let r = 0, g = 0, b = 0;
    if (i < 30) b = i * 8.5;
    else if (i < 90) { g = (i - 30) * 4.25; b = 255; }
    else if (i < 120) { g = 255; b = 255 - (i - 90) * 8.5; }
    else if (i < 150) { r = (i - 120) * 8.5; g = 255; }
    else if (i < 210) { r = 255; g = 255 - (i - 150) * 4.25; }
    else { r = 255; g = b = (i - 210) * 5.55; }
    colors.push([r, g, b]);
  }
  return colors;
}

initCanvases() {
  if (!this.elements) return;

  const { spectrumCanvas, waterfallCanvas } = this.elements;

  spectrumCanvas.width = spectrumCanvas.clientWidth;
  spectrumCanvas.height = spectrumCanvas.clientHeight || 150;
  waterfallCanvas.width = spectrumCanvas.clientWidth;
  waterfallCanvas.height = this.waterfallHeight;

  const ctx = spectrumCanvas.getContext('2d');
  this.spectrumGradient = ctx.createLinearGradient(0, 0, 0, spectrumCanvas.height);
  this.spectrumGradient.addColorStop(0, '#00ffff');
  this.spectrumGradient.addColorStop(0.5, '#00ff00');
  this.spectrumGradient.addColorStop(1, '#0044ff');

  this.clearWaterfall();
  this.drawGrid();
}

drawGrid() {
  const { ctx, width, height } = this.getCanvasContext('spectrum');
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, width, height);

  const dbMin = -120;
  const dbMax = -10;
  const dbRange = dbMax - dbMin;

  ctx.strokeStyle = 'rgba(255,255,255,0.15)';
  ctx.lineWidth = 1;
  ctx.font = '11px monospace';
  ctx.fillStyle = '#aaa';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';

  for (let db = dbMin; db <= dbMax; db += 10) {
    const y = height - ((db - dbMin) / dbRange) * height;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
    ctx.fillText(`${db} dB`, 5, y - 5);
  }

  const totalBw = 3.2e6;
  const freqStartHz = (this.centerFreq * 1e6) - (totalBw / 2);
  const freqEndHz = (this.centerFreq * 1e6) + (totalBw / 2);

  const approxSpacingPx = 80;
  const minStepHz = totalBw * (approxSpacingPx / width);

  let stepHz;
  if (minStepHz <= 100e3) stepHz = 100e3;
  else if (minStepHz <= 200e3) stepHz = 200e3;
  else if (minStepHz <= 500e3) stepHz = 500e3;
  else stepHz = 1e6;

  const firstMarkHz = Math.ceil(freqStartHz / stepHz) * stepHz;

  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let freqHz = firstMarkHz; freqHz <= freqEndHz; freqHz += stepHz) {
    const x = ((freqHz - freqStartHz) / totalBw) * width;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
    const labelMHz = (freqHz / 1e6).toFixed(2);
    ctx.fillText(`${labelMHz} MHz`, x, height - 14);
  }

  ctx.strokeStyle = '#ff3333';
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(width / 2, 0);
  ctx.lineTo(width / 2, height);
  ctx.stroke();
  ctx.setLineDash([]);
}


updateSpectrum(fftData) {
  if (!this.elements) return;

  const ctx = this.elements.spectrumCanvas.getContext('2d');
  const width = this.elements.spectrumCanvas.width;
  const height = this.elements.spectrumCanvas.height;

  ctx.clearRect(0, 0, width, height);
  this.drawGrid();

  ctx.strokeStyle = this.spectrumGradient;
  ctx.lineWidth = 1.5;
  ctx.beginPath();

  const binWidth = width / this.spectrumLength;

  for (let i = 0; i < this.spectrumLength; i++) {
    const db = fftData[i];
    const clampedDb = Math.max(this.refLevel - this.dbRange, Math.min(this.refLevel, db));
    const y = height - ((clampedDb - (this.refLevel - this.dbRange)) * height / this.dbRange);
    const x = i * binWidth;

    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }

  ctx.stroke();
}

updateWaterfall(fftData) {
  if (!this.elements) return;

  const ctx = this.elements.waterfallCanvas.getContext('2d');
  const width = this.elements.waterfallCanvas.width;
  const height = this.elements.waterfallCanvas.height;

  const imageData = ctx.getImageData(0, 0, width, height - 1);
  ctx.putImageData(imageData, 0, 1);

  const lineImage = ctx.createImageData(width, 1);
  const data = lineImage.data;

  for (let x = 0; x < width; x++) {
    const bin = Math.floor((x / width) * this.spectrumLength);
    const db = fftData[bin];
    const clampedDb = Math.max(this.refLevel - this.dbRange, Math.min(this.refLevel, db));
    let intensity = Math.floor(255 * (clampedDb - (this.refLevel - this.dbRange)) / this.dbRange);
    intensity = Math.min(255, Math.max(0, intensity));
    const [r, g, b] = this.waterfallColors[intensity];
    const idx = x * 4;
    data[idx] = r;
    data[idx + 1] = g;
    data[idx + 2] = b;
    data[idx + 3] = 255;
  }

  ctx.putImageData(lineImage, 0, 0);
}

  getWaterfallColor(value) {
    // Use the precomputed color map
    return this.waterfallColors[Math.min(255, Math.max(0, value))];
  }

  clearWaterfall() {
    const { ctx, width, height } = this.getCanvasContext('waterfall');
    ctx.fillStyle = '#121a24';
    ctx.fillRect(0, 0, width, height);
    this.waterfallData = [];
  }

  getCanvasContext(type) {
    const canvas = this.elements[`${type}Canvas`];
    return {
      ctx: canvas.getContext('2d'),
      width: canvas.width,
      height: canvas.height
    };
  }

  handleResize() {
    this.initCanvases();
    if (this.sdrRunning) {
      this.drawGrid();
    }
  }

  normalizeFFT(fftData) {
    // Convert to Float32Array if needed
    const data = Array.isArray(fftData) ? new Float32Array(fftData) : fftData;
    
    // Apply smoothing and normalization
    const normalized = new Float32Array(data.length);
    let max = -Infinity;
    let min = Infinity;
    
    // Find min/max for normalization
    for (let i = 0; i < data.length; i++) {
      if (data[i] > max) max = data[i];
      if (data[i] < min) min = data[i];
    }
    
    const range = max - min;
    for (let i = 0; i < data.length; i++) {
      // Normalize to 0-1 range then scale to dB range
      normalized[i] = ((data[i] - min) / range) * this.dbRange + this.refLevel;
    }
    
    return normalized;
  }

  handleDisconnection(reason) {
    this.sdrRunning = false;
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
    this.stopVisualization();
    this.notifyStatus(false);
    this.notifyError(reason);
    
    // Update UI
    if (this.elements) {
      this.elements.sdrStartBtn.disabled = false;
      this.elements.sdrStopBtn.disabled = true;
      this.elements.sdrRecordBtn.disabled = true;
      this.elements.sdrConnectionIndicator.className = 'status-indicator disconnected';
      this.elements.sdrStatusText.textContent = 'Disconnected: ' + reason;
    }
  }

  setupWebSocket() {
    if (this.socket) {
      this.socket.close();
    }

    this.socket = new WebSocket(this.wsUrl);

    this.socket.onopen = () => {
      console.log('SDR WebSocket connected');
      this.startVisualization();
      this.notifyStatus(true);
    };

    this.socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // Handle status messages
        if (data.status === "disconnected") {
          this.handleDisconnection(data.message);
          return;
        }
        
        // Process FFT data
        if (data.fft && Array.isArray(data.fft)) {
          // Normalize FFT data
          const normalizedFFT = this.normalizeFFT(data.fft);
          this.updateSpectrum(normalizedFFT);
          this.updateWaterfall(normalizedFFT);
          
          // Update center frequency display
          if (data.center_freq) {
            this.centerFreq = data.center_freq;
            if (this.elements.sdrFrequency) {
              this.elements.sdrFrequency.value = (data.center_freq / 1e6).toFixed(3);
            }
          }
        }
      } catch (error) {
        console.error('Error processing WebSocket message:', error);
      }
    };

    this.socket.onclose = (event) => {
      console.log('SDR WebSocket disconnected:', event.code, event.reason);
      this.handleDisconnection(event.reason || "Connection closed");
    };

    this.socket.onerror = (error) => {
      console.error('SDR WebSocket error:', error);
      this.notifyError('Connection error');
    };
  }

  async connect() {
    try {
      const response = await fetch(`${this.apiBaseUrl}/api/sdr/connect`, {
        method: 'POST'
      });
      
      const data = await response.json();
      
      if (response.ok) {
        this.sdrRunning = true;
        this.setupWebSocket();
        this.updateUI(true);
        return data;
      } else {
        throw new Error(data.message || 'Connection failed');
      }
    } catch (error) {
      console.error('Connection error:', error);
      throw error;
    }
  }

  async disconnect() {
    try {
      const response = await fetch(`${this.apiBaseUrl}/api/sdr/disconnect`, {
        method: 'POST'
      });
      
      const data = await response.json();
      
      if (response.ok) {
        this.sdrRunning = false;
        if (this.socket) {
          this.socket.close();
          this.socket = null;
        }
        this.updateUI(false);
        return data;
      } else {
        throw new Error(data.message || 'Disconnection failed');
      }
    } catch (error) {
      console.error('Disconnection error:', error);
      throw error;
    }
  }

  updateUI(connected) {
    if (!this.elements) return;
    
    const { 
      sdrStartBtn, sdrStopBtn, sdrRecordBtn, 
      sdrConnectionIndicator, sdrStatusText 
    } = this.elements;
    
    sdrStartBtn.disabled = connected;
    sdrStopBtn.disabled = !connected;
    sdrRecordBtn.disabled = !connected;
    
    sdrConnectionIndicator.className = `status-indicator ${connected ? 'connected' : 'disconnected'}`;
    sdrStatusText.textContent = connected ? 'Connected' : 'Disconnected';
  }

async setFrequency(freqMHz) {
  if (!this.sdrRunning) {
    throw new Error("Device not connected");
  }

  // Convert to number and validate
  const freq = Number(freqMHz);
  if (isNaN(freq)) {
    throw new Error("Invalid frequency value");
  }

  // Ensure frequency is in valid range (24-1766 MHz for RTL-SDR)
  if (freq < 24 || freq > 1766) {
    throw new Error("Frequency must be between 24-1766 MHz");
  }

  try {
    const response = await fetch(`${this.apiBaseUrl}/api/sdr/frequency`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
body: JSON.stringify({ frequency: freq })
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to set frequency');
    }
    
    // Update UI if successful
    if (this.elements?.sdrFrequency) {
      this.elements.sdrFrequency.value = freq.toFixed(3);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Frequency set error:', error.message);
    this.notifyError(error.message);
    throw error;
  }
}


async adjustFrequency(stepMHz) {
  if (!this.elements?.sdrFrequency || !this.sdrRunning) return;
  
  const currentFreq = parseFloat(this.elements.sdrFrequency.value);
  const newFreq = Math.max(24, Math.min(1766, currentFreq + stepMHz));
  
  try {
    await this.setFrequency(newFreq);
    // Don't need to set the value here as setFrequency will update it
  } catch (error) {
    console.error('Frequency adjust failed:', error);
    this.notifyError(`Frequency adjust failed: ${error.message}`);
  }
}
  startVisualization() {
    this.stopVisualization();
    const render = () => {
      // This will be called by WebSocket data
      this.animationFrameId = requestAnimationFrame(render);
    };
    this.animationFrameId = requestAnimationFrame(render);
  }

  stopVisualization() {
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
  }

  async setMode(mode) {
    if (!this.sdrRunning) return;
    
    try {
      const response = await fetch(`${this.apiBaseUrl}/api/sdr/mode`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ mode })
      });
      
      const data = await response.json();
      
      if (response.ok) {
        return data;
      } else {
        throw new Error(data.message || 'Failed to set mode');
      }
    } catch (error) {
      this.notifyError(error.message);
      throw error;
    }
  }

  async setGain(gainType, value) {
    if (!this.sdrRunning) return;
    
    try {
      const response = await fetch(`${this.apiBaseUrl}/api/sdr/gain`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ gain_type: gainType, value })
      });
      
      const data = await response.json();
      
      if (response.ok) {
        return data;
      } else {
        throw new Error(data.message || 'Failed to set gain');
      }
    } catch (error) {
      this.notifyError(error.message);
      throw error;
    }
  }

  async setAGC(enabled) {
    if (!this.sdrRunning) return;
    
    try {
      const response = await fetch(`${this.apiBaseUrl}/api/sdr/agc`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ enabled })
      });
      
      const data = await response.json();
      
      if (response.ok) {
        return data;
      } else {
        throw new Error(data.message || 'Failed to set AGC');
      }
    } catch (error) {
      this.notifyError(error.message);
      throw error;
    }
  }

  async setVolume(level) {
    if (!this.elements) return;
    
    try {
      this.elements.sdrAudioOutput.volume = level / 100;
    } catch (error) {
      this.notifyError(error.message);
      throw error;
    }
  }

  async toggleRecording() {
    if (!this.sdrRunning) return;
    
    this.recording = !this.recording;
    
    if (this.elements) {
      if (this.recording) {
        this.elements.sdrRecordBtn.classList.add('btn-danger');
        this.elements.sdrRecordBtn.classList.remove('btn-secondary');
        this.elements.sdrRecordBtn.innerHTML = '<i class="fas fa-stop-circle"></i> Recording';
      } else {
        this.elements.sdrRecordBtn.classList.remove('btn-danger');
        this.elements.sdrRecordBtn.classList.add('btn-secondary');
        this.elements.sdrRecordBtn.innerHTML = '<i class="fas fa-record-vinyl"></i> Record';
      }
    }
  }

  async addMemoryChannel() {
    if (!this.elements || !this.sdrRunning) return;
    
    const currentFreq = parseFloat(this.elements.sdrFrequency.value);
    const currentMode = this.elements.sdrMode.value;
    
    const btn = document.createElement('button');
    btn.className = 'btn btn-sm btn-secondary memory-btn';
    btn.dataset.freq = currentFreq.toFixed(3);
    btn.textContent = `${currentFreq.toFixed(3)} ${currentMode.toUpperCase()}`;
    
    this.elements.sdrMemoryBank.insertBefore(btn, this.elements.sdrAddMemory);
  }
async updateStatus() {
  try {
    const response = await fetch(`${this.apiBaseUrl}/api/sdr/status`);
    const data = await response.json();

    if (response.ok && this.elements) {
      // Convert frequency from Hz to MHz
      const freqMHz = (typeof data.frequency === 'number') ? data.frequency / 1e6 : 0;

      // Safely get mode string or default to empty string
      const mode = (typeof data.mode === 'string' && data.mode) ? data.mode.toUpperCase() : '';

      this.elements.sdrDeviceInfo.textContent = data.connected 
        ? `RTL-SDR @ ${freqMHz.toFixed(3)} MHz${mode ? `, ${mode}` : ''}`
        : '';

      return data;
    } else {
      throw new Error(data.message || 'Failed to get SDR status');
    }
  } catch (error) {
    this.notifyError(error.message);
    throw error;
  }
}

  onStatusUpdate(callback) {
    if (typeof callback === 'function') {
      this.statusCallbacks.push(callback);
    }
  }

  notifyStatus(connected) {
    this.statusCallbacks.forEach(cb => {
      try {
        cb(connected);
      } catch (e) {
        console.error('Error in status callback:', e);
      }
    });
  }

  onError(callback) {
    if (typeof callback === 'function') {
      this.errorCallbacks.push(callback);
    }
  }

  notifyError(message) {
    this.errorCallbacks.forEach(cb => {
      try {
        cb(message);
      } catch (e) {
        console.error('Error in error callback:', e);
      }
    });
  }
}