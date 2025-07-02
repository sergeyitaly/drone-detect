        
    
        // System State
        let audioContext;
        let microphone;
        let audioAnalyser;
        let gainNode;
        let isListening = false;
        let analysisStartTime;
        let lastDetectionTime = 0;
        let animationFrameId;
        let lastFrameTime = 0;

        // Then later in your code, add these new state variables
        let ongoingDetection = null;
        let detectionStartTime = null;
        // Audio Processing Parameters
        const BUFFER_SIZE = 2048;
        let SAMPLE_RATE = 44100; // Will be updated with actual rate
        const DRONE_FREQ_RANGE = [260, 440]; // Hz
        const MIN_DETECTION_DURATION = 100; // ms
        const FRAME_INTERVAL = 100; // Process audio every 100ms (10fps) instead of 60fps

        // DOM Elements
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const statusIndicator = document.getElementById('statusIndicator');
        const statusText = document.getElementById('statusText');
        const droneAlert = document.getElementById('droneAlert');
        const alertDetails = document.getElementById('alertDetails');
        const audioLevelEl = document.getElementById('audioLevel');
        const dominantFrequencyEl = document.getElementById('dominantFrequency');
        const harmonicScoreEl = document.getElementById('harmonicScore');
        const analysisDurationEl = document.getElementById('analysisDuration');
        const detectionsList = document.getElementById('detectionsList');
        const gainControl = document.getElementById('gainControl');
        const gainValue = document.getElementById('gainValue');
        const sensitivity = document.getElementById('sensitivity');
        const sensitivityValue = document.getElementById('sensitivityValue');

        // Initialize Charts with sample data
        const frequencyCtx = document.getElementById('frequencyChart').getContext('2d');

        
        // Frequency Spectrum Chart
        const frequencyChart = new Chart(frequencyCtx, {
            type: 'bar',
            data: {
                labels: Array.from({length: 100}, (_, i) => i * 5),
                datasets: [{
                    label: 'Frequency Power',
                    data: Array(100).fill(-80),
                    backgroundColor: (ctx) => {
                        const freq = ctx.dataIndex * 5;
                        const inDroneRange = freq >= DRONE_FREQ_RANGE[0] && freq <= DRONE_FREQ_RANGE[1];
                        return inDroneRange ? 'rgba(234, 84, 85, 0.7)' : 'rgba(65, 140, 216, 0.7)';
                    },
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                devicePixelRatio: 2, // Double resolution for sharper chart

                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (context) => `${context.label}Hz: ${context.raw.toFixed(2)}`
                        }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Frequency (Hz)' },
                        min: 0,
                        max: 500
                    },
                    y: {
                        title: { display: true, text: 'Power (dB)' },
                        min: -100,
                        max: 0
                    }
                }
            }
        });
        // WebSocket and API connection management
        let websocket = null;
        let currentSessionId = null;
        let isConnected = false;
        const API_BASE_URL = "http://localhost:8000";

        // Connect to WebSocket
        async function connectWebSocket() {
            try {
                const wsUrl = API_BASE_URL.replace('http', 'ws') + '/ws/detections';
                websocket = new WebSocket(wsUrl);

                websocket.onopen = () => {
                    isConnected = true;
                    updateStatus(true, 'Connected to server');
                    console.log('WebSocket connection established');
                };

                websocket.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    handleWebSocketMessage(data);
                };

                websocket.onclose = () => {
                    isConnected = false;
                    updateStatus(false, 'Disconnected from server');
                    console.log('WebSocket connection closed');
                    // Attempt to reconnect after a delay
                    setTimeout(connectWebSocket, 3000);
                };

                websocket.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    updateStatus(false, 'Connection error');
                };

            } catch (error) {
                console.error('WebSocket connection error:', error);
                updateStatus(false, 'Connection failed');
            }
        }

        // Handle incoming WebSocket messages
        function handleWebSocketMessage(data) {
            switch (data.type) {
                case 'connection':
                    console.log('WebSocket connection status:', data.status);
                    if (data.session_id) {
                        currentSessionId = data.session_id;
                    }
                    break;
                    
                case 'session':
                    if (data.status === 'started') {
                        currentSessionId = data.session_id;
                        console.log('New session started:', currentSessionId);
                    }
                    break;
                    
                case 'detection':
                    // Handle incoming detection data
                    addDetectionLog(data);
                    updateDetectionStats(data);
                    if (data.confidence > 0.7) {
                        showDroneAlert(data);
                    }
                    break;
                    
                case 'audio_metrics':
                    // Update visualizations with backend data
                    updateVisualizations(data.frequency_data, data.time_data);
                    updateAudioLevel(data.rms);
                    break;
                    
                case 'keepalive':
                    // Just acknowledge keepalive
                    break;
                    
                default:
                    console.log('Unknown WebSocket message type:', data.type);
            }
        }

        async function startListeningSession() {
            try {
                const response = await fetch(`${API_BASE_URL}/sessions`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        sensitivity: parseFloat(sensitivity.value),
                        frequency_range: DRONE_FREQ_RANGE
                    })
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Failed to start session');
                }
                
                return await response.json();
            } catch (error) {
                console.error('Error:', error);
                throw error;
            }
        }

        async function stopListeningSession() {
            try {
                if (!currentSessionId) return;
                
                const response = await fetch(`${API_BASE_URL}/sessions/${currentSessionId}/stop`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const data = await response.json();
                console.log('Listening session stopped:', data.session_id);
                return data;
                
            } catch (error) {
                console.error('Error stopping listening session:', error);
                updateStatus(false, 'Stop listening failed');
                throw error;
            }
        }

        async function getCurrentSession() {
            try {
                const response = await fetch(`${API_BASE_URL}/sessions/current`);
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                return await response.json();
                
            } catch (error) {
                console.error('Error getting current session:', error);
                return null;
            }
        }

        async function getDetections(sessionId = null) {
            try {
                const url = sessionId 
                    ? `${API_BASE_URL}/detections?session_id=${sessionId}`
                    : `${API_BASE_URL}/detections`;
                    
                const response = await fetch(url);
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                return await response.json();
                
            } catch (error) {
                console.error('Error getting detections:', error);
                return [];
            }
        }

        async function calibrateFanNoise() {
            try {
                const response = await fetch(`${API_BASE_URL}/calibrate_fan_noise`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        duration: 5, // seconds
                        gain: parseFloat(gainControl.value)
                    })
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                return await response.json();
                
            } catch (error) {
                console.error('Error calibrating fan noise:', error);
                throw error;
            }
        }

        // Update system status
        function updateStatus(active, text) {
            statusIndicator.className = active ? 'status-indicator active' : 'status-indicator inactive';
            statusText.textContent = text;
        }

        // Update current time
        function updateCurrentTime() {
            const now = new Date();
            document.getElementById('currentTime').textContent = now.toLocaleTimeString();
            
            if (isListening && analysisStartTime) {
                const duration = new Date(now - analysisStartTime);
                const minutes = duration.getUTCMinutes().toString().padStart(2, '0');
                const seconds = duration.getUTCSeconds().toString().padStart(2, '0');
                analysisDurationEl.textContent = `${minutes}:${seconds}`;
            }
            
            requestAnimationFrame(updateCurrentTime);
        }

        function addDetectionLog(entry) {
            const now = Date.now();
            
            // Remove empty state if exists
            const emptyState = detectionsList.querySelector('.empty-state');
            if (emptyState) detectionsList.removeChild(emptyState);
            
            const logEntry = document.createElement('div');
            logEntry.className = `log-entry ${entry.confidence > 0.7 ? 'high-confidence' : ''}`;
            logEntry.innerHTML = `
                <div class="log-time">${new Date(entry.timestamp).toLocaleTimeString()}</div>
                <div class="log-details">
                    <span class="log-freq">${Math.round(entry.frequency)}Hz</span>
                    <span class="log-confidence">${Math.round(entry.confidence * 100)}%</span>
                    <span class="log-duration">${(entry.duration / 1000).toFixed(1)}s</span>
                </div>
                <div class="log-harmonics">${entry.harmonics || 0} harmonics</div>
            `;
            
            // Add click handler to show waveform
            logEntry.addEventListener('click', () => {
                showDetectionWaveform(entry);
            });
            
            detectionsList.prepend(logEntry);
            
            // Limit to 50 entries
            while (detectionsList.children.length > 50) {
                detectionsList.removeChild(detectionsList.lastChild);
            }
        }

        
        function showDetectionWaveform(detection) {
            const modal = document.getElementById('waveformModal');
            const modalTitle = document.getElementById('modalTitle');
            const modalFrequency = document.getElementById('modalFrequency');
            const modalConfidence = document.getElementById('modalConfidence');
            const modalHarmonics = document.getElementById('modalHarmonics');
            const modalTimestamp = document.getElementById('modalTimestamp');
            
            // Update modal content
            modalTitle.textContent = `Detection Details - ${new Date(detection.timestamp).toLocaleTimeString()}`;
            modalFrequency.textContent = `${Math.round(detection.frequency)}Hz`;
            modalConfidence.textContent = `${Math.round(detection.confidence * 100)}%`;
            modalHarmonics.textContent = detection.harmonics || 0;
            modalTimestamp.textContent = new Date(detection.timestamp).toLocaleString();
            
            // Show the modal
            modal.style.display = 'flex';
            
            // Create or update waveform chart
            const ctx = document.getElementById('detectionWaveform').getContext('2d');
            
            // Destroy previous chart if it exists
            if (ctx.chart) {
                ctx.chart.destroy();
            }
            
            // Generate data for the chart
            const dataPoints = 200;
            const waveformData = Array(dataPoints).fill(0).map((_, i) => {
                // Create a waveform that matches the detection frequency
                const freq = detection.frequency / 50; // Scale for visualization
                return 0.8 * Math.sin(i * freq * 0.1) * Math.exp(-0.005 * i);
            });
            
            // Create new chart
            ctx.chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: Array.from({length: dataPoints}, (_, i) => i),
                    datasets: [{
                        label: 'Audio Waveform',
                        data: waveformData,
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        fill: true,
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    devicePixelRatio: 2, // Double resolution for sharper chart
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: (context) => `Amplitude: ${context.raw.toFixed(4)}`
                            }
                        }
                    },
                    scales: {
                        x: {
                            title: { display: true, text: 'Time (samples)' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        },
                        y: {
                            title: { display: true, text: 'Amplitude' },
                            min: -1,
                            max: 1,
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        }
                    }
                }
            });
            
            // Add close button handler if not already added
            if (!modal.closeHandlerAdded) {
                modal.querySelector('.close-modal').addEventListener('click', () => {
                    modal.style.display = 'none';
                });
                
                // Close modal when clicking outside
                modal.addEventListener('click', (e) => {
                    if (e.target === modal) {
                        modal.style.display = 'none';
                    }
                });
                
                modal.closeHandlerAdded = true;
            }
        }
                
        // Update detection stats
        function updateDetectionStats(detection) {
            dominantFrequencyEl.textContent = `${Math.round(detection.frequency)} Hz`;
            harmonicScoreEl.textContent = `${Math.round(detection.confidence * 100)}%`;
        }

        // Update audio level display
        function updateAudioLevel(db) {
            audioLevelEl.textContent = `${Math.round(db)} dB`;
        }

        // Show drone alert
        function showDroneAlert(detection) {
            const confidence = Math.round(detection.confidence * 100);
            const freq = Math.round(detection.frequency);
            const harmonics = detection.harmonics || 0;
            
            document.getElementById('alertType').textContent = 
                confidence > 80 ? "HIGH CONFIDENCE DRONE DETECTED!" : "Potential Drone Detected";
            
            alertDetails.textContent = `${freq}Hz | ${confidence}% confidence | ${harmonics} harmonics`;
            droneAlert.style.display = 'flex';
            
            setTimeout(() => {
                droneAlert.style.display = 'none';
            }, 5000);
        }

        // Update all visualizations
        function updateVisualizations(freqData, timeData) {
            if (!freqData || !timeData) {
                // Generate mock data if no real data is available
                freqData = Array(100).fill(0).map((_, i) => {
                    const baseFreq = i * 5;
                    // Simulate some noise with occasional peaks
                    let value = -80 + Math.random() * 20;
                    // Add some peaks in drone range
                    if (baseFreq >= DRONE_FREQ_RANGE[0] && baseFreq <= DRONE_FREQ_RANGE[1] && Math.random() > 0.7) {
                        value = -40 + Math.random() * 20;
                    }
                    return value;
                });
                
                timeData = Array(100).fill(0).map(() => Math.random() * 0.5 - 0.25);
            }
            
            // Update frequency chart
            frequencyChart.data.datasets[0].data = freqData.slice(0, 100);
            frequencyChart.update('none');
            
            // Update waveform chart (downsampled)
            const waveformPoints = [];
            for (let i = 0; i < timeData.length; i += Math.ceil(timeData.length / 100)) {
                waveformPoints.push({x: i, y: timeData[i]});
            }
            
            // Update harmonic chart if we have frequency data
            if (freqData.length > 0) {
                const peaks = findPeaks(freqData, DRONE_FREQ_RANGE);
                if (peaks.length > 0) {
                    const fundamental = peaks.reduce((a, b) => 
                        a.amplitude > b.amplitude ? a : b
                    );
                    
                    const harmonics = [];
                    for (let n = 1; n <= 5; n++) {
                        const harmonicFreq = fundamental.frequency * n;
                        if (harmonicFreq > 500) break;
                        
                        const bin = Math.floor(harmonicFreq / (SAMPLE_RATE / 2 / freqData.length));
                        if (bin < freqData.length) {
                            harmonics.push({
                                x: harmonicFreq,
                                y: freqData[bin],
                                harmonic: n
                            });
                        }
                    }
                    
                } else {
                    // Clear harmonics if no peaks found
                }
            }
        }

        function findPeaks(frequencies, range) {
            const peaks = [];
            const minAmp = -80; // Increased sensitivity for 50-60 dB signals
            const minPeakDistance = 5; // Reduced to catch closer peaks
            const harmonicTolerance = 0.03; // 3% frequency tolerance for harmonics
        
            // Calculate frequency resolution
            const binSize = SAMPLE_RATE / 2 / frequencies.length;
            const startBin = Math.max(0, Math.floor(range[0] / binSize));
            const endBin = Math.min(frequencies.length - 1, Math.ceil(range[1] / binSize));
        
            // Find all peaks in the target range
            for (let i = startBin; i <= endBin; i++) {
                if (frequencies[i] < minAmp) continue;
        
                const isPeak = (i === 0 || frequencies[i] >= frequencies[i - 1]) && 
                              (i === frequencies.length - 1 || frequencies[i] >= frequencies[i + 1]);
        
                if (isPeak) {
                    peaks.push({
                        frequency: i * binSize,
                        amplitude: frequencies[i],
                        bin: i
                    });
                }
            }
        
            // Sort by amplitude (descending)
            peaks.sort((a, b) => b.amplitude - a.amplitude);
        
            // Find harmonic series
            const harmonicSeries = [];
            for (const peak of peaks) {
                const harmonics = [peak];
                let validHarmonics = 1; // Count the fundamental
        
                // Check for harmonics (up to 5th)
                for (let n = 2; n <= 5; n++) {
                    const targetFreq = peak.frequency * n;
                    const tolerance = targetFreq * harmonicTolerance;
        
                    // Find the strongest peak within tolerance
                    let bestHarmonic = null;
                    for (const candidate of peaks) {
                        if (Math.abs(candidate.frequency - targetFreq) <= tolerance) {
                            if (!bestHarmonic || candidate.amplitude > bestHarmonic.amplitude) {
                                bestHarmonic = candidate;
                            }
                        }
                    }
        
                    if (bestHarmonic) {
                        harmonics.push(bestHarmonic);
                        validHarmonics++;
                    }
                }
        
                // Only consider series with at least 3 harmonics (fundamental + 2)
                if (validHarmonics >= 3) {
                    harmonicSeries.push({
                        fundamental: peak,
                        harmonics: harmonics,
                        count: validHarmonics,
                        totalPower: harmonics.reduce((sum, h) => sum + h.amplitude, 0)
                    });
                }
            }
        
            // Return the best harmonic series (if any)
            if (harmonicSeries.length > 0) {
                // Sort by harmonic count then total power
                harmonicSeries.sort((a, b) => {
                    if (b.count !== a.count) return b.count - a.count;
                    return b.totalPower - a.totalPower;
                });
        
                return harmonicSeries[0].harmonics;
            }
        
            return [];
        }
        
        function processAudioData(timestamp) {
            if (!audioAnalyser || !isListening) {
                animationFrameId = null;
                return;
            }
            
            // Throttle processing to reduce CPU load
            if (timestamp - lastFrameTime < FRAME_INTERVAL) {
                animationFrameId = requestAnimationFrame(processAudioData);
                return;
            }
            lastFrameTime = timestamp;
        
            // Frequency data
            const frequencyData = new Uint8Array(audioAnalyser.frequencyBinCount);
            audioAnalyser.getByteFrequencyData(frequencyData);
            
            // Time domain data
            const timeData = new Float32Array(audioAnalyser.fftSize);
            audioAnalyser.getFloatTimeDomainData(timeData);
            
            // Batch DOM updates to reduce reflows
            requestAnimationFrame(() => {
                // Convert to dB scale for visualization
                const scaledFrequencyData = Array.from(frequencyData).map(val => {
                    return (val / 255) * 100 - 100;
                });
                
                // Update visualizations
                updateVisualizations(scaledFrequencyData, Array.from(timeData));
                
                // Calculate RMS for audio level
                let sum = 0;
                for (let i = 0; i < timeData.length; i++) {
                    sum += timeData[i] * timeData[i];
                }
                const rms = Math.sqrt(sum / timeData.length);
                const db = 20 * Math.log10(rms);
                updateAudioLevel(isFinite(db) ? db : -100);
            });
            
            // Perform drone detection analysis (in a web worker if possible)
            analyzeForDrones(frequencyData);
            
            animationFrameId = requestAnimationFrame(processAudioData);
        }        

        function analyzeForDrones(frequencyData) {
            const now = Date.now();
            
            // Only analyze every 500ms to reduce CPU usage
            if (now - lastDetectionTime < 500) return;
            
            lastDetectionTime = now;
            
            // Convert frequency data to dB scale (assuming 0-255 input)
            const dBData = Array.from(frequencyData).map(val => {
                return 20 * Math.log10((val + 1) / 255 * 100); // +1 to avoid log(0)
            });
        
            // Find peaks in the frequency spectrum
            const peaks = findPeaks(dBData, DRONE_FREQ_RANGE);
            
            if (peaks.length > 0) {
                // Find the strongest peak in our target range
                const strongestPeak = peaks.reduce((a, b) => 
                    a.amplitude > b.amplitude ? a : b
                );
                
                // Verify this is actually a significant peak (70 dB)
                if (strongestPeak.amplitude < 70) {
                    resetOngoingDetection();
                    return;
                }
        
                // Check for harmonics (2nd, 3rd, and 4th)
                const harmonicTolerance = 0.03; // 3% frequency tolerance
                const harmonics = [];
                
                for (let n = 2; n <= 4; n++) {
                    const targetFreq = strongestPeak.frequency * n;
                    const targetBin = Math.floor(targetFreq / (SAMPLE_RATE / 2 / dBData.length));
                    const minBin = Math.max(0, targetBin - 2);
                    const maxBin = Math.min(dBData.length - 1, targetBin + 2);
                    
                    // Find strongest signal within ±2 bins of harmonic frequency
                    let maxAmp = -Infinity;
                    let harmonicFreq = 0;
                    
                    for (let bin = minBin; bin <= maxBin; bin++) {
                        if (dBData[bin] > maxAmp) {
                            maxAmp = dBData[bin];
                            harmonicFreq = bin * (SAMPLE_RATE / 2 / dBData.length);
                        }
                    }
                    
                    if (maxAmp > 40) { // Minimum harmonic amplitude (40 dB)
                        harmonics.push({
                            frequency: harmonicFreq,
                            amplitude: maxAmp,
                            harmonic: n
                        });
                    }
                }
                
                // Calculate confidence score
                const amplitudeScore = Math.min(1, (strongestPeak.amplitude - 40) / 30); // 40-70 dB → 0-1
                const harmonicScore = harmonics.length / 3; // 0-1 based on found harmonics
                const confidence = amplitudeScore * 0.6 + harmonicScore * 0.4;
                
                if (confidence > 0.6) { // Higher confidence threshold
                    if (!ongoingDetection) {
                        ongoingDetection = {
                            frequency: strongestPeak.frequency,
                            confidence: confidence,
                            harmonics: harmonics.length,
                            amplitudes: [strongestPeak.amplitude, ...harmonics.map(h => h.amplitude)],
                            startTime: now
                        };
                        detectionStartTime = now;
                    } else {
                        // Update ongoing detection with weighted average
                        ongoingDetection.frequency = ongoingDetection.frequency * 0.7 + strongestPeak.frequency * 0.3;
                        ongoingDetection.confidence = ongoingDetection.confidence * 0.7 + confidence * 0.3;
                        ongoingDetection.harmonics = Math.round(
                            ongoingDetection.harmonics * 0.7 + harmonics.length * 0.3
                        );
                        // Update amplitudes (weighted)
                        ongoingDetection.amplitudes[0] = ongoingDetection.amplitudes[0] * 0.7 + strongestPeak.amplitude * 0.3;
                        for (let i = 0; i < harmonics.length; i++) {
                            if (i < ongoingDetection.amplitudes.length - 1) {
                                ongoingDetection.amplitudes[i+1] = ongoingDetection.amplitudes[i+1] * 0.7 + harmonics[i].amplitude * 0.3;
                            }
                        }
                    }
                } else {
                    resetOngoingDetection();
                }
            } else {
                resetOngoingDetection();
            }
            
            // Check for sustained detection
            if (ongoingDetection && now - detectionStartTime >= MIN_DETECTION_DURATION) {
                const avgAmplitude = ongoingDetection.amplitudes.reduce((a, b) => a + b, 0) / ongoingDetection.amplitudes.length;
                
                const detection = {
                    timestamp: new Date().toISOString(),
                    frequency: ongoingDetection.frequency,
                    confidence: ongoingDetection.confidence,
                    harmonics: ongoingDetection.harmonics,
                    dbLevel: avgAmplitude,
                    duration: now - detectionStartTime
                };
                
                logDetection(detection);
                resetOngoingDetection();
            }
        }
        

        function resetOngoingDetection() {
            ongoingDetection = null;
            detectionStartTime = null;
        }
        
        function logDetection(detection) {
            addDetectionLog(detection);
            updateDetectionStats(detection);
            
            if (detection.confidence > 0.7) {
                showDroneAlert(detection);
            }
            
            // Send detection to server if WebSocket is connected
            if (isConnected && websocket) {
                websocket.send(JSON.stringify({
                    type: 'detection',
                    ...detection,
                    session_id: currentSessionId
                }));
            }
        }

        // Start audio processing with backend integration
        async function startListening() {
            try {
                updateStatus(false, 'Starting session...');
                
                // Start backend listening session
                await startListeningSession();
                
                // Start local audio processing for visualization
                await startLocalAudioProcessing();
                
                isListening = true;
                analysisStartTime = new Date();
                startBtn.disabled = true;
                stopBtn.disabled = false;
                updateStatus(true, 'Listening for drones...');
                
                // Start processing audio data
                processAudioData();
                
            } catch (error) {
                console.error('Error starting listening:', error);
                updateStatus(false, 'Start failed');
                stopListening();
            }
        }
        // Add mock detection for testing
        function addMockDetection(freq, confidence, harmonics) {
            const mockDetection = {
                timestamp: new Date().toISOString(),
                frequency: freq,
                confidence: confidence,
                harmonics: harmonics,
                rms: -20 + Math.random() * 10
            };
            
            addDetectionLog(mockDetection);
            updateDetectionStats(mockDetection);
            
            if (confidence > 0.7) {
                showDroneAlert(mockDetection);
            }
        }

        // Separate function for local audio processing (visualization only)
        async function startLocalAudioProcessing() {
            try {
                // Create audio context
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                SAMPLE_RATE = audioContext.sampleRate;
                
                // Create nodes
                audioAnalyser = audioContext.createAnalyser();
                audioAnalyser.fftSize = BUFFER_SIZE;
                audioAnalyser.smoothingTimeConstant = 0.8;
                
                gainNode = audioContext.createGain();
                updateGain();
                
                // Get microphone stream
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: false,
                        sampleRate: 44100
                    } 
                });
                
                // Create source
                microphone = audioContext.createMediaStreamSource(stream);
                
                // Connect nodes: mic → gain → analyser → destination
                microphone.connect(gainNode);
                gainNode.connect(audioAnalyser);
                audioAnalyser.connect(audioContext.destination);
                
            } catch (error) {
                console.error('Audio initialization error:', error);
                
                // If microphone access fails, use mock data
                updateStatus(true, 'Using simulated data (mic not available)');
                
                // Start processing mock data
                processMockData();
                
                throw error;
            }
        }

        // Process mock data when microphone isn't available
        function processMockData() {
            if (!isListening) return;
            
            // Generate random frequency data with occasional peaks
            const freqData = Array(100).fill(0).map((_, i) => {
                const baseFreq = i * 5;
                // Simulate some noise with occasional peaks
                let value = -80 + Math.random() * 20;
                // Add some peaks in drone range
                if (baseFreq >= DRONE_FREQ_RANGE[0] && baseFreq <= DRONE_FREQ_RANGE[1] && Math.random() > 0.7) {
                    value = -40 + Math.random() * 20;
                }
                return value;
            });
            
            // Generate random time domain data
            const timeData = Array(100).fill(0).map(() => Math.random() * 0.5 - 0.25);
            
            // Update visualizations
            updateVisualizations(freqData, timeData);
            
            // Random audio level
            updateAudioLevel(-30 + Math.random() * 20);
            
            // Continue processing
            animationFrameId = requestAnimationFrame(processMockData);
        }

        // Stop audio processing
        async function stopListening() {
            try {
                // Stop backend listening session
                await stopListeningSession();
            } catch (error) {
                console.error('Error stopping backend session:', error);
            }
            
            // Stop local audio processing
            await stopLocalAudioProcessing();
            
            isListening = false;
            startBtn.disabled = false;
            stopBtn.disabled = true;
            updateStatus(false, 'Ready to start');
            
            // Reset displays
            audioLevelEl.textContent = '- dB';
            dominantFrequencyEl.textContent = '- Hz';
            harmonicScoreEl.textContent = '0%';
        }

        // Separate function for stopping local audio
        async function stopLocalAudioProcessing() {
            if (microphone) {
                microphone.disconnect();
                microphone = null;
            }
            
            if (audioContext) {
                await audioContext.close();
                audioContext = null;
            }
            
            if (animationFrameId) {
                cancelAnimationFrame(animationFrameId);
                animationFrameId = null;
            }
        }

        // Update gain from control
        function updateGain() {
            if (gainNode) {
                gainNode.gain.value = parseFloat(gainControl.value);
                gainValue.textContent = gainControl.value;
            }
        }

        // Update sensitivity from control
        function updateSensitivity() {
            sensitivityValue.textContent = sensitivity.value;
            if (isConnected && websocket) {
                websocket.send(JSON.stringify({
                    type: 'sensitivity_update',
                    value: parseFloat(sensitivity.value)
                }));
            }
        }

        // Clear detection logs
        function clearLogs() {
            detectionLog = [];
            detectionsList.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-microphone"></i>
                    <p>Waiting for audio input...</p>
                </div>
            `;
        }

        
        document.addEventListener('DOMContentLoaded', () => {
            // Button references must match your HTML IDs exactly
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            const clearLogsBtn = document.getElementById('clearLogs');
            const gainControl = document.getElementById('gainControl');
            const sensitivity = document.getElementById('sensitivity');
            const calibrateBtn = document.getElementById('calibrateBtn');
            const refreshBtn = document.getElementById('refresh-gnss');
            // Sanity check
            if (!startBtn || !stopBtn || !clearLogsBtn || !gainControl || 
                !sensitivity || !calibrateBtn || !refreshBtn) {
                console.error('One or more required elements not found in the DOM');
                return;
            }
        
            // Initialize GNSS monitoring
            initializeGNSSMonitoring();
        
            // Audio detection controls
            startBtn.addEventListener('click', startListening);
            stopBtn.addEventListener('click', stopListening);
            clearLogsBtn.addEventListener('click', clearLogs);
            gainControl.addEventListener('input', updateGain);
            sensitivity.addEventListener('input', updateSensitivity);
        
            calibrateBtn.addEventListener('click', async () => {
                try {
                    updateStatus(false, 'Calibrating fan noise...');
                    calibrateBtn.disabled = true;
                    const result = await calibrateFanNoise();
                    console.log('Calibration result:', result);
                    updateStatus(true, 'Calibration complete');
                    setTimeout(() => {
                        updateStatus(isListening, isListening ? 'Listening for drones...' : 'Ready to start');
                    }, 2000);
                } catch (error) {
                    console.error('Calibration failed:', error);
                    updateStatus(false, 'Calibration failed');
                } finally {
                    calibrateBtn.disabled = false;
                }
            });
        
            // Other initial UI updates
            connectWebSocket();
            updateCurrentTime();
            updateStatus(false, 'Connecting to server...');
            updateVisualizations();
            updateGNSSStatus('USB GPS not connected', false);
        
            // Session restoration
            getCurrentSession().then(session => {
                if (session?.is_active) {
                    currentSessionId = session.session_id;
                    updateStatus(true, 'Resumed listening session');
                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                    isListening = true;
                    analysisStartTime = new Date(session.start_time);
        
                    getDetections(currentSessionId).then(detections => {
                        if (detections.length === 0) {
                            addMockDetection(150, 0.82, 3);
                            addMockDetection(220, 0.65, 2);
                        } else {
                            detections.reverse().forEach(det => addDetectionLog(det));
                        }
                    });
                } else {
                    updateStatus(false, 'Ready to start');
                }
            });
        });
        
        const gnssStatus = {
            connectionType: null,
            device: null,
            usbReadInterval: null,
            webSocketReconnectAttempts: 0,
            maxReconnectAttempts: 5,
            baseReconnectDelay: 2000,
            isExplicitDisconnect: false,
            lastHeartbeat: null,
            heartbeatInterval: null,
            gps: { inView: 0, total: 32, snr: [], cn0: [] },
            glonass: { inView: 0, total: 24, snr: [], cn0: [] },
            galileo: { inView: 0, total: 30, snr: [], cn0: [] },
            beidou: { inView: 0, total: 35, snr: [], cn0: [] },
            lastUpdate: null
        };
                
        let gnssWebSocket = null;
        
        function initializeGNSSMonitoring() {
            // Check if required elements exist
            const requiredElements = [
                 'refresh-gnss', 'gnss-status-text',
                'gnss-connection-indicator', 'gnss-last-update',
                'satelliteComparisonChart'
            ];
            
            for (const id of requiredElements) {
                if (!document.getElementById(id)) {
                    console.error(`Missing required element: #${id}`);
                    return;
                }
            }

            // Set up refresh button
            document.getElementById('refresh-gnss').addEventListener('click', refreshGNSSData);

            // Start with WebSocket by default
            gnssStatus.isExplicitDisconnect = false;
            initGNSSWebSocket();
        }
        async function openWithRetry(device, maxRetries = 2) {
            let lastError;
            for (let i = 0; i < maxRetries; i++) {
                try {
                    await device.open();
                    return; // Success
                } catch (error) {
                    lastError = error;
                    console.warn(`Open attempt ${i + 1} failed:`, error);
                    await new Promise(resolve => setTimeout(resolve, 200 * (i + 1))); // Backoff delay
                }
            }
            throw lastError;
        }
        
        async function claimInterface(device, interfaceNumber, alternateSetting) {
            try {
                if (device.detachKernelDriver) {
                    try {
                        await device.detachKernelDriver(interfaceNumber);
                        console.log(`Detached kernel driver from IF#${interfaceNumber}`);
                    } catch (err) {
                        console.warn(`Could not detach kernel driver from IF#${interfaceNumber}:`, err);
                    }
                }
        
                await device.claimInterface(interfaceNumber);
                if (alternateSetting !== 0) {
                    await device.selectAlternateInterface(interfaceNumber, alternateSetting);
                }
            } catch (error) {
                throw new Error(`Failed to claim IF#${interfaceNumber}: ${error.message}`);
            }
        }
                
        // Main auto-connect function
        async function autoConnectGNSS() {
            if (gnssStatus.autoConnectAttempts >= gnssStatus.maxAutoConnectAttempts) {
                console.log('Max auto-connect attempts reached');
                return;
            }
        
            gnssStatus.autoConnectAttempts++;
            updateGNSSStatus("Attempting auto-connect...", false);
        
            // Try USB first if available
            if (await isUSBDeviceAvailable()) {
                try {
                    await connectUSB();
                    return; // Success
                } catch (usbError) {
                    console.log('USB connection failed:', usbError.message);
                }
            }
        
            // Fall back to WebSocket
            try {
                await connectWebSocket();
            } catch (wsError) {
                console.error('WebSocket connection failed:', wsError);
                // Schedule retry with exponential backoff
                const delay = Math.min(30000, 2000 * Math.pow(2, gnssStatus.autoConnectAttempts));
                setTimeout(autoConnectGNSS, delay);
            }
        }
        
// Enhanced USB Connection
async function connectUSB() {
    try {
        const devices = await navigator.usb.getDevices();
        if (devices.length === 0) {
            throw new Error('No authorized USB devices');
        }

        const device = devices[0];
        await initializeDevice(device);
        gnssStatus.device = device;
        gnssStatus.connectionType = 'usb';
        updateGNSSStatus("Connected via USB", true);
        
        // Setup monitoring
        setupUSBMontoring();
        return true;
    } catch (error) {
        console.error('USB connection failed:', error);
        await cleanupUSBConnection();
        throw error;
    }
}

// Setup USB Monitoring
function setupUSBMontoring() {
    // Watch for device disconnect
    gnssStatus.usbWatchdog = setInterval(async () => {
        if (!gnssStatus.device?.opened) {
            console.log('USB device disconnected');
            await cleanupUSBConnection();
            autoConnectGNSS();
        }
    }, 5000);
    
    // Watch for new connect events
    navigator.usb.addEventListener('connect', async () => {
        if (gnssStatus.connectionType !== 'usb') {
            autoConnectGNSS();
        }
    });
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Start auto-connection
    autoConnectGNSS();
    initializeCombinedSNRChart(); 
    initializeCN0Chart();
    initializeAnomalyChart();
    // Setup refresh button
    document.getElementById('refresh-gnss').addEventListener('click', () => {
        if (gnssStatus.connectionType === 'usb') {
            connectUSB();
        } else {
            connectWebSocket();
        }
    });

});



// USB Device Detection
        async function isUSBDeviceAvailable() {
            if (!navigator.usb) return false;
            
            try {
                const devices = await navigator.usb.getDevices();
                return devices.length > 0;
            } catch (error) {
                console.log('USB device check failed:', error);
                return false;
            }
        }

        function setupAutoReconnect() {
            if (gnssStatus.connectionType === 'usb') {
                // USB reconnection will happen through tryUSBConnection()
            } else {
                // WebSocket auto-reconnect
                setTimeout(() => {
                    if (!gnssStatus.connectionType) {
                        initGNSSWebSocket();
                    }
                }, 2000);
            }
        }
                
        async function initializeDevice(device) {
            try {
                await openWithRetry(device);
        
                // Device reset with recovery time
                if (device.reset) {
                    try {
                        await device.reset();
                        console.log('Device reset performed');
                        await new Promise(resolve => setTimeout(resolve, 1500)); // Increased recovery time
                    } catch (resetError) {
                        console.warn('Device reset failed:', resetError);
                    }
                }
        
                // Configuration selection with fallback
                if (!device.configuration) {
                    try {
                        await device.selectConfiguration(1);
                    } catch (configError) {
                        console.warn('Failed to select configuration:', configError);
                        // Try alternative configuration if available
                        if (device.configurations.length > 1) {
                            try {
                                await device.selectConfiguration(2);
                            } catch (altConfigError) {
                                console.warn('Failed alternate configuration:', altConfigError);
                            }
                        }
                    }
                }
        
                // u-blox specific configuration (if u-blox device)
                if (device.vendorId === 0x1546) { // u-blox vendor ID
                    try {
                        console.log('Configuring u-blox device...');
                        
                        // Enable NMEA output on all ports
                        const configPayload = new Uint8Array([
                            0x06, 0x00, // UBX-CFG-PRT message class/id
                            0x14, 0x00, // Length
                            0x01, 0x00, 0x00, 0x00, // Port ID (USB)
                            0x00, 0x00, 0x00, 0x00, // Reserved
                            0xFF, 0xFF, 0x00, 0x00, // TX ready
                            0x00, 0x00, 0x00, 0x00, // Mode (USB doesn't use baud rate)
                            0x03, 0x00, // InProtoMask (UBX + NMEA)
                            0x03, 0x00, // OutProtoMask (UBX + NMEA)
                            0x00, 0x00, // Flags
                            0x00, 0x00  // Reserved
                        ]);
                        
                        await device.controlTransferOut({
                            requestType: 'vendor',
                            recipient: 'device',
                            request: 0x00, // Send UBX message
                            value: 0x0000,
                            index: 0x0000
                        }, configPayload);
                        
                        // Enable GSV messages (satellite view)
                        const gsvConfig = new Uint8Array([
                            0x06, 0x01, // UBX-CFG-MSG message class/id
                            0x08, 0x00, // Length
                            0xF0, 0x03, // NMEA GSV message class/id
                            0x01,       // Rate on USB (1 = every cycle)
                            0x00, 0x00, 0x00, 0x00 // Reserved
                        ]);
                        
                        await device.controlTransferOut({
                            requestType: 'vendor',
                            recipient: 'device',
                            request: 0x00,
                            value: 0x0000,
                            index: 0x0000
                        }, gsvConfig);
                        
                        console.log('u-blox configuration sent');
                        
                        // Short delay to allow configuration to take effect
                        await new Promise(resolve => setTimeout(resolve, 500));
                    } catch (configError) {
                        console.warn('u-blox configuration failed:', configError);
                        // Continue even if configuration fails - device might have defaults
                    }
                }
        
                // Interface and endpoint selection
                let interfaceClaimed = false;
                for (const iface of device.configuration.interfaces) {
                    for (const alt of iface.alternates) {
                        const inEndpoint = alt.endpoints.find(ep => 
                            ep.direction === 'in' && ep.type === 'bulk'
                        );
        
                        if (!inEndpoint) continue;
        
                        try {
                            await claimInterface(device, iface.interfaceNumber, alt.alternateSetting);
                            console.log(`Claimed IF#${iface.interfaceNumber}, Alt#${alt.alternateSetting}, ` +
                                       `Endpoint ${inEndpoint.endpointNumber}`);
                            
                            // Additional check for u-blox devices
                            if (device.vendorId === 0x1546) {
                                // Clear any buffered data
                                try {
                                    await device.transferIn(inEndpoint.endpointNumber, 256);
                                } catch (clearError) {
                                    console.warn('Buffer clear failed:', clearError);
                                }
                            }
                            
                            gnssStatus.device = device;
                            startUSBReading(device, iface.interfaceNumber, inEndpoint.endpointNumber);
                            interfaceClaimed = true;
                            break;
                        } catch (claimError) {
                            console.warn(`Failed to claim IF#${iface.interfaceNumber}:`, claimError);
                            continue;
                        }
                    }
                    if (interfaceClaimed) break;
                }
        
                if (!interfaceClaimed) {
                    throw new Error('No usable interface with IN endpoint found');
                }
        
                // Start a watchdog timer for USB connection
                gnssStatus.usbWatchdog = setInterval(() => {
                    if (!device.opened) {
                        console.warn('USB device disconnected unexpectedly!');
                        cleanupUSBConnection();
                        initGNSSWebSocket(); // Fall back to WebSocket
                    }
                }, 5000);
        
            } catch (error) {
                console.error('Device initialization failed:', error);
                if (device.opened) {
                    try {
                        await device.close();
                    } catch (closeError) {
                        console.warn('Error closing device:', closeError);
                    }
                }
                throw error;
            }
        }
        
        
        function startUSBReading(device, interfaceNumber, endpointNumber) {
            if (gnssStatus.usbReadInterval) {
                clearInterval(gnssStatus.usbReadInterval);
            }
        
            const textDecoder = new TextDecoder('ascii', { stream: true });
            let buffer = '';
            const nmeaRegex = /\$[A-Z]{5}[^$]*?\*[0-9A-F]{2}/g;
            let errorCount = 0;
            const maxErrors = 3;
        
            gnssStatus.usbReadInterval = setInterval(async () => {
                try {
                    const result = await device.transferIn(endpointNumber, 256);
                    //console.log('USB transfer result:', result);
        
                    if (result.status === 'ok' && result.data) {
                        const rawData = textDecoder.decode(result.data, { stream: true });
                        //console.log('Raw USB data:', rawData);
                        buffer += rawData;
        
                        // Process complete sentences
                        let match;
                        while ((match = nmeaRegex.exec(buffer)) !== null) {
                            const sentence = match[0].trim();
                            const parsed = parseNMEASentence(sentence);
                            if (parsed) {
                                //console.log('Parsed NMEA:', parsed);
                                processGNSSUpdate({
                                    systems: {
                                        [parsed.system]: {
                                            in_view: parsed.totalSats,
                                            snr: parsed.snr
                                        }
                                    }
                                });
                            }
                        }
        
                        // Keep only the leftover incomplete sentence (if any)
                        const lastAsteriskIndex = buffer.lastIndexOf('*');
                        if (lastAsteriskIndex !== -1 && lastAsteriskIndex + 3 <= buffer.length) {
                            buffer = buffer.slice(lastAsteriskIndex + 3);
                        }
        
                        errorCount = 0; // reset error counter on success
                    } else {
                        console.warn('USB transfer status not OK');
                    }
                } catch (error) {
                    console.error('USB read error:', error);
                    errorCount++;
                    if (errorCount >= maxErrors) {
                        updateGNSSStatus('USB read error - disconnecting', false);
                        clearInterval(gnssStatus.usbReadInterval);
                        gnssStatus.usbReadInterval = null;
                        cleanupUSBConnection();
                        initGNSSWebSocket();
                    } else {
                        console.log(`USB read error count: ${errorCount}, retrying...`);
                    }
                }
            }, 100);  // 100ms interval as a balance
        }
        

        function initGNSSWebSocket() {
            if (gnssWebSocket) return;
        
            const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
            const wsUrl = `${protocol}://${window.location.host}/ws/gnss`;
            
            gnssWebSocket = new WebSocket(wsUrl);
        
            gnssWebSocket.onopen = () => {
                gnssStatus.connectionType = 'websocket';
                updateGNSSStatus('Connected via WebSocket', true, 'websocket');
            };
        
            gnssWebSocket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'gnss_data') {
                        processGNSSUpdate(data.payload);
                    }
                } catch (error) {
                    console.error('Error parsing message:', error);
                }
            };
        
            gnssWebSocket.onclose = () => {
                if (gnssStatus.connectionType === 'websocket') {
                    updateGNSSStatus("Disconnected", false);
                    setTimeout(initGNSSWebSocket, 2000);
                }
            };
        
            gnssWebSocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                gnssWebSocket.close();
            };
        }
        function cleanupWebSocketConnection() {
            gnssStatus.isExplicitDisconnect = true;
            
            if (gnssStatus.heartbeatInterval) {
                clearInterval(gnssStatus.heartbeatInterval);
                gnssStatus.heartbeatInterval = null;
            }
            
            if (gnssWebSocket) {
                gnssWebSocket.close();
                gnssWebSocket = null;
            }
            
            gnssStatus.connectionType = null;
            updateGNSSStatus("WebSocket disconnected", false);
        }
        
        
async function cleanupUSBConnection() {
    if (gnssStatus.usbReadInterval) {
        clearInterval(gnssStatus.usbReadInterval);
        gnssStatus.usbReadInterval = null;
    }
    
    if (gnssStatus.device) {
        try {
            // Graceful shutdown
            if (gnssStatus.device.opened) {
                await gnssStatus.device.releaseInterface(gnssStatus.device.configuration.interfaces[0].interfaceNumber);
                await gnssStatus.device.close();
            }
        } catch (error) {
            console.error('Cleanup error:', error);
        } finally {
            gnssStatus.device = null;
        }
    }
    
    updateGNSSStatus('USB disconnected', false);
    setupAutoReconnect();
}

        function checkHeartbeat() {
            if (!gnssWebSocket) return;
            
            if (Date.now() - gnssStatus.lastHeartbeat > 15000) {
                try {
                    gnssWebSocket.send(JSON.stringify({ type: 'ping' }));
                } catch (e) {
                    console.error('Ping failed:', e);
                }
            }
            
            if (Date.now() - gnssStatus.lastHeartbeat > 30000) {
                console.log('Connection stale, reconnecting...');
                initGNSSWebSocket();
            }
        }
        
        
        function refreshGNSSData() {
            if (gnssStatus.connectionType === 'websocket' && gnssWebSocket) {
                gnssWebSocket.send(JSON.stringify({ command: 'refresh' }));
            } else if (gnssStatus.connectionType === 'usb' && gnssStatus.device) {
                updateGNSSDisplay();
            }
        }
        
        function processGNSSUpdate(data) {
            if (!data?.systems) return;
            
            if (data.systems.gps) {
                gnssStatus.gps.inView = data.systems.gps.in_view || 0;
                gnssStatus.gps.snr = data.systems.gps.snr || [];
                gnssStatus.gps.cn0 = data.systems.gps.cn0 || [];
            }
            if (data.systems.glonass) {
                gnssStatus.glonass.inView = data.systems.glonass.in_view || 0;
                gnssStatus.glonass.snr = data.systems.glonass.snr || [];
                gnssStatus.glonass.cn0 = data.systems.glonass.cn0 || [];
            }
            if (data.systems.galileo) {
                gnssStatus.galileo.inView = data.systems.galileo.in_view || 0;
                gnssStatus.galileo.snr = data.systems.galileo.snr || [];
                gnssStatus.galileo.cn0 = data.systems.galileo.cn0 || [];
            }
            if (data.systems.beidou) {
                gnssStatus.beidou.inView = data.systems.beidou.in_view || 0;
                gnssStatus.beidou.snr = data.systems.beidou.snr || [];
                gnssStatus.beidou.cn0 = data.systems.beidou.cn0 || [];
            }
            
            gnssStatus.lastUpdate = Date.now();
            updateGNSSDisplay();
        }

        function updateGNSSStatus(text, isConnected, connectionType = null) {
            const statusElement = document.getElementById('gnss-status-text');
            const indicator = document.getElementById('gnss-connection-indicator');
            
            gnssStatus.connectionType = connectionType;
            
            if (statusElement) statusElement.textContent = text;
            if (indicator) {
                indicator.className = isConnected ? 'status-indicator connected' : 'status-indicator disconnected';
            }
            
            if (connectionType) {
                const mainStatus = document.getElementById('connection-status');
                if (mainStatus) {
                    mainStatus.textContent = `GNSS: ${text}`;
                    mainStatus.style.backgroundColor = isConnected ? '#4cc9f0' : '#ffc107';
                }
            }
        }
        

    // Helper function to get system-specific colors
function getSystemColor(system, border = false) {
    const colors = {
        gps: border ? 'rgba(65, 140, 216, 1)' : 'rgba(65, 140, 216, 0.7)',
        glonass: border ? 'rgba(234, 84, 85, 1)' : 'rgba(234, 84, 85, 0.7)',
        galileo: border ? 'rgba(75, 192, 192, 1)' : 'rgba(75, 192, 192, 0.7)',
        beidou: border ? 'rgba(153, 102, 255, 1)' : 'rgba(153, 102, 255, 0.7)'
    };
    return colors[system] || (border ? 'rgba(255, 255, 255, 1)' : 'rgba(255, 255, 255, 0.7)');
}

        

// Initialize Anomalous SNR Drop Chart (with consistent colors)
function initializeAnomalyChart() {
    const ctx = document.getElementById('snrAnomalyChart').getContext('2d');
    gnssStatus.anomalyChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['GPS', 'GLONASS', 'Galileo', 'BeiDou'],
            datasets: [{
                label: 'SNR Drop Duration (s)',
                data: [0, 0, 0, 0],
                backgroundColor: [
                    getSystemColor('gps'),      // GPS (blue)
                    getSystemColor('glonass'),  // GLONASS (red)
                    getSystemColor('galileo'), // Galileo (teal)
                    getSystemColor('beidou')    // BeiDou (purple)
                ],
                borderColor: [
                    getSystemColor('gps', true),      // GPS (blue border)
                    getSystemColor('glonass', true),  // GLONASS (red border)
                    getSystemColor('galileo', true),  // Galileo (teal border)
                    getSystemColor('beidou', true)    // BeiDou (purple border)
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: { 
                    beginAtZero: true,
                    title: { text: 'Duration (s)' }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: (ctx) => 
                            `${ctx.raw}s SNR drop (${ctx.raw > 10 ? 'Jamming Likely' : 'Normal'})`
                    }
                }
            }
        }
    });
}


// Track SNR drop duration per system
const snrDropDuration = { gps: 0, glonass: 0, galileo: 0, beidou: 0 };

// Initialize C/N₀ Trend Chart
function initializeCN0Chart() {
    const ctx = document.getElementById('cn0TrendChart').getContext('2d');
    gnssStatus.cn0Chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Array.from({ length: 30 }, (_, i) => `${30 - i}s`), // Last 30 seconds
            datasets: [
                {
                    label: 'Avg C/N₀ (dB-Hz)',
                    data: Array(30).fill(0),
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    tension: 0.3,
                    fill: true,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { 
                    min: 20, 
                    max: 50, 
                    title: { display: true, text: 'C/N₀ (dB-Hz)' },
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    ticks: { color: '#eee' }
                },
                x: { 
                    title: { display: true, text: 'Time' },
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    ticks: { color: '#eee' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#eee' }
                },
                annotation: {
                    annotations: {
                        jamThreshold: {
                            type: 'line',
                            yMin: 30,
                            yMax: 30,
                            borderColor: 'red',
                            borderWidth: 2,
                            borderDash: [6, 6],
                            label: {
                                content: 'Jamming Suspected',
                                enabled: true,
                                position: 'end',
                                backgroundColor: 'rgba(255,0,0,0.7)',
                                color: '#fff'
                            }
                        }
                    }
                }
            }
        }
    });
}

// Update C/N₀ Chart with average across all systems
function updateCN0Chart() {
    if (!gnssStatus.cn0Chart) return;

    const systems = ['gps', 'glonass', 'galileo', 'beidou'];
    let totalCN0 = 0, count = 0;

    systems.forEach(sys => {
        const cn0Values = gnssStatus[sys].cn0 || [];
        cn0Values.forEach(val => {
            if (val > 0) {
                totalCN0 += val;
                count++;
            }
        });
    });

    const avgCN0 = count > 0 ? +(totalCN0 / count).toFixed(1) : 0;

    gnssStatus.cn0Chart.data.datasets[0].data.shift();
    gnssStatus.cn0Chart.data.datasets[0].data.push(avgCN0);
    gnssStatus.cn0Chart.update();
}


function updateAnomalyChart() {
    if (!gnssStatus.anomalyChart) return;
    
    const systems = ['gps', 'glonass', 'galileo', 'beidou'];
    const thresholds = { gps: 35, glonass: 33, galileo: 37, beidou: 36 };
    
    systems.forEach((sys, idx) => {
        const avgSNR = gnssStatus[sys].snr?.reduce((a, b) => a + b, 0) / gnssStatus[sys].snr?.length || 0;
        
        // Update drop duration
        if (avgSNR < thresholds[sys]) {
            snrDropDuration[sys]++;
            // Change to red if drop >10s, else use system color
            gnssStatus.anomalyChart.data.datasets[0].backgroundColor[idx] = 
                snrDropDuration[sys] > 10 ? 'rgba(255, 99, 132, 0.7)' : getSystemColor(sys);
        } else {
            snrDropDuration[sys] = 0;
            gnssStatus.anomalyChart.data.datasets[0].backgroundColor[idx] = getSystemColor(sys);
        }
        
        gnssStatus.anomalyChart.data.datasets[0].data[idx] = snrDropDuration[sys];
    });
    
    gnssStatus.anomalyChart.update();
}








function updateGNSSDisplay() {
    // Update satellite counts and signal strengths
    ['gps', 'glonass', 'galileo', 'beidou'].forEach(system => {
        const element = document.getElementById(`gnss-${system}-status`);
        if (!element) return;
        
        const systemData = gnssStatus[system] || {};
        
        // Update satellite count
        const countElement = element.querySelector('.satellite-count');
        if (countElement) {
            countElement.textContent = `${systemData.inView || 0}/${systemData.total || 0}`;
        }
        
        // Update signal strength meter based on average SNR
        const strengthElement = element.querySelector('.signal-strength');
        if (strengthElement) {
            const validSNRs = (systemData.snr || []).filter(s => s > 0);
            const avgSNR = validSNRs.length > 0 ? 
                Math.round(validSNRs.reduce((a, b) => a + b, 0) / validSNRs.length) : 0;
            
            const widthPercent = Math.min(100, avgSNR * 2); // Scale 0-50 dB-Hz to 0-100%
            strengthElement.style.width = `${widthPercent}%`;
            strengthElement.title = `Avg SNR: ${avgSNR} dB-Hz`;

            // Color coding based on signal quality
            if (avgSNR > 35) {
                strengthElement.style.backgroundColor = '#4CAF50'; // Good
            } else if (avgSNR > 20) {
                strengthElement.style.backgroundColor = '#FFC107'; // Fair
            } else {
                strengthElement.style.backgroundColor = '#F44336'; // Poor
            }
        }
    });
    
    // Update all relevant charts
    updateSatelliteChart();
    updateCombinedSNRChart();
    updateCN0Chart();
    updateAnomalyChart();
    
    // Update last update time
    const lastUpdateElement = document.getElementById('gnss-last-update');
    if (lastUpdateElement) {
        lastUpdateElement.textContent = new Date(gnssStatus.lastUpdate).toLocaleTimeString();
    }
}

function initializeCombinedSNRChart() {
    const ctx = document.getElementById('satelliteSNRChart').getContext('2d');

    gnssStatus.snrChartPaused = false;  // Add pause flag

    gnssStatus.combinedSNRChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [], // Satellite PRNs will go here
            datasets: [
                {
                    label: 'GPS',
                    backgroundColor: 'rgba(65, 140, 216, 0.7)',
                    borderColor: 'rgba(65, 140, 216, 1)',
                    borderWidth: 1,
                    data: []
                },
                {
                    label: 'GLONASS',
                    backgroundColor: 'rgba(234, 84, 85, 0.7)',
                    borderColor: 'rgba(234, 84, 85, 1)',
                    borderWidth: 1,
                    data: []
                },
                {
                    label: 'Galileo',
                    backgroundColor: 'rgba(75, 192, 192, 0.7)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1,
                    data: []
                },
                {
                    label: 'BeiDou',
                    backgroundColor: 'rgba(153, 102, 255, 0.7)',
                    borderColor: 'rgba(153, 102, 255, 1)',
                    borderWidth: 1,
                    data: []
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 50,
                    title: {
                        display: true,
                        text: 'SNR (dB-Hz)',
                        color: '#eee',
                        font: { size: 10 }
                    },
                    ticks: {
                        color: '#ccc',
                        font: { size: 9 }
                    },
                    grid: {
                        color: 'rgba(255,255,255,0.1)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Satellite PRN',
                        color: '#eee',
                        font: { size: 10 }
                    },
                    ticks: {
                        color: '#ccc',
                        font: { size: 8 }
                    },
                    grid: {
                        color: 'rgba(255,255,255,0.1)'
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#eee',
                        font: { size: 10 },
                        boxWidth: 12
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            return `${context.dataset.label}: ${context.raw} dB-Hz`;
                        }
                    }
                }
            }
        }
    });

    // Toggle pause on click (same as anomaly chart logic)
    document.getElementById('satelliteSNRChart').addEventListener('click', () => {
        gnssStatus.snrChartPaused = !gnssStatus.snrChartPaused;
    });
}

// Update the combined SNR chart
function updateCombinedSNRChart() {
    if (!gnssStatus.combinedSNRChart) return;

    const systems = ['gps', 'glonass', 'galileo', 'beidou'];
    const labels = [];
    const datasets = [];

    systems.forEach((system, index) => {
        const snrValues = gnssStatus[system].snr || [];
        
        // Create satellite PRN labels for this system
        const systemLabels = [];
        for (let i = 0; i < snrValues.length; i++) {
            const prefix = system === 'gps' ? 'G' : 
                          system === 'glonass' ? 'R' : 
                          system === 'galileo' ? 'E' : 'C';
            systemLabels.push(`${prefix}${(i+1).toString().padStart(2, '0')}`);
        }

        // Add to combined labels if not already present
        systemLabels.forEach(label => {
            if (!labels.includes(label)) {
                labels.push(label);
            }
        });

        // Update dataset for this system
        gnssStatus.combinedSNRChart.data.datasets[index].data = snrValues;
    });

    // Update labels
    gnssStatus.combinedSNRChart.data.labels = labels;
    gnssStatus.combinedSNRChart.update();
}


function updateSatelliteChart() {
    const gpsPercent = Math.round((gnssStatus.gps.inView / gnssStatus.gps.total) * 100);
    const glonassPercent = Math.round((gnssStatus.glonass.inView / gnssStatus.glonass.total) * 100);
    const galileoPercent = Math.round((gnssStatus.galileo.inView / gnssStatus.galileo.total) * 100);
    const beidouPercent = Math.round((gnssStatus.beidou.inView / gnssStatus.beidou.total) * 100);
    
    satelliteComparisonChart.data.datasets[0].data = [
        gpsPercent,
        glonassPercent,
        galileoPercent,
        beidouPercent
    ];
    
    satelliteComparisonChart.update();
}

// Robust NMEA Parser with Error Recovery
function parseNMEASentence(sentence) {
    if (!sentence || typeof sentence !== 'string') return null;
    
    // Basic validation
    if (!sentence.startsWith('$') || sentence.length < 6) return null;
    
    try {
        // Extract checksum if present
        const checksumIndex = sentence.indexOf('*');
        let content, checksum;
        
        if (checksumIndex > 0) {
            content = sentence.slice(1, checksumIndex);
            checksum = sentence.slice(checksumIndex + 1);
            
            // Validate checksum
            let calculatedChecksum = 0;
            for (let i = 0; i < content.length; i++) {
                calculatedChecksum ^= content.charCodeAt(i);
            }
            
            if (checksum && parseInt(checksum, 16) !== calculatedChecksum) {
                console.warn('Invalid NMEA checksum:', sentence);
                return null;
            }
        } else {
            content = sentence.slice(1);
        }
        
        const parts = content.split(',');
        if (parts.length < 1) return null;
        
        const talker = parts[0].slice(0, 2);
        const sentenceType = parts[0].slice(2);
        
        // Only process GSV messages for satellite data
        if (sentenceType === 'GSV') {
            const systemMap = { 
                GP: 'gps', 
                GL: 'glonass', 
                GA: 'galileo', 
                BD: 'beidou',
                GN: 'gnss' // Generic GNSS
            };
            
            const system = systemMap[talker];
            if (!system) return null;
            
            // Basic validation
            if (parts.length < 4) return null;
            
            const totalMsgs = parseInt(parts[1], 10);
            const msgNum = parseInt(parts[2], 10);
            const totalSats = parseInt(parts[3], 10);
            
            if (isNaN(totalMsgs) || isNaN(msgNum) || isNaN(totalSats)) {
                return null;
            }
            
            // Extract SNR values (every 4th field starting at index 7)
            const snr = [];
            for (let i = 7; i < parts.length && i < 31; i += 4) {
                const snrValue = parseInt(parts[i], 10);
                if (!isNaN(snrValue) && snrValue > 0) {
                    snr.push(snrValue);
                }
            }
            
            return {
                type: 'satellite_data',
                system,
                totalSats,
                snr,
                timestamp: Date.now()
            };
        }
        
        return null;
        
    } catch (error) {
        console.warn('Error parsing NMEA:', error);
        return null;
    }
}

// Initialize Satellite Comparison Chart
const satelliteComparisonCtx = document.getElementById('satelliteComparisonChart').getContext('2d');
const satelliteComparisonChart = new Chart(satelliteComparisonCtx, {
    type: 'bar',
    data: {
        labels: ['GPS', 'GLONASS', 'Galileo', 'BeiDou'],
        datasets: [{
            label: 'Satellite Coverage',
            data: [0, 0, 0, 0],
            backgroundColor: [
                'rgba(65, 140, 216, 0.7)',
                'rgba(234, 84, 85, 0.7)',
                'rgba(75, 192, 192, 0.7)',
                'rgba(153, 102, 255, 0.7)'
            ],
            borderColor: [
                'rgba(65, 140, 216, 1)',
                'rgba(234, 84, 85, 1)',
                'rgba(75, 192, 192, 1)',
                'rgba(153, 102, 255, 1)'
            ],
            borderWidth: 1
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        devicePixelRatio: 2, // Double resolution for sharper chart

        plugins: {
            legend: { display: false },
            tooltip: {
                callbacks: {
                    label: function(context) {
                        const system = context.label.toLowerCase();
                        const inView = gnssStatus[system].inView;
                        const total = gnssStatus[system].total;
                        return `${context.label}: ${context.raw}% (${inView}/${total} satellites)`;
                    }
                }
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                max: 100,
                title: { display: true, text: 'Coverage Percentage' },
                ticks: { callback: value => value + '%' }
            }
        }
    }
});

// main.js
import SDRController from './sdr-module.js';

document.addEventListener('DOMContentLoaded', function() {
  // Get all UI elements
  const elements = {
    sdrStartBtn: document.getElementById('sdrStartBtn'),
    sdrStopBtn: document.getElementById('sdrStopBtn'),
    sdrRecordBtn: document.getElementById('sdrRecordBtn'),
    sdrFrequency: document.getElementById('sdrFrequency'),
    sdrMode: document.getElementById('sdrMode'),
    sdrBandwidth: document.getElementById('sdrBandwidth'),
    sdrLnaGain: document.getElementById('sdrLnaGain'),
    sdrMixerGain: document.getElementById('sdrMixerGain'),
    sdrAgc: document.getElementById('sdrAgc'),
    sdrSquelch: document.getElementById('sdrSquelch'),
    sdrVolume: document.getElementById('sdrVolume'),
    sdrAudioFilter: document.getElementById('sdrAudioFilter'),
    sdrAudioOutput: document.getElementById('sdrAudioOutput'),
    waterfallCanvas: document.getElementById('waterfallCanvas'),
    spectrumCanvas: document.getElementById('spectrumCanvas'),
    sdrStatusText: document.getElementById('sdr-status-text'),
    sdrConnectionIndicator: document.getElementById('sdr-connection-indicator'),
    sdrDeviceInfo: document.getElementById('sdr-device-info'), // Fixed typo here (was 'device')
    sdrVolumeValue: document.getElementById('sdrVolumeValue'),
    sdrLnaValue: document.getElementById('sdrLnaValue'),
    sdrMixerValue: document.getElementById('sdrMixerValue'),
    sdrAgcText: document.getElementById('sdrAgcText'),
    sdrSquelchValue: document.getElementById('sdrSquelchValue'),
    sdrMemoryBank: document.getElementById('sdrMemoryBank'),
    sdrAddMemory: document.getElementById('sdrAddMemory'),
    sdrFreqUp: document.getElementById('sdrFreqUp'),
    sdrFreqDown: document.getElementById('sdrFreqDown')
  };

  // Initialize SDR controller
  const sdrController = new SDRController('http://localhost:8000');
  
  // Set up callbacks
  sdrController.onStatusUpdate((connected) => {
    if (connected) {
      elements.sdrConnectionIndicator.className = 'status-indicator connected';
      elements.sdrStatusText.textContent = 'Connected';
      elements.sdrStartBtn.disabled = true;
      elements.sdrStopBtn.disabled = false;
      elements.sdrRecordBtn.disabled = false;
      
      // Ensure canvases are properly initialized when connected
      if (elements.waterfallCanvas && elements.spectrumCanvas) {
        sdrController.initCanvases();
      }
    } else {
      elements.sdrConnectionIndicator.className = 'status-indicator disconnected';
      elements.sdrStatusText.textContent = 'Disconnected';
      elements.sdrStartBtn.disabled = false;
      elements.sdrStopBtn.disabled = true;
      elements.sdrRecordBtn.disabled = true;
    }
  });

  sdrController.onError((message) => {
    const modal = document.getElementById('sdrErrorModal');
    const errorText = document.getElementById('sdrErrorText');
    if (modal && errorText) {
      errorText.textContent = message;
      modal.hidden = false;
    } else {
      console.error('SDR Error:', message);
    }
  });

  // Initialize the SDR controller with UI elements
  sdrController.init(elements);

  // Make sure canvases have proper dimensions
  function resizeCanvases() {
    const container = document.querySelector('.sdr-display-container');
    if (container && elements.waterfallCanvas && elements.spectrumCanvas) {
      const width = container.clientWidth;
      elements.waterfallCanvas.width = width;
      elements.spectrumCanvas.width = width;
      
      // Reinitialize displays if SDR is running
      if (sdrController.sdrRunning) {
        sdrController.initCanvases();
      }
    }
  }

  // Initial resize
  resizeCanvases();
  
  // Handle window resize
  window.addEventListener('resize', resizeCanvases);

  // Close error modal
  document.querySelector('.close-modal')?.addEventListener('click', () => {
    const modal = document.getElementById('sdrErrorModal');
    if (modal) {
      modal.hidden = true;
    }
  });
});