class AudioProcessor extends AudioWorkletProcessor {
    process(inputs) {
        const inputData = inputs[0][0];
        if (!inputData) return true;
        
        this.port.postMessage({
            type: 'audio_data',
            data: Array.from(inputData) 
        });
        
        return true;
    }
}

registerProcessor('audio-processor', AudioProcessor);