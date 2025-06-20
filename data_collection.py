import sounddevice as sd
import numpy as np
import os
from datetime import datetime
import soundfile as sf

SAMPLE_RATE = 44100
DURATION = 2  # seconds
CHANNELS = 1

DATA_DIR = "training_data"
DRONE_DIR = os.path.join(DATA_DIR, "drone")
NON_DRONE_DIR = os.path.join(DATA_DIR, "non_drone")

os.makedirs(DRONE_DIR, exist_ok=True)
os.makedirs(NON_DRONE_DIR, exist_ok=True)

def record_sample(output_dir):
    print("Recording...")
    audio_data = sd.rec(int(DURATION * SAMPLE_RATE), 
                       samplerate=SAMPLE_RATE, 
                       channels=CHANNELS,
                       dtype='float32')
    sd.wait()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"sample_{timestamp}.wav")
    sf.write(filename, audio_data, SAMPLE_RATE)
    print(f"Saved to {filename}")

def main():
    print("1. Record drone sample")
    print("2. Record non-drone sample")
    choice = input("Select (1/2): ")
    
    if choice == "1":
        record_sample(DRONE_DIR)
    elif choice == "2":
        record_sample(NON_DRONE_DIR)
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()