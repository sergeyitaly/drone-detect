import os
import numpy as np
import librosa
import soundfile as sf
import joblib
import noisereduce as nr
from scipy import signal
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import warnings
from detector_models import MLDroneDetector, MLShaheedDetector, preprocess_audio, extract_features

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration
DATA_DIR = "training_data"
DRONE_DIR = os.path.join(DATA_DIR, "drone")
SHAHEED_DIR = os.path.join(DATA_DIR, "shaheed")
NON_DRONE_DIR = os.path.join(DATA_DIR, "non_drone")
SAMPLE_RATE = 44100
DRONE_FREQ_RANGE = (80, 250)  # Hz
SHAHEED_FREQ_RANGE = (120, 180)  # Hz - Specific to Shaheed drones
MIN_AUDIO_LENGTH = 1.0  # Minimum audio length in seconds
AUGMENTATION_FACTOR = 3  # How many augmented versions to create per sample

def augment_audio(audio, sr):
    """Apply audio augmentations to increase dataset size"""
    augmented = []
    
    # Original
    augmented.append(audio)
    
    if AUGMENTATION_FACTOR >= 1:
        # Time stretching (slower)
        stretched = librosa.effects.time_stretch(audio, rate=0.8)
        if len(stretched) >= len(audio):
            augmented.append(stretched[:len(audio)])
        else:
            stretched = np.pad(stretched, (0, max(0, len(audio)-len(stretched))), mode='constant')
            augmented.append(stretched)
    
    if AUGMENTATION_FACTOR >= 2:
        # Pitch shifting (higher)
        pitched_up = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
        augmented.append(pitched_up)
    
    if AUGMENTATION_FACTOR >= 3:
        # Pitch shifting (lower)
        pitched_down = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-1)
        augmented.append(pitched_down)
    
    if AUGMENTATION_FACTOR >= 4:
        # Noise addition
        noise = np.random.normal(0, 0.002, len(audio))
        augmented.append(audio + noise)
    
    return augmented[:AUGMENTATION_FACTOR+1]

def load_audio_files(directory, label, for_shaheed=False):
    """Load and process audio files from a directory"""
    features = []
    labels = []
    
    # Get all WAV files
    file_list = sorted([f for f in os.listdir(directory) if f.lower().endswith('.wav')])
    
    if not file_list:
        print(f"Warning: No WAV files found in {directory}")
        return features, labels
        
    for filename in tqdm(file_list, desc=f"Processing {os.path.basename(directory)}"):
        try:
            filepath = os.path.join(directory, filename)
            audio, sr = sf.read(filepath)
            
            # Convert stereo to mono if needed
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
                
            # Skip files that are too short
            if len(audio) < MIN_AUDIO_LENGTH * sr:
                print(f"Skipping {filename} - too short")
                continue
                
            # Resample if needed
            if sr != SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
                sr = SAMPLE_RATE
            
            # Apply augmentations
            for augmented_audio in augment_audio(audio, sr):
                # Process the audio
                processed_audio = preprocess_audio(augmented_audio, sr, for_shaheed)
                if processed_audio is None:
                    continue
                
                # Extract features
                feats = extract_features(processed_audio, sr, for_shaheed)
                if feats:
                    features.append(feats)
                    labels.append(label)
                    
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    return features, labels

def evaluate_detector(detector, features, labels, detector_name):
    """Proper evaluation using consistent features"""
    correct = 0
    total = len(features)
    
    print(f"\n{detector_name} Evaluation:")
    print("="*50)
    
    for feat, true_label in zip(features, labels):
        try:
            # Convert features to array in correct order
            if hasattr(detector, 'feature_order') and detector.feature_order:
                X = np.array([[feat.get(k, 0) for k in detector.feature_order]])
            else:
                X = np.array([[v for v in feat.values()]])
            
            pred = detector.predict(X)[0]
            correct += (pred == true_label)
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            continue
    
    accuracy = correct / total
    print(f"\nAccuracy: {accuracy:.1%} ({correct}/{total})")
    print("="*50)
    return accuracy

def save_detector(detector, filename):
    """Save detector with error handling"""
    os.makedirs('models', exist_ok=True)
    try:
        joblib.dump(detector, filename)
        print(f"Successfully saved {filename}")
    except Exception as e:
        print(f"Error saving {filename}: {str(e)}")

def main():
    print("Building drone detection models...")
    
    # Load data with progress bars
    print("\nLoading drone samples...")
    drone_feats, drone_labels = load_audio_files(DRONE_DIR, label=1)
    
    print("\nLoading shaheed samples...")
    shaheed_feats, shaheed_labels = load_audio_files(SHAHEED_DIR, label=1, for_shaheed=True)
    
    print("\nLoading non-drone samples...")
    non_drone_feats, non_drone_labels = load_audio_files(NON_DRONE_DIR, label=0)
    
    # Verify we have enough samples
    if len(drone_feats) < 10 or len(shaheed_feats) < 10 or len(non_drone_feats) < 10:
        print("\nError: Not enough training samples")
        print(f"- Drone samples: {len(drone_feats)} (need at least 10)")
        print(f"- Shaheed samples: {len(shaheed_feats)} (need at least 10)")
        print(f"- Non-drone samples: {len(non_drone_feats)} (need at least 10)")
        return
    
    # Create and evaluate drone detector
    drone_detector = MLDroneDetector()
    print("\nTraining Drone Detector...")
    drone_detector.fit(
        drone_feats + non_drone_feats + shaheed_feats,
        [1]*len(drone_feats) + [0]*len(non_drone_feats) + [0]*len(shaheed_feats)
    )
    
    print("\nEvaluating Drone Detector...")
    drone_accuracy = evaluate_detector(
        drone_detector,
        drone_feats + non_drone_feats + shaheed_feats,
        [1]*len(drone_feats) + [0]*len(non_drone_feats) + [0]*len(shaheed_feats),
        "DroneDetector"
    )
    
    # Create and evaluate shaheed detector
    shaheed_detector = MLShaheedDetector()
    print("\nTraining Shaheed Detector...")
    shaheed_detector.fit(
        shaheed_feats + drone_feats + non_drone_feats,
        [1]*len(shaheed_feats) + [0]*len(drone_feats) + [0]*len(non_drone_feats)
    )
    
    print("\nEvaluating Shaheed Detector...")
    shaheed_accuracy = evaluate_detector(
        shaheed_detector,
        shaheed_feats + drone_feats + non_drone_feats,
        [1]*len(shaheed_feats) + [0]*len(drone_feats) + [0]*len(non_drone_feats),
        "ShaheedDetector"
    )
    
    # Save the detectors
    save_detector(drone_detector, 'models/drone_detector.pkl')
    save_detector(shaheed_detector, 'models/shaheed_detector.pkl')
    
    print("\nTraining complete:")
    print(f"- Drone detector accuracy: {drone_accuracy:.1%}")
    print(f"- Shaheed detector accuracy: {shaheed_accuracy:.1%}")
    print("\nRecommendations:")
    print("- Add more training samples (aim for 50+ per category)")
    print("- Try different machine learning models (SVM, Neural Networks)")
    print("- Collect more diverse environmental recordings")

if __name__ == "__main__":
    main()