import numpy as np
import librosa
from scipy import signal
import noisereduce as nr
from sklearn.base import BaseEstimator
import joblib
from sklearn.ensemble import RandomForestClassifier

SAMPLE_RATE = 44100
MIN_AUDIO_LENGTH = 1.0  # Minimum audio length in seconds
DRONE_FREQ_RANGE = (80, 250)  # Hz
SHAHEED_FREQ_RANGE = (120, 180)  # Hz - Specific to Shaheed drones
SHAHEED_REFERENCE_SPECTRUM = None

def load_models():
    """Load trained models with fallback to default models"""
    try:
        drone_detector = joblib.load("models/drone_detector.pkl")
        shaheed_detector = joblib.load("models/shaheed_detector.pkl")
        
        # Ensure feature order is set
        if not hasattr(drone_detector, 'feature_order'):
            drone_detector.feature_order = [
                'harmonic_ratio', 
                'f0_std', 
                'drone_band_energy',
                'f0_mean',
                'voiced_ratio',
                'energy_ratio'
            ]
            
        if not hasattr(shaheed_detector, 'feature_order'):
            shaheed_detector.feature_order = [
                'harmonic_ratio',
                'f0_std',
                'spectral_correlation',
                'mfcc_std',
                'spectral_centroid',
                'spectral_bandwidth',
                'mfcc_mean'
            ]
            
    except Exception as e:
        print(f"Error loading models: {str(e)}, using default models")
        drone_detector = MLDroneDetector()
        shaheed_detector = MLShaheedDetector()
    
    return drone_detector, shaheed_detector

class BaseDetector(BaseEstimator):
    def predict(self, features):
        raise NotImplementedError

class MLDroneDetector(BaseDetector):
    """Machine Learning-based Drone Detector"""
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_order = [
            'harmonic_ratio', 
            'f0_std', 
            'drone_band_energy',
            'f0_mean',
            'voiced_ratio',
            'energy_ratio'
        ]
    
    def fit(self, features, labels):
        """Train the model on extracted features"""
        try:
            # Convert list of dicts to 2D array
            X = np.array([[f.get(k, 0) for k in self.feature_order] for f in features])
            y = np.array(labels)
            self.model.fit(X, y)
            return self
        except Exception as e:
            print(f"Error in fitting drone detector: {str(e)}")
            raise
    
    def predict(self, X):
        """Predict using the trained model"""
        try:
            if isinstance(X, list):
                if isinstance(X[0], dict):  # List of feature dicts
                    X = np.array([[x.get(k, 0) for k in self.feature_order] for x in X])
                else:  # List of arrays
                    X = np.array(X)
            
            if isinstance(X, np.ndarray):
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                return self.model.predict(X)
            
            raise ValueError("Unsupported input type for predict()")
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return np.zeros(1)  # Return safe default

class MLShaheedDetector(BaseDetector):
    """Machine Learning-based Shaheed Detector"""
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_order = [
            'harmonic_ratio',
            'f0_std',
            'spectral_correlation',
            'mfcc_std',
            'spectral_centroid',
            'spectral_bandwidth',
            'mfcc_mean'
        ]
    
    def fit(self, features, labels):
        """Train the model on extracted features"""
        try:
            X = np.array([[f.get(k, 0) for k in self.feature_order] for f in features])
            y = np.array(labels)
            self.model.fit(X, y)
            return self
        except Exception as e:
            print(f"Error in fitting shaheed detector: {str(e)}")
            raise
    
    def predict(self, X):
        """Predict using the trained model"""
        try:
            if isinstance(X, list):
                if isinstance(X[0], dict):  # List of feature dicts
                    X = np.array([[x.get(k, 0) for k in self.feature_order] for x in X])
                else:  # List of arrays
                    X = np.array(X)
            
            if isinstance(X, np.ndarray):
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                return self.model.predict(X)
            
            raise ValueError("Unsupported input type for predict()")
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return np.zeros(1)  # Return safe default

def preprocess_audio(audio_data, sr=SAMPLE_RATE, for_shaheed=False):
    """Audio preprocessing shared between training and detection"""
    try:
        if len(audio_data) < MIN_AUDIO_LENGTH * sr:
            return None
            
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Normalize and add small epsilon to avoid divide by zero
        audio_data = librosa.util.normalize(audio_data) + 1e-10
        
        freq_range = SHAHEED_FREQ_RANGE if for_shaheed else DRONE_FREQ_RANGE
        
        # Bandpass filter with stable coefficients
        sos = signal.butter(4, [freq_range[0]-5, freq_range[1]+5], 
                          'bandpass', fs=sr, output='sos')
        audio_data = signal.sosfiltfilt(sos, audio_data)
        
        # Reduce noise
        audio_data = nr.reduce_noise(y=audio_data, sr=sr, stationary=True)
        
        return audio_data
    except Exception as e:
        print(f"Preprocessing error: {str(e)}")
        return None

def extract_features(audio_data, sr, for_shaheed=False):
    """Feature extraction shared between training and detection"""
    features = {}
    try:
        if audio_data is None or len(audio_data) < 2048:
            return None
            
        freq_range = SHAHEED_FREQ_RANGE if for_shaheed else DRONE_FREQ_RANGE
        
        # 1. Fundamental frequency features
        f0, voiced_flag, _ = librosa.pyin(
            audio_data,
            fmin=max(50, freq_range[0]-20),
            fmax=min(500, freq_range[1]+20),
            sr=sr,
            frame_length=2048
        )
        valid_f0 = f0[voiced_flag]
        features['f0_mean'] = np.mean(valid_f0) if len(valid_f0) > 0 else 0
        features['f0_std'] = np.std(valid_f0) if len(valid_f0) > 1 else 0
        features['voiced_ratio'] = np.mean(voiced_flag) if len(voiced_flag) > 0 else 0

        # 2. Harmonic features
        harmonic = librosa.effects.harmonic(audio_data, margin=4)
        features['harmonic_ratio'] = np.mean(harmonic / (np.abs(audio_data)+1e-6))
        
        # 3. Spectral features
        S = np.abs(librosa.stft(audio_data, n_fft=2048))
        freqs = librosa.fft_frequencies(sr=sr)
        drone_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        
        features['drone_band_energy'] = np.mean(S[drone_mask])
        features['total_energy'] = np.mean(S)
        features['energy_ratio'] = features['drone_band_energy'] / (features['total_energy'] + 1e-6)
        
        # Shaheed-specific features
        if for_shaheed:
            features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(S=S))
            features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(S=S))
            
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfccs)
            features['mfcc_std'] = np.std(mfccs)
            
            if SHAHEED_REFERENCE_SPECTRUM is not None:
                current_spectrum = np.mean(S, axis=1)
                min_len = min(len(current_spectrum), len(SHAHEED_REFERENCE_SPECTRUM))
                features['spectral_correlation'] = np.corrcoef(
                    current_spectrum[:min_len],
                    SHAHEED_REFERENCE_SPECTRUM[:min_len]
                )[0, 1]
            else:
                features['spectral_correlation'] = 0
                
        return features
    except Exception as e:
        print(f"Feature extraction failed: {str(e)}")
        return None