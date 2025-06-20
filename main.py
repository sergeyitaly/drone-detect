import os
import numpy as np
import librosa
import sounddevice as sd
from fastapi import FastAPI, HTTPException, WebSocket, Depends, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import soundfile as sf
import noisereduce as nr
from scipy import signal
import joblib
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Float, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import json
import uuid
import asyncio
from threading import Thread
import psutil
from concurrent.futures import ThreadPoolExecutor
import queue
from typing import Optional
import glob
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.staticfiles import StaticFiles
import pickle
from sklearn.base import BaseEstimator
from detector_models import load_models, MLDroneDetector, MLShaheedDetector, BaseDetector, preprocess_audio, extract_features, load_models
from sklearn.preprocessing import StandardScaler
import time
import sys
from dotenv import load_dotenv
load_dotenv()

# Configuration
SAMPLE_RATE = 44100
DURATION = 2  # seconds
CHANNELS = 1
DRONE_FREQ_RANGE = (80, 250)  # Hz
SHAHEED_FREQ_RANGE = (120, 180)  # Hz - Specific to Shaheed drones
MIN_CONFIDENCE = 0.8
CPU_THRESHOLD = 95  # Skip processing if CPU usage exceeds this
CALIBRATION_DURATION = 3  # seconds for noise calibration
ACTIVE_NOISE_REDUCTION_ENABLED = True
SHAHEED_REFERENCE_SPECTRUM = None
DRONE_DETECTOR, SHAHEED_DETECTOR = load_models()

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Global state
is_listening = False
current_websocket = None
audio_stream = None
current_session_id = str(uuid.uuid4())
FAN_NOISE_PROFILE = None
AMBIENT_NOISE_PROFILE = None  # For active noise reduction
thread_message_queue = queue.Queue()
executor = ThreadPoolExecutor(max_workers=1)

class DetectionEvent(Base):
    __tablename__ = "detection_events"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    confidence = Column(Float)  
    is_drone = Column(Boolean)
    is_shaheed = Column(Boolean, nullable=True)
    frequency = Column(Float)
    model_version = Column(String)
    audio_sample_path = Column(String)
    session_id = Column(String)
    drone_type = Column(String)
    spectral_features = Column(String)

# Create tables
Base.metadata.create_all(bind=engine)

# FastAPI app
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DetectionResult(BaseModel):
    timestamp: datetime
    confidence: float
    is_drone: bool
    is_shaheed: Optional[bool] = None
    frequency: Optional[float] = None
    session_id: str = None
    drone_type: str = None
    spectral_features: dict = None

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def save_audio_sample(audio_data, sample_rate=SAMPLE_RATE):
    os.makedirs("detected_drones", exist_ok=True)
    filename = f"detected_drones/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{current_session_id}.wav"
    sf.write(filename, audio_data, sample_rate)
    return filename

def load_fan_noise_profile():
    global FAN_NOISE_PROFILE
    try:
        os.makedirs("profiles", exist_ok=True)
        if os.path.exists("profiles/fan_noise_profile_latest.npy"):
            FAN_NOISE_PROFILE = np.load("profiles/fan_noise_profile_latest.npy")
            print("Loaded latest fan noise profile")
    except Exception as e:
        print(f"Error loading fan profile: {e}")

def load_shaheed_reference_spectrum():
    global SHAHEED_REFERENCE_SPECTRUM
    try:
        os.makedirs("profiles", exist_ok=True)
        if os.path.exists("profiles/shaheed_reference_latest.npy"):
            SHAHEED_REFERENCE_SPECTRUM = np.load("profiles/shaheed_reference_latest.npy")
            print("Loaded latest Shaheed reference spectrum")
    except Exception as e:
        print(f"Error loading Shaheed reference: {e}")

def load_ambient_noise_profile():
    global AMBIENT_NOISE_PROFILE
    try:
        os.makedirs("profiles", exist_ok=True)
        if os.path.exists("profiles/ambient_noise_profile_latest.npy"):
            AMBIENT_NOISE_PROFILE = np.load("profiles/ambient_noise_profile_latest.npy")
            print("Loaded latest ambient noise profile")
    except Exception as e:
        print(f"Error loading ambient profile: {e}")

def perform_initial_calibration():
    """Run this when application starts to calibrate noise profiles"""
    print("Performing initial noise calibration...")
    try:
        # Load existing profiles first
        load_ambient_noise_profile()
        load_fan_noise_profile()
        load_shaheed_reference_spectrum()
        
        # Calibrate ambient noise if no profile exists
        if AMBIENT_NOISE_PROFILE is None:
            print("No ambient noise profile found, calibrating...")
            calibrate_ambient_noise()
        
        print("Initial calibration complete")
    except Exception as e:
        print(f"Initial calibration failed: {e}")

def calibrate_ambient_noise(duration=CALIBRATION_DURATION):
    """Calibrate the ambient noise profile"""
    global AMBIENT_NOISE_PROFILE
    print(f"Calibrating ambient noise for {duration} seconds...")
    
    try:
        # Record ambient noise
        print("Please ensure quiet environment for calibration...")
        ambient_noise = sd.rec(int(duration * SAMPLE_RATE),
                             samplerate=SAMPLE_RATE,
                             channels=1)
        sd.wait()
        AMBIENT_NOISE_PROFILE = ambient_noise[:, 0]  # Get single channel
        
        # Save the profile
        os.makedirs("profiles", exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        profile_path = f"profiles/ambient_noise_profile_{timestamp}.npy"
        np.save(profile_path, AMBIENT_NOISE_PROFILE)
        
        # Also save as latest
        latest_path = "profiles/ambient_noise_profile_latest.npy"
        np.save(latest_path, AMBIENT_NOISE_PROFILE)
        
        print(f"Ambient noise profile saved to {profile_path}")
        return AMBIENT_NOISE_PROFILE
    except Exception as e:
        print(f"Error calibrating ambient noise: {e}")
        raise

def create_fan_noise_profile(duration=CALIBRATION_DURATION, max_retries=3):
    global FAN_NOISE_PROFILE
    print("Recording fan noise profile...")
    print("Please ensure only fan noise is present during recording...")
    
    for attempt in range(max_retries):
        try:
            fan_noise = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
            sd.wait()  # Wait for recording to finish
            FAN_NOISE_PROFILE = fan_noise[:, 0]  # Get single channel
            
            os.makedirs("profiles", exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            profile_path = f"profiles/fan_noise_profile_{timestamp}.npy"
            np.save(profile_path, FAN_NOISE_PROFILE)
            
            # Also save as latest
            latest_path = "profiles/fan_noise_profile_latest.npy"
            np.save(latest_path, FAN_NOISE_PROFILE)
            
            print(f"Saved fan profile to {profile_path}")
            return FAN_NOISE_PROFILE
        
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed after {max_retries} attempts: {e}")
            print(f"Attempt {attempt + 1} failed, retrying... Error: {e}")
            time.sleep(1)  # Wait before retrying
            if sys.platform == "darwin":
                os.system("sudo killall coreaudiod 2>/dev/null")  # Reset macOS audio

def create_shaheed_reference(audio_data, sr=SAMPLE_RATE):
    try:
        audio_data = preprocess_audio(audio_data, SAMPLE_RATE)
        S = np.abs(librosa.stft(audio_data, n_fft=2048))
        SHAHEED_REFERENCE_SPECTRUM = np.mean(S, axis=1)
        
        os.makedirs("profiles", exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        profile_path = f"profiles/shaheed_reference_{timestamp}.npy"
        np.save(profile_path, SHAHEED_REFERENCE_SPECTRUM)
        
        # Also save as latest
        latest_path = "profiles/shaheed_reference_latest.npy"
        np.save(latest_path, SHAHEED_REFERENCE_SPECTRUM)
        
        print(f"Saved Shaheed reference spectrum to {profile_path}")
        return SHAHEED_REFERENCE_SPECTRUM
    except Exception as e:
        print(f"Error creating Shaheed reference: {e}")
        raise

def active_noise_reduction(audio_data):
    """Apply active noise reduction using ambient noise profile"""
    if not ACTIVE_NOISE_REDUCTION_ENABLED or AMBIENT_NOISE_PROFILE is None:
        return audio_data
    
    try:
        # Ensure we don't exceed the profile length
        min_length = min(len(audio_data), len(AMBIENT_NOISE_PROFILE))
        if min_length == 0:
            return audio_data
            
        audio_reduced = nr.reduce_noise(
            y=audio_data[:min_length],
            y_noise=AMBIENT_NOISE_PROFILE[:min_length],
            sr=SAMPLE_RATE,
            stationary=True,
            prop_decrease=0.95,
            n_fft=2048,
            win_length=512
        )
        
        # Pad or trim to match original length
        if len(audio_reduced) < len(audio_data):
            audio_reduced = np.pad(audio_reduced, (0, len(audio_data) - len(audio_reduced)))
        elif len(audio_reduced) > len(audio_data):
            audio_reduced = audio_reduced[:len(audio_data)]
            
        return audio_reduced
    except Exception as e:
        print(f"Active noise reduction failed: {e}")
        return audio_data
    
def detect_drone_from_audio(audio_data):
    try:
        print("\n" + "="*40)
        print("Starting drone detection analysis")
        print("="*40)
        
        # CPU check
        cpu_usage = psutil.cpu_percent()
        if cpu_usage > CPU_THRESHOLD:
            print(f"⚠️ Skipping detection - CPU usage {cpu_usage}% > threshold {CPU_THRESHOLD}%")
            return False, False, 0, None, None
            
        # Preprocessing
        audio_processed = preprocess_audio(audio_data, sr=SAMPLE_RATE)
        if len(audio_processed) < 1024:  # Minimum samples for analysis
            print("Audio too short after preprocessing")
            return False, False, 0, None, None
            
        # Feature extraction
        features = extract_features(audio_processed, sr=SAMPLE_RATE)
        if features is None:
            print("Feature extraction failed")
            return False, False, 0, None, None
            
        # Model predictions
        print("\nRunning model predictions...")
        
        # Drone detection
        drone_features = np.array([[features.get(k, 0) for k in DRONE_DETECTOR.feature_order]])
        is_drone = bool(DRONE_DETECTOR.predict(drone_features)[0])
        drone_confidence = DRONE_DETECTOR.predict_proba(drone_features)[0][1] if hasattr(DRONE_DETECTOR, 'predict_proba') else (0.9 if is_drone else 0.1)
        
        print(f"Drone model prediction: {'✅ DETECTED' if is_drone else '❌ Not detected'} (confidence: {drone_confidence:.2f})")
        
        # Shaheed detection (only if drone detected)
        is_shaheed = False
        shaheed_confidence = 0.0
        main_freq = features.get('f0_mean', None)
        
        if is_drone:
            shaheed_features = np.array([[features.get(k, 0) for k in SHAHEED_DETECTOR.feature_order]])
            is_shaheed = bool(SHAHEED_DETECTOR.predict(shaheed_features)[0])
            shaheed_confidence = SHAHEED_DETECTOR.predict_proba(shaheed_features)[0][1] if hasattr(SHAHEED_DETECTOR, 'predict_proba') else (0.9 if is_shaheed else 0.1)
            
            print(f"Shaheed model prediction: {'✅ POSITIVE' if is_shaheed else '❌ Negative'} (confidence: {shaheed_confidence:.2f})")
            
            # Frequency range validation
            if main_freq:
                if is_shaheed and not (SHAHEED_FREQ_RANGE[0] <= main_freq <= SHAHEED_FREQ_RANGE[1]):
                    print(f"⚠️ Shaheed frequency {main_freq:.1f}Hz outside expected range {SHAHEED_FREQ_RANGE}")
                    is_shaheed = False
                    shaheed_confidence *= 0.7  # Reduce confidence
                
                if not (DRONE_FREQ_RANGE[0] <= main_freq <= DRONE_FREQ_RANGE[1]):
                    print(f"⚠️ Drone frequency {main_freq:.1f}Hz outside expected range {DRONE_FREQ_RANGE}")
                    is_drone = False
                    drone_confidence *= 0.7  # Reduce confidence
        
        # Combined confidence
        combined_confidence = (drone_confidence + shaheed_confidence) / 2 if is_drone else drone_confidence
        print(f"\nFinal detection: {'🚁 DRONE' if is_drone else 'No drone'} | {'💣 SHAHEED' if is_shaheed else ''}")
        print(f"Combined confidence: {combined_confidence:.2f}, Main frequency: {main_freq:.1f}Hz")
        
        return is_drone, is_shaheed, combined_confidence, main_freq, features
        
    except Exception as e:
        print(f"🔥 Error in detect_drone_from_audio: {e}")
        return False, False, 0, None, None
            
def audio_callback(indata, frames, time, status):
    global current_session_id
    
    if not is_listening:
        return
    
    try:
        print("\n🎧 Audio callback triggered")
        if status:
            print(f"Audio stream status: {status}")
        
        # Basic audio stats
        audio_data = indata[:, 0]
        print(f"Audio chunk: {len(audio_data)} samples ({len(audio_data)/SAMPLE_RATE:.2f}s)")
        print(f"Audio levels - Max: {np.max(audio_data):.4f}, Min: {np.min(audio_data):.4f}, RMS: {np.sqrt(np.mean(audio_data**2)):.4f}")
        
        # Detection
        is_drone, is_shaheed, confidence, freq, features = detect_drone_from_audio(audio_data)
        
        # Handle detection
        if is_drone and confidence >= MIN_CONFIDENCE:
            print(f"\n🚨 ALERT: {'Shaheed' if is_shaheed else 'Drone'} detected!")
            print(f"Confidence: {confidence:.2f}, Frequency: {freq:.1f}Hz")
            
            # Save audio sample
            audio_path = save_audio_sample(audio_data)
            print(f"Saved audio sample to: {audio_path}")
            
            # Database logging
            db = SessionLocal()
            try:
                drone_type = "Shaheed" if is_shaheed else "Generic"
                db_event = DetectionEvent(
                    is_drone=True,
                    is_shaheed=is_shaheed,
                    confidence=float(confidence),
                    frequency=float(freq) if freq else None,
                    model_version="enhanced-2.0",
                    audio_sample_path=audio_path,
                    session_id=current_session_id,
                    drone_type=drone_type,
                    spectral_features=json.dumps(features) if features else None
                )
                db.add(db_event)
                db.commit()
                
                # Prepare WebSocket message
                detection_data = {
                    "type": "detection",
                    "data": {
                        "timestamp": db_event.timestamp.isoformat(),
                        "confidence": float(confidence),
                        "is_drone": True,
                        "is_shaheed": is_shaheed,
                        "frequency": float(freq) if freq else None,
                        "session_id": current_session_id,
                        "db_id": db_event.id,
                        "drone_type": drone_type,
                        "spectral_features": features
                    }
                }
                thread_message_queue.put(detection_data)
                print("Detection logged to database and queued for WebSocket")
                
            except Exception as e:
                db.rollback()
                print(f"Database error: {e}")
            finally:
                db.close()
                
    except Exception as e:
        print(f"Audio callback error: {e}")

async def process_message_queue():
    global current_websocket
    while True:
        try:
            while not thread_message_queue.empty():
                message = thread_message_queue.get_nowait()
                if current_websocket:
                    try:
                        await current_websocket.send_json(message)
                    except Exception as e:
                        print(f"WebSocket send error: {e}")
                        current_websocket = None
                thread_message_queue.task_done()
            
            await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Message processing error: {e}")
            await asyncio.sleep(1)

@app.on_event("startup")
async def startup_event():
    # Run calibration in background to not block startup
    def run_calibration():
        perform_initial_calibration()
    
    Thread(target=run_calibration, daemon=True).start()

@app.websocket("/ws/detections")
async def websocket_endpoint(websocket: WebSocket):
    global current_websocket
    await websocket.accept()
    current_websocket = websocket
    
    try:
        await websocket.send_json({
            "type": "connection",
            "status": "established",
            "session_id": current_session_id
        })

        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "keepalive"})
    except WebSocketDisconnect:
        print("Client disconnected normally")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if current_websocket == websocket:
            current_websocket = None

@app.post("/start_listening")
async def start_listening():
    global is_listening, audio_stream, current_session_id
    
    if not is_listening:
        current_session_id = str(uuid.uuid4())
        is_listening = True
        audio_stream = sd.InputStream(
            callback=audio_callback,
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            blocksize=int(SAMPLE_RATE * DURATION),
            dtype='float32'
        )
        Thread(target=audio_stream.start, daemon=True).start()
    
    return {"status": "listening", "session_id": current_session_id}

@app.post("/stop_listening")
async def stop_listening():
    global is_listening, audio_stream
    
    if is_listening and audio_stream:
        is_listening = False
        audio_stream.stop()
        audio_stream.close()
        audio_stream = None
    
    return {"status": "stopped", "session_id": current_session_id}

@app.post("/calibrate_noise")
async def calibrate_noise(duration: float = CALIBRATION_DURATION):
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            profile = create_fan_noise_profile(duration)
            return {
                "status": "success",
                "message": f"Fan noise profile created with {len(profile)} samples",
                "saved_path": "profiles/fan_noise_profile_*.npy"
            }
        except Exception as e:
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail=str(e))
            print(f"Attempt {attempt + 1} failed, retrying... Error: {e}")
            time.sleep(retry_delay)
            # On macOS, sometimes restarting the audio helps
            if sys.platform == "darwin":
                os.system("sudo killall coreaudiod 2>/dev/null")

@app.post("/calibrate_ambient_noise")
async def api_calibrate_ambient_noise(duration: float = CALIBRATION_DURATION):
    try:
        profile = calibrate_ambient_noise(duration)
        return {
            "status": "success",
            "message": f"Ambient noise profile created with {len(profile)} samples",
            "saved_path": "profiles/ambient_noise_profile_*.npy"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/calibrate_shaheed")
async def calibrate_shaheed(duration: float = CALIBRATION_DURATION):
    try:
        print(f"Recording Shaheed reference for {duration} seconds...")
        audio_data = sd.rec(int(duration * SAMPLE_RATE),
                          samplerate=SAMPLE_RATE,
                          channels=1)
        sd.wait()
        audio_data = audio_data[:, 0]
        
        spectrum = create_shaheed_reference(audio_data)
        return {
            "status": "success",
            "message": f"Shaheed reference spectrum created",
            "saved_path": "profiles/shaheed_reference_*.npy"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/toggle_active_noise_reduction")
async def toggle_active_noise_reduction(enabled: bool = None):
    global ACTIVE_NOISE_REDUCTION_ENABLED
    if enabled is not None:
        ACTIVE_NOISE_REDUCTION_ENABLED = enabled
    else:
        ACTIVE_NOISE_REDUCTION_ENABLED = not ACTIVE_NOISE_REDUCTION_ENABLED
    
    return {
        "status": "success",
        "active_noise_reduction": ACTIVE_NOISE_REDUCTION_ENABLED
    }

@app.get("/detections", response_model=list[DetectionResult])
async def get_detections(db: Session = Depends(get_db)):
    detections = db.query(DetectionEvent).order_by(DetectionEvent.timestamp.desc()).limit(100).all()
    return [{
        "timestamp": detection.timestamp,
        "confidence": detection.confidence,
        "is_drone": detection.is_drone,
        "is_shaheed": detection.is_shaheed,
        "frequency": detection.frequency,
        "session_id": detection.session_id,
        "drone_type": detection.drone_type,
        "spectral_features": json.loads(detection.spectral_features) if detection.spectral_features else None
    } for detection in detections]

@app.get("/detections/{session_id}", response_model=list[DetectionResult])
async def get_session_detections(session_id: str, db: Session = Depends(get_db)):
    detections = db.query(DetectionEvent)\
        .filter(DetectionEvent.session_id == session_id)\
        .order_by(DetectionEvent.timestamp.desc()).all()
    return [{
        "timestamp": detection.timestamp,
        "confidence": detection.confidence,
        "is_drone": detection.is_drone,
        "is_shaheed": detection.is_shaheed,
        "frequency": detection.frequency,
        "session_id": detection.session_id,
        "drone_type": detection.drone_type,
        "spectral_features": json.loads(detection.spectral_features) if detection.spectral_features else None
    } for detection in detections]

@app.delete("/detections/")
async def delete_all_detections(db: Session = Depends(get_db)):
    try:
        detections = db.query(DetectionEvent).all()
        
        for detection in detections:
            if detection.audio_sample_path and os.path.exists(detection.audio_sample_path):
                try:
                    os.remove(detection.audio_sample_path)
                except OSError as e:
                    print(f"Error deleting audio file: {e}")
        
        db.query(DetectionEvent).delete()
        db.commit()
        
        return {"status": "success", "message": "All detections deleted"}
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    asyncio.run(server.serve())