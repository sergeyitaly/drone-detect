import os
import numpy as np
import librosa
import sounddevice as sd
from fastapi import FastAPI, HTTPException, WebSocket, Depends, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import soundfile as sf
import noisereduce as nr
from scipy import signal
import joblib
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Float, text, inspect
#from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import json
import uuid
import asyncio
from threading import Thread
import psutil
from concurrent.futures import ThreadPoolExecutor
import queue
from typing import Optional, Dict
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from detector_models import load_models, preprocess_audio, extract_features, load_models
import time
import sys
from dotenv import load_dotenv
import webrtcvad  # For voice activity n
from fastapi import Body 
import threading 
from uuid import uuid4
import webrtcvad
import collections
import threading
from typing import Optional, List, Any
from datetime import datetime, timedelta
from sqlalchemy.sql import func
from datetime import datetime
from models import *
import re


VAD_AGGRESSIVENESS = 3  # Most aggressive setting
VOICE_ACTIVITY_FRAME_MS = 30  # milliseconds
VOICE_ACTIVITY_PADDING_MS = 300  # milliseconds
SAMPLE_RATE = 16000  
VAD_FRAME_SIZE = int(SAMPLE_RATE * VOICE_ACTIVITY_FRAME_MS / 1000)


load_dotenv()

SAMPLE_RATE = 44100
DURATION = 2  # seconds
CHANNELS = 1
DRONE_FREQ_RANGE = (80, 250)  # Hz
SHAHEED_FREQ_RANGE = (120, 180)  # Hz
MIN_CONFIDENCE = 0.8
CPU_THRESHOLD = 95
CALIBRATION_DURATION = 5  # Longer calibration for better fan profile
FAN_NOISE_CUTOFF = 5000  # Hz - fan noise is typically above this
NOTCH_FREQS = [4000, 6000]  # Common MacBook fan frequencies to notch out
NOISE_REDUCTION_AGGRESSION = 0.99  # More aggressive reduction for fans
# At the top of your file, declare the global variables
DRONE_DETECTOR = None
SHAHEED_DETECTOR = None
DRONE_DETECTOR, SHAHEED_DETECTOR = load_models()

app = FastAPI()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\""))
    conn.commit()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Global state
is_listening = False
current_websocket = None
audio_stream = None
#current_session_id = str(uuid.uuid4())
current_session_id = None
session_lock = threading.Lock() 

FAN_NOISE_PROFILE = None
AMBIENT_NOISE_PROFILE = None
thread_message_queue = queue.Queue()
executor = ThreadPoolExecutor(max_workers=1)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    #allow_websockets=["*"],
)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# Audio Processing Functions
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def apply_notch_filters(audio_data, sr, notch_freqs):
    for freq in notch_freqs:
        q = 30  # Quality factor
        b, a = signal.iirnotch(freq, q, sr)
        audio_data = signal.lfilter(b, a, audio_data)
    return audio_data

def enhanced_noise_reduction(audio_data):
    """Specialized noise reduction pipeline for MacBook fan noise"""
    try:
        # 1. Bandpass filter to focus on drone frequencies
        filtered = butter_bandpass_filter(
            audio_data, 
            DRONE_FREQ_RANGE[0], 
            FAN_NOISE_CUTOFF, 
            SAMPLE_RATE
        )
        
        # 2. Notch filters for specific fan frequencies
        filtered = apply_notch_filters(filtered, SAMPLE_RATE, NOTCH_FREQS)
        
        # 3. Spectral noise reduction if profile exists
        if FAN_NOISE_PROFILE is not None:
            min_length = min(len(filtered), len(FAN_NOISE_PROFILE))
            if min_length > 0:
                filtered = nr.reduce_noise(
                    y=filtered[:min_length],
                    y_noise=FAN_NOISE_PROFILE[:min_length],
                    sr=SAMPLE_RATE,
                    stationary=True,
                    prop_decrease=NOISE_REDUCTION_AGGRESSION,
                    n_fft=2048,
                    win_length=512
                )
        
        return filtered
    except Exception as e:
        print(f"Enhanced noise reduction failed: {e}")
        return audio_data

def save_audio_sample(audio_data, sample_rate=SAMPLE_RATE):
    os.makedirs("detected_drones", exist_ok=True)
    filename = f"detected_drones/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{current_session_id}.wav"
    sf.write(filename, audio_data, sample_rate)
    return filename

# Noise Profile Management
def load_fan_noise_profile():
    global FAN_NOISE_PROFILE
    try:
        os.makedirs("profiles", exist_ok=True)
        if os.path.exists("profiles/fan_noise_profile_latest.npy"):
            FAN_NOISE_PROFILE = np.load("profiles/fan_noise_profile_latest.npy")
            print("Loaded latest fan noise profile")
    except Exception as e:
        print(f"Error loading fan profile: {e}")

def create_fan_noise_profile(duration=CALIBRATION_DURATION, max_retries=3):
    global FAN_NOISE_PROFILE
    print("Recording fan noise profile...")
    print("Please ensure only fan noise is present during recording...")
    
    for attempt in range(max_retries):
        try:
            fan_noise = sd.rec(int(duration * SAMPLE_RATE), 
                              samplerate=SAMPLE_RATE, 
                              channels=1)
            sd.wait()
            FAN_NOISE_PROFILE = fan_noise[:, 0]
            
            # Save profile
            os.makedirs("profiles", exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            profile_path = f"profiles/fan_noise_profile_{timestamp}.npy"
            np.save(profile_path, FAN_NOISE_PROFILE)
            np.save("profiles/fan_noise_profile_latest.npy", FAN_NOISE_PROFILE)
            
            print(f"Saved fan profile to {profile_path}")
            return FAN_NOISE_PROFILE
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed after {max_retries} attempts: {e}")
            print(f"Attempt {attempt + 1} failed, retrying... Error: {e}")
            time.sleep(1)
            if sys.platform == "darwin":
                os.system("sudo killall coreaudiod 2>/dev/null")

# Detection Functions
def detect_drone_from_audio(audio_data):
    try:
        # CPU check
        if psutil.cpu_percent() > CPU_THRESHOLD:
            return False, False, 0, None, None
            
        # Preprocessing with enhanced noise reduction
        clean_audio = enhanced_noise_reduction(audio_data)
        audio_processed = preprocess_audio(clean_audio, sr=SAMPLE_RATE)
        
        if len(audio_processed) < 1024:
            return False, False, 0, None, None
            
        # Feature extraction
        features = extract_features(audio_processed, sr=SAMPLE_RATE)
        if features is None:
            return False, False, 0, None, None
            
        # Model predictions
        drone_features = np.array([[features.get(k, 0) for k in DRONE_DETECTOR.feature_order]])
        is_drone = bool(DRONE_DETECTOR.predict(drone_features)[0])
        drone_confidence = DRONE_DETECTOR.predict_proba(drone_features)[0][1] if hasattr(DRONE_DETECTOR, 'predict_proba') else (0.9 if is_drone else 0.1)
        
        # Shaheed detection
        is_shaheed = False
        shaheed_confidence = 0.0
        main_freq = features.get('f0_mean', None)
        
        if is_drone and main_freq:
            shaheed_features = np.array([[features.get(k, 0) for k in SHAHEED_DETECTOR.feature_order]])
            is_shaheed = bool(SHAHEED_DETECTOR.predict(shaheed_features)[0])
            shaheed_confidence = SHAHEED_DETECTOR.predict_proba(shaheed_features)[0][1] if hasattr(SHAHEED_DETECTOR, 'predict_proba') else (0.9 if is_shaheed else 0.1)
            
            # Frequency validation
            if is_shaheed and not (SHAHEED_FREQ_RANGE[0] <= main_freq <= SHAHEED_FREQ_RANGE[1]):
                is_shaheed = False
                shaheed_confidence *= 0.7
                
            if not (DRONE_FREQ_RANGE[0] <= main_freq <= DRONE_FREQ_RANGE[1]):
                is_drone = False
                drone_confidence *= 0.7
        
        combined_confidence = (drone_confidence + shaheed_confidence) / 2 if is_drone else drone_confidence
        return is_drone, is_shaheed, combined_confidence, main_freq, features
        
    except Exception as e:
        print(f"Detection error: {e}")
        return False, False, 0, None, None

class VoiceActivityDetector:
    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.audio_buffer = collections.deque(maxlen=100)
        self.lock = threading.Lock()
        self.active = False
        self.sample_rate = SAMPLE_RATE
        
    def process_frame(self, frame):
        """Process a single audio frame for voice activity"""
        with self.lock:
            if len(frame) != VAD_FRAME_SIZE * 2:  # 16-bit samples
                frame = frame[:VAD_FRAME_SIZE * 2]
            try:
                return self.vad.is_speech(frame, self.sample_rate)
            except:
                return False
    
    def update_buffer(self, audio_data):
        """Update the audio buffer with new data"""
        with self.lock:
            self.audio_buffer.extend(audio_data)
    
    def get_audio_context(self, padding_frames):
        """Get audio context around current frame"""
        with self.lock:
            return list(self.audio_buffer)[-padding_frames:]

class ActiveNoiseSuppressor:
    def __init__(self):
        self.vad = VoiceActivityDetector()
        self.noise_profile = None
        self.sample_rate = SAMPLE_RATE
        self.is_speech = False
        self.speech_history = collections.deque(maxlen=10)
        
    def update_noise_profile(self, audio_data):
        """Update noise profile during silent periods"""
        if not self.is_speech:
            if self.noise_profile is None:
                self.noise_profile = audio_data.copy()
            else:
                # Exponential moving average for noise profile
                self.noise_profile = 0.9 * self.noise_profile + 0.1 * audio_data
    
    def suppress_noise(self, audio_data):
        """
        Apply noise suppression based on voice activity detection
        Returns suppressed audio and voice activity flag
        """
        # Resample if needed (assuming original is 44100)
        if len(audio_data) != VAD_FRAME_SIZE:
            audio_data_16k = librosa.resample(audio_data, orig_sr=44100, target_sr=self.sample_rate)
        else:
            audio_data_16k = audio_data
            
        # Convert to 16-bit PCM for VAD
        audio_int16 = (audio_data_16k * 32767).astype('int16').tobytes()
        
        # Detect voice activity
        self.vad.update_buffer(audio_int16)
        is_speech = self.vad.process_frame(audio_int16)
        self.speech_history.append(is_speech)
        
        # Use majority voting over recent frames to reduce flickering
        self.is_speech = sum(self.speech_history) > len(self.speech_history) / 2
        
        # Update noise profile during non-speech periods
        self.update_noise_profile(audio_data)
        
        # Apply noise reduction if we have a profile
        if self.noise_profile is not None and len(self.noise_profile) == len(audio_data):
            if self.is_speech:
                # Less aggressive reduction during speech
                reduced_noise = nr.reduce_noise(
                    y=audio_data,
                    y_noise=self.noise_profile,
                    sr=44100,
                    stationary=True,
                    prop_decrease=0.7
                )
            else:
                # More aggressive reduction during silence
                reduced_noise = nr.reduce_noise(
                    y=audio_data,
                    y_noise=self.noise_profile,
                    sr=44100,
                    stationary=True,
                    prop_decrease=0.95
                )
            return reduced_noise, self.is_speech
        
        return audio_data, self.is_speech

# Initialize the suppressor at startup
NOISE_SUPPRESSOR = ActiveNoiseSuppressor()

def audio_callback(indata, frames, time, status):
    global current_session_id
    
    if not is_listening or current_session_id is None:
        return
    
    try:
        audio_data = indata[:, 0]
        
        # Apply noise suppression
        suppressed_audio, is_voice = NOISE_SUPPRESSOR.suppress_noise(audio_data)
        if is_voice:
            return
            
        clean_audio = enhanced_noise_reduction(suppressed_audio)
        is_drone, is_shaheed, confidence, freq, features = detect_drone_from_audio(clean_audio)
        
        if is_drone and confidence >= MIN_CONFIDENCE:
            # Create database session once per callback
            db = SessionLocal()
            try:
                # Save audio sample
                audio_path = save_audio_sample(clean_audio)
                
                # Create detection
                detection = DetectionEvent(
                    timestamp=datetime.utcnow(),
                    is_drone=True,
                    is_shaheed=is_shaheed,
                    confidence=float(confidence),
                    frequency=float(freq) if freq else None,
                    model_version="enhanced-2.1",
                    audio_sample_path=audio_path,
                    session_id=current_session_id,  # Use current session
                    drone_type="Shaheed" if is_shaheed else "Generic",
                    spectral_features=json.dumps(features) if features else None
                )
                
                db.add(detection)
                db.commit()
                db.refresh(detection)
                
                # Prepare WebSocket message
                ws_msg = {
                    "type": "detection",
                    "data": {
                        "timestamp": detection.timestamp.isoformat(),
                        "confidence": detection.confidence,
                        "is_drone": detection.is_drone,
                        "is_shaheed": detection.is_shaheed,
                        "frequency": detection.frequency,
                        "session_id": detection.session_id,
                        "drone_type": detection.drone_type,
                        "spectral_features": json.loads(detection.spectral_features) if detection.spectral_features else None
                    }
                }
                
                # Send via WebSocket if connected
                if current_websocket:
                    asyncio.run_coroutine_threadsafe(
                        current_websocket.send_json(ws_msg),
                        asyncio.get_event_loop()
                    )
                
            except Exception as db_error:
                db.rollback()
                print(f"Database error: {db_error}")
            finally:
                db.close()
                
    except Exception as e:
        print(f"Audio processing error: {e}")

# Add this endpoint to configure noise suppression
@app.post("/configure_noise_suppression")
async def configure_noise_suppression(
    aggressiveness: int = Body(3, embed=True),
    voice_threshold: float = Body(0.5, embed=True)
):
    """
    Configure noise suppression parameters
    - aggressiveness: 0-3 (3 is most aggressive)
    - voice_threshold: 0-1 (confidence threshold for voice detection)
    """
    global NOISE_SUPPRESSOR
    NOISE_SUPPRESSOR.vad.vad.set_mode(min(3, max(0, aggressiveness)))
    return {"status": "updated", "aggressiveness": aggressiveness, "voice_threshold": voice_threshold}


def initialize_database():
    # First ensure UUID extension exists
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\""))
        conn.commit()
    
    # Then create all tables
    Base.metadata.create_all(bind=engine)
    
    # Verify creation
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    print("Existing tables:", existing_tables)
    
    if 'sessions' not in existing_tables:
        raise RuntimeError("Failed to create sessions table!")
    
# Modified startup_event with proper initialization sequence
@app.on_event("startup")
async def startup_event():
    # 1. First initialize database
    initialize_database()
    
    # 2. Run other initializations in background thread
    def run_initializations():
        try:
            # Initialize noise suppressor with default settings
            global NOISE_SUPPRESSOR
            NOISE_SUPPRESSOR = ActiveNoiseSuppressor()
            print("Noise suppressor initialized with default settings")
            
            # Configure noise suppression
            NOISE_SUPPRESSOR.vad.vad.set_mode(3)  # Most aggressive mode
            print(f"Noise suppression configured with aggressiveness: 3")
            
            # Load or create fan noise profile
            load_fan_noise_profile()
            
            if FAN_NOISE_PROFILE is None:
                print("No fan noise profile found, creating one automatically...")
                try:
                    create_fan_noise_profile()
                    print("Automatic fan noise profile creation successful")
                except Exception as e:
                    print(f"Failed to automatically create fan profile: {e}")
                    print("Please create a fan noise profile manually using /calibrate_fan_noise endpoint")
            else:
                print("System ready with existing fan noise profile")
                                
        except Exception as e:
            print(f"Startup initialization error: {e}")
    
    # Run all initializations in background thread
    Thread(target=run_initializations, daemon=True).start()

@app.get("/sessions/current", response_model=Optional[SessionResponse])
async def get_current_session(db: Session = Depends(get_db)):
    db_session = db.query(Session).filter(
        Session.is_active == True
    ).order_by(Session.created_at.desc()).first()
    
    if not db_session:
        return None  # Instead of raising 404
    
    return {
        "session_id": db_session.id,
        "status": "active",
        "created_at": db_session.created_at,
        "sensitivity": db_session.sensitivity,
        "frequency_range": db_session.frequency_range
    }


# Replace the websocket endpoint with this fixed version
@app.websocket("/ws/detections")
async def websocket_endpoint(websocket: WebSocket):
    allowed_origins = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost", 
        "http://127.0.0.1"
    ]
    origin = websocket.headers.get("origin")

    if origin and origin not in allowed_origins:
        await websocket.close(code=1008)  # Policy Violation
        return
    
    await websocket.accept()
    global current_websocket, current_session_id
    
    try:
        # Get or create session ID in a thread-safe way
        with session_lock:
            if current_session_id is None:
                current_session_id = str(uuid4())
            session_id = current_session_id
            
        current_websocket = websocket
        
        # Initial connection message
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "session_id": session_id
        })
        
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                    
                elif data.get("type") == "start_listening":
                    # Handle session start with lock
                    with session_lock:
                        current_session_id = str(uuid4())
                        is_listening = True
                        session_id = current_session_id
                    
                    await websocket.send_json({
                        "type": "session",
                        "status": "started",
                        "session_id": session_id
                    })
                    
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "keepalive"})
                
    except WebSocketDisconnect:
        print(f"Client disconnected: {websocket.client}")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if current_websocket == websocket:
            current_websocket = None
            
@app.post("/start_listening")
async def start_listening():
    global is_listening, current_session_id, audio_stream
    
    with session_lock:
        if not is_listening:
            current_session_id = str(uuid.uuid4())  # Generate new session ID
            is_listening = True
            
            if audio_stream is None:
                audio_stream = sd.InputStream(
                    callback=audio_callback,
                    channels=CHANNELS,
                    samplerate=SAMPLE_RATE,
                    blocksize=int(SAMPLE_RATE * DURATION),
                    dtype='float32'
                )
                audio_stream.start()
            
            print(f"Started new session: {current_session_id}")
            
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

@app.post("/calibrate_fan_noise")
async def calibrate_fan_noise(duration: float = CALIBRATION_DURATION):
    try:
        profile = create_fan_noise_profile(duration)
        return {
            "status": "success",
            "message": f"Fan noise profile created with {len(profile)} samples",
            "saved_path": "profiles/fan_noise_profile_*.npy"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


@app.post("/sessions/", response_model=SessionResponse)
async def create_session(
    session_data: dict = Body(...),
    db: Session = Depends(get_db)
):
    global current_session_id, is_listening, audio_stream
    
    try:
        new_session = Session(
            id=str(uuid.uuid4()),
            is_active=True,
            sensitivity=session_data.get('sensitivity'),
            frequency_range=session_data.get('frequency_range'),
            created_at=func.now()
        )
        
        db.add(new_session)
        db.commit()
        db.refresh(new_session)

        current_session_id = new_session.id
        is_listening = True
        
        if audio_stream is None:
            audio_stream = sd.InputStream(
                callback=audio_callback,
                channels=CHANNELS,
                samplerate=SAMPLE_RATE,
                blocksize=int(SAMPLE_RATE * DURATION),
                dtype='float32'
            )
            audio_stream.start()
        
        return {
            "session_id": new_session.id,
            "status": "active",
            "created_at": new_session.created_at,
            "sensitivity": new_session.sensitivity,
            "frequency_range": new_session.frequency_range
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
            
@app.post("/sessions/{session_id}/stop", response_model=SessionStopResponse)
async def stop_session(
    session_id: str, 
    db: Session = Depends(get_db)
):
    """Stop an active session with cleanup"""
    db_session = db.query(Session).filter(
        Session.id == session_id,
        Session.is_active == True
    ).first()
    
    if not db_session:
        raise HTTPException(status_code=404, detail="Active session not found")
    
    try:
        db_session.is_active = False
        db_session.stopped_at = func.now()
        db.commit()
        
        # Update global state
        global current_session_id
        if current_session_id == session_id:
            current_session_id = None
        
        return {
            "session_id": session_id,
            "status": "stopped",
            "stopped_at": db_session.stopped_at,
            "duration_seconds": db_session.duration
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to stop session: {str(e)}")

@app.websocket("/ws/session/{session_id}")
async def websocket_session(
    websocket: WebSocket,
    session_id: str,
    db: Session = Depends(get_db)
):
    """WebSocket connection for real-time session updates"""
    await websocket.accept()
    
    # Verify session exists
    db_session = db.query(Session).filter(
        Session.id == session_id
    ).first()
    
    if not db_session:
        await websocket.close(code=1008, reason="Session not found")
        return
    
    try:
        # Send initial session state
        await websocket.send_json({
            "type": "session_state",
            "data": SessionResponse.model_validate(db_session).model_dump()
        })
        
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                
            elif data.get("type") == "update_settings":
                # Handle settings updates
                db_session.sensitivity = data.get("sensitivity", db_session.sensitivity)
                db.commit()
                await websocket.send_json({
                    "type": "settings_updated",
                    "sensitivity": db_session.sensitivity
                })
                
    except WebSocketDisconnect:
        print(f"Client disconnected from session {session_id}")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()

@app.get("/detections", response_model=List[DetectionResult])
async def get_detections(
    session_id: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get detections with optional session filtering"""
    query = db.query(DetectionEvent)
    
    if session_id:
        query = query.filter(DetectionEvent.session_id == session_id)
    
    detections = query.order_by(
        DetectionEvent.timestamp.desc()
    ).limit(limit).all()
    
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

# Database operations
def store_gnss_data(db: Session, data: GNSSDataSchema) -> GNSSDataSchema:
    # Convert Pydantic schema to SQLAlchemy model
    db_data = GNSSData(**data.model_dump())  # Changed from dict() to model_dump()
    db.add(db_data)
    db.commit()
    db.refresh(db_data)
    # Convert back to Pydantic schema before returning
    return GNSSDataSchema.model_validate(db_data)  # Changed from from_orm


def get_previous_gnss_data(db: Session, device_id: str = None) -> GNSSDataSchema:
    # Query using SQLAlchemy model
    query = db.query(GNSSData)
    if device_id:
        query = query.filter(GNSSData.device_id == device_id)
    result = query.order_by(GNSSData.timestamp.desc()).first()
    
    if not result:
        return GNSSDataSchema(
            timestamp=datetime.now(),
            gps_sats=0,
            glonass_sats=0,
            galileo_sats=0,
            beidou_sats=0,
            snr_values={},
            device_id=device_id
        )
    # Convert SQLAlchemy model to Pydantic schema
    return GNSSDataSchema.model_validate(result)

def query_jamming_events(db: Session, hours: int = 24):
    return db.query(JammingEvent).filter(
        JammingEvent.start_time >= datetime.now() - timedelta(hours=hours)
    ).all()

def create_security_incident(
    db: Session,
    incident_type: str,
    severity: float,
    description: str
):
    db_event = JammingEvent(
        start_time=datetime.now(),
        severity=severity,
        affected_systems=["all"],
        confidence=severity,
        description=description
    )
    db.add(db_event)
    db.commit()
    db.refresh(db_event)
    return db_event

# Jamming analysis
def calculate_confidence(sat_drop: bool, low_snr: bool, inconsistent: bool) -> float:
    """Calculate jamming confidence score (0-1)"""
    confidence = 0.0
    if sat_drop:
        confidence += 0.6
    if low_snr:
        confidence += 0.3
    if inconsistent:
        confidence += 0.3
    return min(1.0, confidence)

def analyze_jamming(db: Session, data: GNSSDataSchema) -> dict:
    """Analyze GNSS data for potential jamming"""
    prev_data = get_previous_gnss_data(db, data.device_id)
    
    # Rule 1: Sudden satellite drop
    sat_drop = any([
        (prev_data.gps_sats - data.gps_sats) > 5,
        (prev_data.glonass_sats - data.glonass_sats) > 5,
    ])
    
    # Rule 2: Low SNR across systems
    low_snr = all(
        np.mean(list(system_snr.values())) < 25 
        for system_snr in data.snr_values.values()
        if system_snr
    )
    
    # Rule 3: Signal inconsistency
    snr_stddev = {
        system: np.std(list(snr.values())) 
        for system, snr in data.snr_values.items()
    }
    inconsistent = any(std > 15 for std in snr_stddev.values())
    
    return {
        "is_jamming": sat_drop or (low_snr and inconsistent),
        "confidence": calculate_confidence(sat_drop, low_snr, inconsistent),
        "metrics": {
            "satellite_drop": sat_drop,
            "low_snr": low_snr,
            "inconsistent_signals": inconsistent
        }
    }


# Alert system (would integrate with your existing WebSocket system)
async def update_detection_parameters(params: dict):
    """Update drone detection parameters when jamming is detected"""
    # This would interface with your existing detection system
    print(f"Updating detection parameters: {params}")

async def broadcast_alert(message: str):
    """Broadcast alert to connected clients"""
    # This would use your existing WebSocket implementation
    print(f"ALERT: {message}")

async def broadcast_jamming_alert():
    """Broadcast jamming alert to connected clients"""
    await broadcast_alert("GNSS jamming detected - increased drone alert level")

# API Endpoints
@app.post("/gnss/register")
async def register_gnss_device(
    device: GNSSDevice,
    db: Session = Depends(get_db)
):
    """Register a new GNSS device"""
    # In a real implementation, you would store device info
    return {"status": "registered", "device_id": device.device_id}

def parseNMEASentence(sentence: str) -> Optional[Dict[str, Any]]:
    """Parse NMEA sentence with strict validation and error handling."""
    if not sentence or not isinstance(sentence, str):
        return None
        
    # More comprehensive NMEA format validation
    if not re.match(r'^\$[A-Z]{2}[A-Z]{3},.*(\*[0-9A-F]{2})?$', sentence):
        return None

    try:
        # Checksum validation
        if '*' in sentence:
            content, checksum_str = sentence[1:].split('*', 1)
            calculated_checksum = 0
            for ch in content:
                calculated_checksum ^= ord(ch)
            try:
                checksum = int(checksum_str.strip(), 16)
                if checksum != calculated_checksum:
                    return None
            except ValueError:
                return None
        else:
            content = sentence[1:]

        parts = content.split(',')
        if len(parts) < 4:  # Minimum fields for GSV
            return None

        talker = parts[0][:2]
        sentence_type = parts[0][2:]

        # Only process GSV messages
        if sentence_type != 'GSV':
            return None

        system_map = {
            'GP': 'gps',
            'GL': 'glonass', 
            'GA': 'galileo',
            'BD': 'beidou',
            'GN': 'gnss'
        }

        system = system_map.get(talker)
        if not system:
            return None

        # Validate and convert numeric fields
        try:
            total_msgs = int(parts[1])
            msg_num = int(parts[2])
            total_sats = int(parts[3])
            
            # Basic validation of GSV message numbers
            if not (1 <= msg_num <= total_msgs):
                return None
        except (ValueError, IndexError):
            return None

        # Extract SNR values more safely
        snr = []
        for i in range(7, min(len(parts), 31), 4):
            try:
                snr_value = int(parts[i])
                if 0 < snr_value <= 99:  # Valid SNR range
                    snr.append(snr_value)
            except (ValueError, IndexError):
                continue

        return {
            'type': 'satellite_data',
            'system': system,
            'total_sats': total_sats,
            'snr': snr,
            'timestamp': datetime.utcnow(),
            'msg_num': msg_num,
            'total_msgs': total_msgs
        }

    except Exception as e:
        # Consider using proper logging here
        print(f"NMEA parsing error: {str(e)}")
        return None
        
@app.post("/gnss/data")
async def receive_gnss_raw_data(
    raw_nmea: str = Body(..., embed=True, max_length=256),  # Add length limit
    db: Session = Depends(get_db)
):
    if not raw_nmea or len(raw_nmea) > 128:  # Typical NMEA max length
        return {"status": "error", "reason": "Invalid data length"}

    parsed = parseNMEASentence(raw_nmea.strip())
    if not parsed or parsed.get('type') != 'satellite_data':
        return {"status": "ignored", "reason": "Not satellite data or invalid NMEA"}

    try:
        # Create data model with all systems
        gnss_data = GNSSDataSchema(
            gps_sats=len(parsed['snr']) if parsed['system'] == 'gps' else 0,
            glonass_sats=len(parsed['snr']) if parsed['system'] == 'glonass' else 0,
            galileo_sats=len(parsed['snr']) if parsed['system'] == 'galileo' else 0,
            beidou_sats=len(parsed['snr']) if parsed['system'] == 'beidou' else 0,
            snr_values={parsed['system']: parsed['snr']},
            raw_nmea=raw_nmea,
            timestamp=datetime.utcnow(),
            msg_sequence=f"{parsed['msg_num']}/{parsed['total_msgs']}"  # Track message sequence
        )
        stored_data = store_gnss_data(db, gnss_data)
        jamming_status = analyze_jamming(db, stored_data)
        if jamming_status.get("is_jamming"):
            await handle_jamming_event(db, jamming_status)
            await broadcast_jamming_alert()

        return {
            "status": "success",
            "system": parsed['system'],
            "satellites": len(parsed['snr']),
            "jamming_status": jamming_status
        }

    except Exception as e:
        print(f"Error processing GNSS data: {str(e)}")
        return {"status": "error", "reason": "Processing failed"}

@app.get("/gnss/jamming-history")
async def get_jamming_history(
    hours: int = 24,
    db: Session = Depends(get_db)
):
    """Get historical jamming events"""
    events = query_jamming_events(db, hours)
    return {"events": events}

# Jamming event handler
async def handle_jamming_event(db: Session, jamming_status: dict):
    """Handle jamming detection event"""
    # 1. Increase drone detection sensitivity
    await update_detection_parameters({
        "audio_sensitivity": 1.5,
        "harmonic_threshold": 0.4
    })
    
    # 2. Notify all connected clients
    await broadcast_alert("GNSS jamming detected - increased drone alert level")
    
    # 3. Log security event
    create_security_incident(
        db=db,
        incident_type="jamming",
        severity=jamming_status["confidence"],
        description="Potential GNSS jamming affecting drone detection"
    )

@app.websocket("/ws/gnss")
async def gnss_websocket(websocket: WebSocket, db: Session = Depends(get_db)):
    await websocket.accept()

    async def send_ping():
        try:
            while True:
                await websocket.send_json({"type": "ping"})
                await asyncio.sleep(10)
        except Exception:
            # If sending ping fails, exit the ping loop (likely disconnect)
            pass

    ping_task = asyncio.create_task(send_ping())

    try:
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=15)
                if msg == 'pong':
                    continue  # just ignore pong
            except asyncio.TimeoutError:
                print("GNSS client timeout, disconnecting")
                await websocket.close()
                break

            latest_db_data = db.query(GNSSData).order_by(GNSSData.timestamp.desc()).first()
            if latest_db_data:
                latest_data = GNSSDataSchema.model_validate(latest_db_data)
                snr_values = latest_data.snr_values or {}

                payload = {
                    "type": "gnss_data",
                    "systems": {
                        "gps": {"in_view": latest_data.gps_sats, "snr": snr_values.get("gps", [])},
                        "glonass": {"in_view": latest_data.glonass_sats, "snr": snr_values.get("glonass", [])},
                        "galileo": {"in_view": latest_data.galileo_sats, "snr": snr_values.get("galileo", [])},
                        "beidou": {"in_view": latest_data.beidou_sats, "snr": snr_values.get("beidou", [])},
                    },
                    "timestamp": latest_data.timestamp.isoformat()
                }
            else:
                payload = {
                    "type": "gnss_data",
                    "systems": {
                        "gps": {"in_view": 0, "snr": []},
                        "glonass": {"in_view": 0, "snr": []},
                        "galileo": {"in_view": 0, "snr": []},
                        "beidou": {"in_view": 0, "snr": []},
                    },
                    "timestamp": datetime.utcnow().isoformat(),
                }

            await websocket.send_json(payload)
            await asyncio.sleep(5)

    except WebSocketDisconnect:
        print("GNSS client disconnected")
    finally:
        ping_task.cancel()

@app.get("/db-check")
async def db_check(db: Session = Depends(get_db)):
    try:
        result = db.execute(text("SELECT current_database(), current_user, version()"))
        row = result.fetchone()
        return {
            "database": row[0],
            "user": row[1],
            "version": row[2],
            "status": "connected"
        }
    except Exception as e:
        return {"error": str(e), "status": "disconnected"}

@app.api_route("/", methods=["GET", "HEAD"])
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

async def process_message_queue():
    global current_websocket
    while True:
        try:
            while not thread_message_queue.empty():
                message = thread_message_queue.get_nowait()
                if current_websocket:
                    await current_websocket.send_json(message)
                thread_message_queue.task_done()
            await asyncio.sleep(0.1)
        except Exception as e:
            await asyncio.sleep(1)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
