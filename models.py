from sqlalchemy import Column, Boolean, Float, String, DateTime, Integer, JSON, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional, Dict, List
from pydantic import UUID4

Base = declarative_base()

# SQLAlchemy Models
class DetectionEvent(Base):
    __tablename__ = "detection_events"
    extend_existing=True

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), default=func.now())
    confidence = Column(Float)  
    is_drone = Column(Boolean)
    is_shaheed = Column(Boolean, nullable=True)
    frequency = Column(Float)
    model_version = Column(String)
    audio_sample_path = Column(String)
    session_id = Column(String)
    drone_type = Column(String)
    spectral_features = Column(JSON)

class Session(Base):
    __tablename__ = "sessions"
    extend_existing=True
    id = Column(UUID(as_uuid=True), primary_key=True, index=True, 
               server_default=text("gen_random_uuid()"))  # Changed to gen_random_uuid()
    is_active = Column(Boolean, default=True)
    sensitivity = Column(Float)
    frequency_range = Column(JSON)  # Changed from ARRAY(Integer) to JSON
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    ended_at = Column(DateTime(timezone=True), nullable=True)

class GNSSData(Base):
    __tablename__ = "gnss_data"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime)
    gps_sats = Column(Integer)
    glonass_sats = Column(Integer)
    galileo_sats = Column(Integer)
    beidou_sats = Column(Integer)
    snr_values = Column(JSON)
    device_id = Column(String, nullable=True)

class GNSSDataSchema(BaseModel):
    timestamp: datetime
    gps_sats: int
    glonass_sats: int
    galileo_sats: int
    beidou_sats: int
    snr_values: Dict[str, List[float]]
    device_id: Optional[str] = None
    model_config = ConfigDict(from_attributes=True)  # Enables ORM mode

class JammingEvent(Base):
    __tablename__ = "jamming_events"
    extend_existing=True
    id = Column(Integer, primary_key=True, index=True)
    start_time = Column(DateTime(timezone=True))
    end_time = Column(DateTime(timezone=True), nullable=True)
    severity = Column(Float)
    affected_systems = Column(JSON)  # Consistently using JSON type
    confidence = Column(Float)
    description = Column(String, nullable=True)

# Pydantic Models
class DetectionResult(BaseModel):
    timestamp: datetime
    confidence: float
    is_drone: bool
    is_shaheed: Optional[bool] = None
    frequency: Optional[float] = None
    session_id: Optional[str] = None
    drone_type: Optional[str] = None
    spectral_features: Optional[Dict] = None  # <-- here only type annotation, no Column()
    model_config = ConfigDict(from_attributes=True)  # Pydantic v2 style

class SessionResponse(BaseModel):
    session_id: UUID4
    status: str
    created_at: datetime
    sensitivity: Optional[float] = None
    frequency_range: Optional[List[float]] = None

    model_config = ConfigDict(from_attributes=True)

class SessionStopResponse(BaseModel):
    session_id: str
    status: str
    stopped_at: datetime

class GNSSDevice(BaseModel):
    device_id: str
    model: str = "QU33N"
    firmware: str
    capabilities: List[str]

    model_config = ConfigDict(from_attributes=True)