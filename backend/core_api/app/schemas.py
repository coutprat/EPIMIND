# backend/core_api/app/schemas.py

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel


class PatientCreate(BaseModel):
    name: str
    identifier: Optional[str] = None
    patient_id: Optional[str] = None  # Alternative ID field
    age: Optional[int] = None
    gender: Optional[str] = None  # Allow gender field
    notes: Optional[str] = None


class PatientRead(BaseModel):
    id: int
    name: str
    identifier: Optional[str]
    age: Optional[int]
    notes: Optional[str]
    created_at: datetime
    patient_id: Optional[str] = None  # Return patient_id if set

    class Config:
        from_attributes = True


class EventCreate(BaseModel):
    patient_id: int
    event_type: Optional[str] = None  # Allow event_type field
    duration_seconds: Optional[float] = None  # Allow duration field
    channel_count: Optional[int] = None  # Allow channel_count field
    sampling_rate: Optional[float] = None  # Allow sampling_rate field
    risk_score: float = 0.0  # Default value
    label: str = "normal"  # Default value
    model_version: Optional[str] = None
    meta: Optional[str] = None


class EventRead(BaseModel):
    id: int
    patient_id: int
    timestamp: datetime
    risk_score: float
    label: str
    model_version: Optional[str]
    meta: Optional[str]
    event_type: Optional[str] = None
    duration_seconds: Optional[float] = None
    channel_count: Optional[int] = None
    sampling_rate: Optional[float] = None

    class Config:
        from_attributes = True


class EventList(BaseModel):
    events: List[EventRead]
