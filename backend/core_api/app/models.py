# backend/core_api/app/models.py

from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field


class Patient(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    identifier: Optional[str] = Field(default=None, description="Hospital ID / MRN / custom ID")
    patient_id: Optional[str] = Field(default=None, description="Alternative patient identifier")
    age: Optional[int] = None
    gender: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Event(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    patient_id: int = Field(foreign_key="patient.id")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_type: Optional[str] = Field(default=None, description="Type of event (e.g., EEG_RECORDING)")
    duration_seconds: Optional[float] = Field(default=None, description="Duration in seconds")
    channel_count: Optional[int] = Field(default=None, description="Number of EEG channels")
    sampling_rate: Optional[float] = Field(default=None, description="Sampling rate in Hz")
    risk_score: float = Field(default=0.0)
    label: str = Field(default="normal")
    model_version: Optional[str] = None
    meta: Optional[str] = Field(default=None, description="JSON string with extra context if needed")
