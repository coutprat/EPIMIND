# backend/core_api/app/routers/patients.py

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from ..db import get_session
from ..models import Patient, Event
from ..schemas import PatientCreate, PatientRead, EventList

router = APIRouter(prefix="/patients", tags=["patients"])


@router.post("/", response_model=PatientRead, status_code=status.HTTP_201_CREATED)
def create_patient(payload: PatientCreate, session: Session = Depends(get_session)):
    """Create a new patient."""
    try:
        # Only include fields that are in the Patient model
        data = payload.model_dump(exclude_unset=False)
        patient = Patient(**data)
        session.add(patient)
        session.commit()
        session.refresh(patient)
        return patient
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Error creating patient: {str(e)}")


@router.get("/", response_model=List[PatientRead])
def list_patients(session: Session = Depends(get_session)):
    """List all patients."""
    patients = session.exec(select(Patient)).all()
    return patients


@router.get("/{patient_id}", response_model=PatientRead)
def get_patient(patient_id: int, session: Session = Depends(get_session)):
    """Get a specific patient by ID."""
    patient = session.get(Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@router.get("/{patient_id}/events", response_model=EventList)
def get_patient_events(patient_id: int, session: Session = Depends(get_session)):
    """Retrieve all events for a specific patient."""
    # Verify patient exists
    patient = session.get(Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Get all events for this patient, ordered by timestamp (most recent first)
    events = session.exec(
        select(Event)
        .where(Event.patient_id == patient_id)
        .order_by(Event.timestamp.desc())
    ).all()
    
    return EventList(events=events)

