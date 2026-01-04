# backend/core_api/app/routers/events.py

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select

from ..db import get_session
from ..models import Event, Patient
from ..schemas import EventCreate, EventRead, EventList

router = APIRouter(
    prefix="/events",
    tags=["events"],
)


@router.post("/", response_model=EventRead)
def create_event(payload: EventCreate, session: Session = Depends(get_session)):
    """
    Create a new Event entry for a given patient.
    """
    # Ensure patient exists
    patient = session.get(Patient, payload.patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient does not exist")

    event = Event(**payload.model_dump())
    session.add(event)
    session.commit()
    session.refresh(event)
    return event


@router.get("/", response_model=EventList)
def list_events(
    patient_id: Optional[int] = Query(default=None),
    session: Session = Depends(get_session),
):
    """
    List events, optionally filtered by patient_id.
    """
    query = select(Event)
    if patient_id is not None:
        query = query.where(Event.patient_id == patient_id)

    events = session.exec(query.order_by(Event.timestamp)).all()
    return EventList(events=events)


@router.get("/patient/{patient_id}", response_model=EventList)
def get_patient_events(patient_id: int, session: Session = Depends(get_session)):
    """
    Retrieve all events for a specific patient by patient ID.
    """
    # Verify patient exists
    patient = session.get(Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Get all events for this patient
    events = session.exec(select(Event).where(Event.patient_id == patient_id).order_by(Event.timestamp.desc())).all()
    return EventList(events=events)
