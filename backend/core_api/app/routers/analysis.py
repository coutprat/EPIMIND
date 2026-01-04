"""
Analysis endpoints for EEG processing.
"""

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Query
from sqlmodel import Session
import numpy as np
import json
from pathlib import Path

from ..db import get_session
from ..models import Patient, Event

router = APIRouter(prefix="/analyze", tags=["analysis"])

# Sample data directory
SAMPLE_DATA_DIR = Path("D:/epimind/ml/data")


def create_mock_analysis_result(
    patient_name: str,
    threshold: float = 0.5,
    smooth_window: int = 5,
    consecutive_windows: int = 3,
) -> dict:
    """
    Generate a mock analysis result for demonstration.
    This simulates what a real EEG analysis would produce.
    """
    # Generate synthetic EEG timeline
    num_windows = 200
    timeline = []
    
    # Create synthetic risk scores with some peaks
    for i in range(num_windows):
        # Base risk with some randomness
        base_risk = 0.3 + 0.1 * np.sin(i / 20) + np.random.normal(0, 0.05)
        
        # Add peaks (simulating seizure events)
        if 50 < i < 70 or 130 < i < 150:
            base_risk += 0.5
        
        risk = float(np.clip(base_risk, 0, 1))
        timeline.append({
            "t_sec": i * 2.0,  # 2 seconds per window
            "prob": risk
        })
    
    # Detect alerts based on threshold
    alerts = []
    in_alert = False
    alert_start = 0
    peak_prob = 0
    
    for i, point in enumerate(timeline):
        if point["prob"] > threshold and not in_alert:
            in_alert = True
            alert_start = i
            peak_prob = point["prob"]
        elif point["prob"] <= threshold and in_alert:
            in_alert = False
            alerts.append({
                "start_sec": timeline[alert_start]["t_sec"],
                "end_sec": timeline[i - 1]["t_sec"],
                "peak_prob": float(peak_prob),
                "duration_sec": float(timeline[i - 1]["t_sec"] - timeline[alert_start]["t_sec"])
            })
        
        if in_alert:
            peak_prob = max(peak_prob, point["prob"])
    
    if in_alert:
        alerts.append({
            "start_sec": timeline[alert_start]["t_sec"],
            "end_sec": timeline[-1]["t_sec"],
            "peak_prob": float(peak_prob),
            "duration_sec": float(timeline[-1]["t_sec"] - timeline[alert_start]["t_sec"])
        })
    
    # Calculate summary metrics
    probs = [p["prob"] for p in timeline]
    peak_prob = max(probs) if probs else 0
    mean_prob = float(np.mean(probs)) if probs else 0
    fp_per_hour = len(alerts) * 2  # Rough estimate
    
    return {
        "patient": patient_name,
        "patient_id": patient_name,
        "filename": f"sample_{patient_name}.npz",
        "fs": 500,  # Sampling frequency
        "window_samples": 1000,
        "stride_samples": 500,
        "stride_sec": 2.0,
        "threshold": threshold,
        "num_windows": num_windows,
        "duration_sec": timeline[-1]["t_sec"],
        "timeline": timeline,
        "alerts": alerts,
        "summary": {
            "alerts_count": len(alerts),
            "peak_probability": peak_prob,
            "mean_probability": mean_prob,
            "fp_estimate_per_hour": fp_per_hour
        },
        "analysis_params": {
            "threshold": threshold,
            "smooth_window": smooth_window,
            "consecutive_windows": consecutive_windows
        }
    }


@router.post("/npz")
async def analyze_npz(
    patient: str = Query(..., description="Patient ID or sample name"),
    threshold: float = Query(0.5, ge=0.1, le=0.9),
    smooth_window: int = Query(5, ge=1),
    consecutive_windows: int = Query(3, ge=1),
):
    """
    Analyze a preprocessed NPZ file (sample data).
    
    This endpoint simulates analyzing pre-processed EEG data.
    In production, this would load actual NPZ files and run inference.
    """
    try:
        # Generate mock analysis result
        result = create_mock_analysis_result(
            patient_name=patient,
            threshold=threshold,
            smooth_window=smooth_window,
            consecutive_windows=consecutive_windows
        )
        
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing sample: {str(e)}"
        )


@router.post("/edf")
async def analyze_edf(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.1, le=0.9),
    smooth_window: int = Query(5, ge=1),
    consecutive_windows: int = Query(3, ge=1),
):
    """
    Analyze an EDF file.
    
    This endpoint accepts EDF files, processes them, and returns risk analysis.
    In production, this would interface with the inference service.
    """
    try:
        # For now, generate mock result with filename from upload
        patient_name = file.filename.replace('.edf', '').replace('.EDF', '')
        
        result = create_mock_analysis_result(
            patient_name=patient_name,
            threshold=threshold,
            smooth_window=smooth_window,
            consecutive_windows=consecutive_windows
        )
        
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing file: {str(e)}"
        )
