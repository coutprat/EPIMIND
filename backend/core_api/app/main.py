"""
Epimind Core API
Manages patient records, EEG events, and risk monitoring data.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .db import init_db
from .config import settings
from .routers import patients, events, analysis


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup, cleanup on shutdown."""
    init_db()
    print("[STARTUP] Database initialized")
    yield
    print("[SHUTDOWN] Application shutting down")


# Initialize FastAPI application
app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Core API for Epimind (patients, events, history)",
    lifespan=lifespan,
)

# CORS middleware to allow requests from frontend and inference service
# Allows requests from Vite dev server (port 5173) and localhost
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers for patients, events, and analysis
app.include_router(patients.router)
app.include_router(events.router)
app.include_router(analysis.router)


@app.get("/health")
def health_check():
    """
    Health check endpoint.
    Returns status and model availability information.
    """
    # Check if a real model is available (currently using fallback/mock)
    # In production, this would check if ONNX/TorchScript model is loaded
    model_available = False  # Mock/fallback is currently active
    model_type = "Dummy" if not model_available else "ONNX/TorchScript"
    
    return {
        "status": "ok",
        "model_available": model_available,
        "model_type": model_type
    }

