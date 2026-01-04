# backend/core_api/app/db.py

from pathlib import Path
from sqlmodel import SQLModel, create_engine, Session

# Use absolute path for SQLite database file in the backend directory
DB_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = DB_DIR / 'epimind.db'
# SQLite URL format: sqlite:///path (with forward slashes even on Windows)
DATABASE_URL = f"sqlite:///{str(DB_PATH).replace(chr(92), '/')}"

engine = create_engine(DATABASE_URL, echo=False)


def init_db() -> None:
    """Initialize database tables on startup. Drop and recreate all tables."""
    from . import models  # ensure models are imported
    # Drop all existing tables and recreate them (for schema updates during development)
    SQLModel.metadata.drop_all(bind=engine)
    SQLModel.metadata.create_all(bind=engine)


def get_session():
    """Get a database session for dependency injection."""
    with Session(engine) as session:
        yield session
