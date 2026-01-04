# backend/core_api/app/config.py

class Settings:
    """
    Simple settings container.
    If you later want to load from environment variables, we can extend this.
    For now, it's enough for app metadata and stays compatible with Pydantic v2.
    """
    def __init__(self) -> None:
        self.app_name: str = "Epimind Core API"


settings = Settings()
