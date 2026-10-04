import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
STORAGE_DIR = BASE_DIR / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
CLEANED_DIR = STORAGE_DIR / "cleaned"
REPORTS_DIR = STORAGE_DIR / "reports"

for dir_path in [STORAGE_DIR, UPLOAD_DIR, CLEANED_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

class Settings:
    PROJECT_NAME: str = "Enterprise AI Data Intelligence Platform"
    API_V1_STR: str = "/api"
    
    # Database: Supports PostgreSQL via environment or falls back to SQLite seamlessly
    # Example PostgreSQL URL: postgresql://postgres:postgres@localhost:5432/dataintel
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        f"sqlite:///{BASE_DIR / 'dataintel.db'}"
    )
    
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"
    ]

settings = Settings()
