import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ALGO_DIR = BASE_DIR / "algo"
DATA_DIR = BASE_DIR / "TITANIUM_V1_FIXED"

os.makedirs(DATA_DIR / "logs", exist_ok=True)
os.makedirs(DATA_DIR / "state", exist_ok=True)

class Settings:
    PROJECT_NAME: str = "Titanium Hedge Dashboard"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    ALPACA_KEY: str = os.getenv("ALPACA_KEY", "")
    ALPACA_SECRET: str = os.getenv("ALPACA_SECRET", "")
    TWELVE_DATA_KEY: str = os.getenv("TWELVE_DATA_KEY", "")

settings = Settings()
