"""Environment and CLI configuration."""

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")


@dataclass
class Settings:
    api_key: str = os.getenv("SARVAM_API_KEY", "")
    model: str = os.getenv("SARVAM_MODEL", "sarvam-105b")
    dataset_path: Path = Path(
        os.getenv("DATASET_PATH", "datasets/eval_dataset.jsonl")
    )
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", "results"))
    max_cases: int = int(os.getenv("MAX_CASES", "0"))
    request_delay_seconds: float = float(
        os.getenv("REQUEST_DELAY_SECONDS", "0.5")
    )
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    temperature: float = float(os.getenv("TEMPERATURE", "0"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "1200"))

    judge_enabled: bool = (
        os.getenv("JUDGE_ENABLED", "false").strip().lower()
        in {"1", "true", "yes"}
    )
    judge_api_key: str = os.getenv("JUDGE_API_KEY", "").strip()
    judge_base_url: str = os.getenv("JUDGE_BASE_URL", "").strip()
    judge_model: str = os.getenv("JUDGE_MODEL", "").strip()
    judge_temperature: float = float(os.getenv("JUDGE_TEMPERATURE", "0"))


settings = Settings()
