"""Settings come from environment variables only. .env.local / .env are loaded if present (never committed)."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SECRET_ENV_NAMES = ("NVIDIA_API_KEY", "ZETARIS_PASSWORD", "ZETARIS_API_KEY", "MODEL_API_KEY", "ZETARIS_USERNAME")


def load_env_files() -> list[str]:
    """Load .env.local then .env (whybot root, then parent hackathon dir). Existing env wins."""
    loaded = []
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return loaded
    extra = os.environ.get("WHYBOT_ENV_FILE")
    candidates = [Path(extra)] if extra else []
    candidates += [ROOT / ".env.local", ROOT / ".env", ROOT.parent / ".env.local"]
    for p in candidates:
        if p.is_file():
            load_dotenv(p, override=False)
            loaded.append(str(p))
    return loaded


def _bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    sample_mode: bool = False
    data_source: str = "zetaris"           # zetaris | replay | dev
    llm_base_url: str = "https://integrate.api.nvidia.com/v1"
    planner_model: str = "nvidia/nemotron-3-ultra-550b-a55b"
    critic_model: str = "nvidia/nemotron-3-ultra-550b-a55b"
    sql_model: str = "nvidia/nemotron-3.5-lightning-30b-a3b"
    writer_model: str = "nvidia/nemotron-3.5-lightning-30b-a3b"
    llm_timeout_s: float = 90.0
    sql_timeout_s: float = 90.0
    run_timeout_s: float = 900.0
    max_iterations: int = 3
    max_sql_repairs: int = 2
    hmem_threshold: float = 0.15
    logs_dir: Path = field(default_factory=lambda: ROOT / "logs")
    fixtures_dir: Path = field(default_factory=lambda: ROOT / "fixtures" / "recorded")
    record_fixtures: bool = False
    meterless_enabled: bool = True
    meterless_data_dir: Path = field(default_factory=lambda: ROOT / ".meterless")
    zetaris_databases: list[str] = field(default_factory=lambda: ["NY_TAXI", "NOAA_GHCN_S3", "HOLIDAYS_REST", "TPCDS_DB"])
    # Evidence allowlist. Real, curated sources only: the synthetic NY_TAXI tables (trips, weather_data, surge_pricing, ...)
    # are excluded on purpose, and raw NY_TAXI.yellow_taxi (38.7M rows, ~170 s uncached scan) is opt-in via ZETARIS_TABLES.
    zetaris_tables: list[str] = field(default_factory=lambda: [
        "whybot.yellow_daily", "whybot.nyc_weather_daily", "whybot.us_holidays", "TPCDS_DB.date_dim", "NY_TAXI.taxi_zones"])

    @classmethod
    def from_env(cls) -> "Settings":
        load_env_files()
        s = cls()
        s.sample_mode = _bool("SAMPLE_MODE", False)
        s.data_source = "replay" if s.sample_mode else os.environ.get("DATA_SOURCE", "zetaris").strip().lower()
        s.llm_base_url = os.environ.get("LLM_BASE_URL", s.llm_base_url)
        s.planner_model = os.environ.get("PLANNER_MODEL", s.planner_model)
        s.critic_model = os.environ.get("CRITIC_MODEL", s.critic_model)
        s.sql_model = os.environ.get("SQL_MODEL", s.sql_model)
        s.writer_model = os.environ.get("WRITER_MODEL", s.writer_model)
        s.llm_timeout_s = float(os.environ.get("LLM_TIMEOUT_S", s.llm_timeout_s))
        s.sql_timeout_s = float(os.environ.get("SQL_TIMEOUT_S", s.sql_timeout_s))
        s.run_timeout_s = float(os.environ.get("RUN_TIMEOUT_S", s.run_timeout_s))
        s.max_iterations = min(3, int(os.environ.get("MAX_ITERATIONS", s.max_iterations)))
        s.hmem_threshold = float(os.environ.get("HMEM_THRESHOLD", s.hmem_threshold))
        s.record_fixtures = _bool("RECORD_FIXTURES", False)
        s.meterless_enabled = _bool("METERLESS_ENABLED", True)
        if os.environ.get("LOGS_DIR"):
            s.logs_dir = Path(os.environ["LOGS_DIR"])
        if os.environ.get("FIXTURES_DIR"):
            s.fixtures_dir = Path(os.environ["FIXTURES_DIR"])
        if os.environ.get("METERLESS_DATA_DIR"):
            s.meterless_data_dir = Path(os.environ["METERLESS_DATA_DIR"])
        if os.environ.get("ZETARIS_DATABASES"):
            s.zetaris_databases = [x.strip() for x in os.environ["ZETARIS_DATABASES"].split(",") if x.strip()]
        if os.environ.get("ZETARIS_TABLES"):
            s.zetaris_tables = [x.strip() for x in os.environ["ZETARIS_TABLES"].split(",") if x.strip()]
        return s


_SECRET_PATTERNS = [re.compile(r"nvapi-[A-Za-z0-9_\-]{8,}"), re.compile(r"(?i)(password|api[_-]?key|token)\s*[=:]\s*\S+")]


def redact(text: str) -> str:
    """Mask anything that looks like a secret, including the live values of known secret env vars."""
    if not isinstance(text, str) or not text:
        return text
    for name in SECRET_ENV_NAMES:
        val = os.environ.get(name)
        if val and len(val) >= 4 and val in text:
            text = text.replace(val, f"<{name}:redacted>")
    for pat in _SECRET_PATTERNS:
        text = pat.sub(lambda m: m.group(0)[:6] + "...redacted", text)
    return text
