"""
Database engine helpers.
"""

from pathlib import Path

from sqlmodel import SQLModel, create_engine

from ..core.config import settings

DB_URL = settings.database_url

_ENGINE = None


def get_engine():
    global _ENGINE
    if _ENGINE is None:
        connect_args = {"check_same_thread": False} if DB_URL.startswith("sqlite") else {}
        if DB_URL.startswith("sqlite:///"):
            db_path = DB_URL.replace("sqlite:///", "")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        _ENGINE = create_engine(DB_URL, echo=False, connect_args=connect_args)
    return _ENGINE


def init_db() -> None:
    engine = get_engine()
    from . import db_models  # noqa: F401
    # Reset schema to ensure columns match models (dev/test only)
    SQLModel.metadata.create_all(engine)
