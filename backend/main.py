import logging
import sys
from typing import Optional

from fastapi import FastAPI

from .api import chat, emergencies, history, reminders, devices, users
from .core.config import settings
from .adapters import predictors
from .repositories import repository
from .services.orchestrator import ConversationOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

app = FastAPI(title=settings.app_name, version=settings.app_version)
ORCHESTRATOR: Optional[ConversationOrchestrator] = None


@app.on_event("startup")
def _startup() -> None:
    global ORCHESTRATOR
    try:
        repository.init_db()
        predictors.load_models()
        ORCHESTRATOR = ConversationOrchestrator()
        app.state.orchestrator = ORCHESTRATOR
    except Exception as exc:  # pragma: no cover
        log.exception("Startup error")
        raise


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


app.include_router(chat.router)
app.include_router(history.router)
app.include_router(reminders.router)
app.include_router(emergencies.router)
app.include_router(devices.router)
app.include_router(users.router)
