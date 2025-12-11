import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ..repositories import repository
from ..schemas import ChatRequest, ChatResponse
from ..services.orchestrator import ConversationOrchestrator
from .deps import get_orchestrator
import unicodedata

router = APIRouter()

log = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, orch: ConversationOrchestrator = Depends(get_orchestrator)) -> Any:
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text is not allowed")
    if not req.device_id.strip():
        raise HTTPException(status_code=400, detail="device_id is required and cannot be empty")
    if not req.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id is required and cannot be empty")
    
    device_id = req.device_id.strip()
    session_id = req.session_id.strip()
    
    log.info("Chat request from device=%s session=%s", device_id, session_id)
    # Normalizar texto de entrada
    raw_text = req.text.strip()
    normalized = unicodedata.normalize("NFD", raw_text)
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    normalized = " ".join(normalized.split())
    norm_text = normalized
    try:
        repository.ensure_device(device_id)
        user_msg_id = repository.save_message(
            device_id=device_id, role="user", text=norm_text, session_id=session_id
        )
    except Exception as exc:  # pragma: no cover
        log.warning("No se pudo guardar mensaje de usuario: %s", exc)
        user_msg_id = None

    try:
        resp = orch.process(norm_text, device_id=device_id, session_id=session_id, source_message_id=user_msg_id)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(exc))

    try:
        repository.save_message(
            device_id=device_id,
            role="assistant",
            text=resp.get("reply", ""),
            payload=resp,
            session_id=session_id,
        )
    except Exception as exc:  # pragma: no cover
        log.warning("No se pudo guardar mensaje del asistente: %s", exc)

    return resp
