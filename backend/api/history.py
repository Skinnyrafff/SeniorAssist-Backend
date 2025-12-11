import logging
from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException

from ..repositories import repository
from ..schemas import HistoryItem

router = APIRouter()
log = logging.getLogger(__name__)


def _serialize_message(msg) -> dict:
    return {
        "id": msg.id,
        "device_id": msg.device_id,
        "session_id": msg.session_id,
        "role": msg.role,
        "text": msg.text,
        "ml_analysis": msg.ml_analysis,
        "flow_metadata": msg.flow_metadata,
        "entities": msg.entities,
        "emergency": msg.emergency,
        "created_at": msg.created_at,
    }


@router.get("/history", response_model=List[HistoryItem])
def history(device_id: str, limit: int = 20, session_id: Optional[str] = None) -> Any:
    try:
        limit = max(1, min(limit, 200))
        msgs = repository.list_messages(device_id=device_id, limit=limit, session_id=session_id)
        return [_serialize_message(m) for m in msgs]
    except Exception as exc:
        log.exception("Error fetching history")
        raise HTTPException(status_code=500, detail=str(exc))
