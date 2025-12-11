import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from typing import List

from ..repositories import repository
from ..schemas import ReminderCreate, ReminderUpdateStatus, ReminderUpdate, ReminderResponse

router = APIRouter()
log = logging.getLogger(__name__)


@router.post("/reminders")
def create_reminder(rem: ReminderCreate):
    try:
        repository.ensure_device(rem.device_id)
        rid = repository.create_reminder(
            device_id=rem.device_id,
            title=rem.title,
            due_at=rem.due_at,
            timezone=rem.timezone,
            status=rem.status,
        )
        return {"id": rid}
    except Exception as exc:
        log.exception("Error creating reminder")
        raise HTTPException(status_code=500, detail=str(exc))


@router.patch("/reminders/{reminder_id}")
def update_reminder(reminder_id: str, payload: ReminderUpdateStatus):
    ok = repository.update_reminder_status(reminder_id, payload.status)
    if not ok:
        raise HTTPException(status_code=404, detail="Reminder not found")
    return {"status": "ok"}


@router.put("/reminders/{reminder_id}")
def put_reminder(reminder_id: str, payload: ReminderUpdate):
    ok = repository.update_reminder_fields(
        reminder_id=reminder_id,
        title=payload.title,
        due_at=payload.due_at,
        timezone=payload.timezone,
        status=payload.status,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Reminder not found")
    return {"status": "ok"}


@router.get("/reminders", response_model=List[ReminderResponse])
def list_reminders(device_id: str, status: Optional[str] = None, limit: int = 50, session_id: Optional[str] = None):
    """
    Obtiene lista de recordatorios para un dispositivo.
    
    Frontend espera:
    - id, title, due_at, status, created_at
    
    Parámetros:
    - device_id: ID del dispositivo (requerido)
    - status: Filtrar por estado (draft, confirmed, done, cancelled)
    - limit: Máximo de recordatorios (default 50, max 200)
    - session_id: Filtrar por sesión (opcional)
    
    Response: Lista de ReminderResponse (frontend-friendly)
    """
    try:
        limit = max(1, min(limit, 200))
        items = repository.list_reminders(device_id=device_id, status=status, limit=limit, session_id=session_id)
        deduped: List[dict] = []
        seen = set()
        for rem in items:
            key = (rem.title, rem.due_at)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(rem)
        # Convertir a ReminderResponse (frontend-friendly)
        responses = []
        for rem in deduped:
            data = rem.dict()
            if data.get("due_at") is not None:
                # Serializar en ISO 8601 UTC con 'Z' al final
                dt = data["due_at"]
                if dt.tzinfo is None:
                    iso_str = dt.isoformat() + "Z"
                else:
                    iso_str = dt.astimezone(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
                data["due_at"] = iso_str
            responses.append(ReminderResponse(**data))
        return responses
    except Exception as exc:
        log.exception("Error listing reminders")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/reminders/{reminder_id}")
def get_reminder(reminder_id: str):
    item = repository.get_reminder_by_id(reminder_id)
    if not item:
        raise HTTPException(status_code=404, detail="Reminder not found")
    return item


@router.delete("/reminders/{reminder_id}")
def delete_reminder(reminder_id: str):
    ok = repository.update_reminder_status(reminder_id, "cancelled")
    if not ok:
        raise HTTPException(status_code=404, detail="Reminder not found")
    return {"status": "cancelled"}
