import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, status

from ..repositories import repository
from ..schemas import EmergencyCreate, EmergencyUpdateStatus, EmergencyStatusResponse, TriggerEmergencyRequest

router = APIRouter()
log = logging.getLogger(__name__)


@router.get("/emergency-status/{device_id}", response_model=EmergencyStatusResponse)
def get_emergency_status(device_id: str):
    """
    Endpoint que consulta la app Android cada 15 segundos.
    
    Retorna:
    - {"status": "ok"} si no hay emergencia activa
    - {"status": "emergency", "protocol": "...", ...} si hay emergencia activa
    """
    try:
        repository.ensure_device(device_id)
        active_emergency = repository.get_active_emergency_for_device(device_id)
        
        if active_emergency:
            return EmergencyStatusResponse(**active_emergency)
        else:
            return EmergencyStatusResponse(status="ok")
    except Exception as exc:
        log.exception("Error getting emergency status for device %s", device_id)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/trigger-emergency", response_model=dict, status_code=status.HTTP_201_CREATED)
def trigger_emergency(payload: TriggerEmergencyRequest):
    """
    Endpoint para activar una emergencia manualmente.
    Útil para testing y para casos donde el sistema frontend activa la emergencia.
    
    Request:
    {
        "device_id": "device-001",
        "protocol": "call_family",
        "reason": "Usuario presionó botón SOS"
    }
    """
    try:
        repository.ensure_device(payload.device_id)
        event_id = repository.create_emergency_event(
            device_id=payload.device_id,
            status="detected",
            reason=payload.reason or f"Manual trigger: {payload.protocol}",
            action=payload.protocol,
        )
        log.info("Emergency triggered for device %s: %s", payload.device_id, event_id)
        return {"status": "triggered", "emergency_id": event_id}
    except Exception as exc:
        log.exception("Error triggering emergency")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/emergencies")
def create_emergency(event: EmergencyCreate):
    try:
        repository.ensure_device(event.device_id)
        eid = repository.create_emergency_event(
            device_id=event.device_id,
            status=event.status,
            session_id=event.session_id,
            reason=event.reason,
            action=event.action,
            contact_name=event.contact_name,
            meta=event.meta,
            source_message_id=event.source_message_id,
        )
        return {"id": eid}
    except Exception as exc:
        log.exception("Error creating emergency event")
        raise HTTPException(status_code=500, detail=str(exc))


@router.patch("/emergencies/{event_id}")
def update_emergency(event_id: str, payload: EmergencyUpdateStatus):
    ok = repository.update_emergency_status(event_id, payload.status, payload.action)
    if not ok:
        raise HTTPException(status_code=404, detail="Emergency event not found")
    return {"status": "ok"}


@router.get("/emergencies")
def list_emergencies(device_id: str, status: Optional[str] = None, limit: int = 50, session_id: Optional[str] = None):
    try:
        limit = max(1, min(limit, 200))
        items = repository.list_emergencies(device_id=device_id, status=status, limit=limit, session_id=session_id)
        return items
    except Exception as exc:
        log.exception("Error listing emergencies")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/emergencies/{event_id}")
def get_emergency(event_id: str):
    item = repository.get_emergency_by_id(event_id)
    if not item:
        raise HTTPException(status_code=404, detail="Emergency event not found")
    return item


@router.patch("/emergencies/{event_id}")
def update_emergency_status(event_id: str, payload: dict):
    """Actualizar estado de emergencia (para resolver, cancelar, etc.)"""
    status = payload.get("status")
    if not status:
        raise HTTPException(status_code=400, detail="status is required")
    ok = repository.update_emergency_status(event_id, status)
    if not ok:
        raise HTTPException(status_code=404, detail="Emergency event not found")
    return {"status": "ok"}


@router.delete("/emergencies/{event_id}")
def delete_emergency(event_id: str):
    ok = repository.update_emergency_status(event_id, "cancelled")
    if not ok:
        raise HTTPException(status_code=404, detail="Emergency event not found")
    return {"status": "cancelled"}
