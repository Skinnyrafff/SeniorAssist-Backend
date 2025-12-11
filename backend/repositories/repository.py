"""
Repository helpers to persist messages and basic entities.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlmodel import Session, select

from . import db, db_models

logger = logging.getLogger(__name__)


REMINDER_STATUSES = {"draft", "confirmed", "cancelled", "done"}
EMERGENCY_STATUSES = {"detected", "confirming", "escalated", "cancelled", "resolved"}


def init_db() -> None:
    db.init_db()


def _now() -> datetime:
    return datetime.utcnow()


def _session() -> Session:
    return Session(db.get_engine(), expire_on_commit=False)


# Users --------------------------------------------------------------------
def create_user(
    *,
    full_name: str,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    language: str = "es",
    timezone: str = "Europe/Madrid",
    consent_accepted: bool = False,
) -> str:
    user = db_models.User(
        id=str(uuid4()),
        full_name=full_name,
        email=email,
        phone=phone,
        language=language,
        timezone=timezone,
        consent_accepted=consent_accepted,
        created_at=_now(),
        updated_at=_now(),
    )
    with _session() as session:
        session.add(user)
        session.commit()
    return user.id


def get_user(user_id: str) -> Optional[db_models.User]:
    with _session() as session:
        return session.get(db_models.User, user_id)


def update_user(
    user_id: str,
    full_name: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    language: Optional[str] = None,
    timezone: Optional[str] = None,
    consent_accepted: Optional[bool] = None,
) -> bool:
    with _session() as session:
        user = session.get(db_models.User, user_id)
        if not user:
            return False
        if full_name is not None:
            user.full_name = full_name
        if email is not None:
            user.email = email
        if phone is not None:
            user.phone = phone
        if language is not None:
            user.language = language
        if timezone is not None:
            user.timezone = timezone
        if consent_accepted is not None:
            user.consent_accepted = consent_accepted
        user.updated_at = _now()
        session.add(user)
        session.commit()
        return True


# Devices ------------------------------------------------------------------
def create_device(
    device_id: str,
    owner_name: Optional[str] = None,
    emergency_contact: Optional[str] = None,
    emergency_phone: Optional[str] = None,
    user_id: Optional[str] = None,
    medical_notes: Optional[str] = None,
    conditions: Optional[list] = None,
) -> str:
    """Registra un nuevo dispositivo en el sistema."""
    with _session() as session:
        existing = session.get(db_models.Device, device_id)
        if existing:
            logger.info("Dispositivo %s ya existe, actualizando datos", device_id)
            if owner_name:
                existing.contact_name = owner_name
            if emergency_phone:
                existing.contact_phone = emergency_phone
            if medical_notes is not None:
                existing.medical_notes = medical_notes
            if conditions is not None:
                existing.conditions = conditions
            existing.updated_at = _now()
            session.add(existing)
            session.commit()
            return device_id
        
        device = db_models.Device(
            device_id=device_id,
            user_id=user_id,
            contact_name=owner_name,
            contact_phone=emergency_phone,
            medical_notes=medical_notes,
            conditions=conditions,
            created_at=_now(),
            updated_at=_now(),
        )
        session.add(device)
        session.commit()
        logger.info("Dispositivo registrado: %s (propietario: %s)", device_id, owner_name or "sin especificar")
    return device_id


def ensure_device(device_id: str, user_id: Optional[str] = None) -> None:
    with _session() as session:
        existing = session.get(db_models.Device, device_id)
        if existing:
            return
        session.add(
            db_models.Device(
                device_id=device_id,
                user_id=user_id,
                created_at=_now(),
                updated_at=_now(),
            )
        )
        session.commit()


def get_device(device_id: str) -> Optional[db_models.Device]:
    with _session() as session:
        return session.get(db_models.Device, device_id)


def update_device(
    device_id: str,
    user_id: Optional[str] = None,
    contact_name: Optional[str] = None,
    contact_phone: Optional[str] = None,
    medical_notes: Optional[str] = None,
    conditions: Optional[Any] = None,
    medications: Optional[Any] = None,
) -> bool:
    with _session() as session:
        dev = session.get(db_models.Device, device_id)
        if not dev:
            return False
        if user_id is not None:
            dev.user_id = user_id
        if contact_name is not None:
            dev.contact_name = contact_name
        if contact_phone is not None:
            dev.contact_phone = contact_phone
        if medical_notes is not None:
            dev.medical_notes = medical_notes
        if conditions is not None:
            dev.conditions = conditions
        if medications is not None:
            dev.medications = medications
        dev.updated_at = _now()
        session.add(dev)
        session.commit()
        return True


def save_message(
    *,
    device_id: str,
    role: str,
    text: str,
    payload: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
) -> str:
    message = db_models.Message(
        id=str(uuid4()),
        device_id=device_id,
        session_id=session_id,
        role=role,
        text=text,
        created_at=_now(),
    )
    if payload:
        intent = payload.get("intent") or {}
        sentiment = payload.get("sentiment") or {}
        emotion = payload.get("emotion") or {}
        
        # Unified ML analysis metadata
        message.ml_analysis = {
            "intent": {
                "label": intent.get("label"),
                "score": intent.get("score"),
                "top_k": intent.get("top_k")
            },
            "sentiment": {
                "label": sentiment.get("label"),
                "score": sentiment.get("score"),
                "top_k": sentiment.get("top_k")
            },
            "emotion": {
                "label": emotion.get("label"),
                "score": emotion.get("score"),
                "top_k": emotion.get("top_k")
            }
        }
        
        # Unified flow decision metadata
        message.flow_metadata = {
            "assigned": payload.get("flow"),
            "reason": payload.get("flow_reason"),
            "source": payload.get("flow_source"),
            "gate": payload.get("gate"),
            "decoder": payload.get("decoder")
        }
        
        message.entities = payload.get("entities")
        message.emergency = bool(payload.get("emergency"))
        
    with _session() as session:
        session.add(message)
        session.commit()
    return message.id


def list_messages(device_id: str, limit: int = 20, session_id: Optional[str] = None) -> List[db_models.Message]:
    with _session() as session:
        stmt = select(db_models.Message).where(db_models.Message.device_id == device_id)
        if session_id:
            stmt = stmt.where(db_models.Message.session_id == session_id)
        stmt = stmt.order_by(db_models.Message.created_at.desc()).limit(limit)
        return list(session.exec(stmt))


def create_reminder(
    *,
    device_id: str,
    title: Optional[str],
    due_at: Optional[datetime],
    timezone: Optional[str],
    status: str = "draft",
) -> str:
    if status not in REMINDER_STATUSES:
        raise ValueError(f"Invalid reminder status: {status}")
    reminder = db_models.Reminder(
        id=str(uuid4()),
        device_id=device_id,
        title=title,
        due_at=due_at,
        timezone=timezone,
        status=status,
        created_at=_now(),
        updated_at=_now(),
    )
    with _session() as session:
        session.add(reminder)
        session.commit()
    return reminder.id


def update_reminder_status(reminder_id: str, status: str) -> bool:
    if status not in REMINDER_STATUSES:
        raise ValueError(f"Invalid reminder status: {status}")
    with _session() as session:
        reminder = session.get(db_models.Reminder, reminder_id)
        if not reminder:
            return False
        reminder.status = status
        reminder.updated_at = _now()
        session.add(reminder)
        session.commit()
        return True


def update_reminder_fields(
    reminder_id: str,
    title: Optional[str],
    due_at: Optional[datetime],
    timezone: Optional[str],
    status: Optional[str],
) -> bool:
    with _session() as session:
        reminder = session.get(db_models.Reminder, reminder_id)
        if not reminder:
            return False
        if title is not None:
            reminder.title = title
        if due_at is not None:
            reminder.due_at = due_at
        if timezone is not None:
            reminder.timezone = timezone
        if status is not None:
            if status not in REMINDER_STATUSES:
                raise ValueError(f"Invalid reminder status: {status}")
            reminder.status = status
        reminder.updated_at = _now()
        session.add(reminder)
        session.commit()
        return True


def list_reminders(
    device_id: str, status: Optional[str] = None, limit: int = 50, session_id: Optional[str] = None
) -> List[db_models.Reminder]:
    with _session() as session:
        stmt = select(db_models.Reminder).where(db_models.Reminder.device_id == device_id)
        if status:
            stmt = stmt.where(db_models.Reminder.status == status)
        stmt = stmt.order_by(db_models.Reminder.created_at.desc()).limit(limit)
        return list(session.exec(stmt))


def get_latest_reminder(device_id: str) -> Optional[db_models.Reminder]:
    with _session() as session:
        stmt = select(db_models.Reminder).where(db_models.Reminder.device_id == device_id)
        stmt = stmt.order_by(db_models.Reminder.created_at.desc()).limit(1)
        res = session.exec(stmt).first()
        return res


def get_reminder_by_id(reminder_id: str) -> Optional[db_models.Reminder]:
    with _session() as session:
        return session.get(db_models.Reminder, reminder_id)


def find_similar_reminder(
    *,
    device_id: str,
    title: Optional[str],
    due_at: Optional[datetime],
    session_id: Optional[str] = None,
    window_hours: int = 24,
) -> Optional[db_models.Reminder]:
    with _session() as session:
        stmt = select(db_models.Reminder).where(db_models.Reminder.device_id == device_id)
        if session_id:
            stmt = stmt.where(db_models.Reminder.session_id == session_id)
        if title:
            stmt = stmt.where(db_models.Reminder.title == title)
        stmt = stmt.order_by(db_models.Reminder.created_at.desc())
        candidates = list(session.exec(stmt))
        if due_at:
            for rem in candidates:
                if rem.due_at:
                    delta = abs((rem.due_at - due_at).total_seconds())
                    if delta <= window_hours * 3600:
                        return rem
        return candidates[0] if candidates else None


def cleanup_reminder_duplicates(
    *, device_id: str, title: Optional[str], due_at: Optional[datetime], keep_id: str
) -> None:
    if not title or not due_at:
        return
    with _session() as session:
        stmt = select(db_models.Reminder).where(db_models.Reminder.device_id == device_id)
        stmt = stmt.where(db_models.Reminder.title == title)
        stmt = stmt.where(db_models.Reminder.due_at == due_at)
        duplicates = list(session.exec(stmt))
        for rem in duplicates:
            if rem.id == keep_id:
                continue
            rem.status = "cancelled"
            rem.updated_at = _now()
            session.add(rem)
        session.commit()


def create_emergency_event(
    *,
    device_id: str,
    status: str = "detected",
    session_id: Optional[str] = None,
    reason: Optional[str] = None,
    action: Optional[str] = None,
    contact_name: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
    source_message_id: Optional[str] = None,
) -> str:
    if status not in EMERGENCY_STATUSES:
        raise ValueError(f"Invalid emergency status: {status}")
    event = db_models.EmergencyEvent(
        id=str(uuid4()),
        device_id=device_id,
        session_id=session_id,
        status=status,
        reason=reason,
        action=action,
        contact_name=contact_name,
        meta=meta,
        source_message_id=source_message_id,
        created_at=_now(),
    )
    with _session() as session:
        session.add(event)
        session.commit()
    return event.id


def update_emergency_status(event_id: str, status: str, action: Optional[str] = None) -> bool:
    if status not in EMERGENCY_STATUSES:
        raise ValueError(f"Invalid emergency status: {status}")
    with _session() as session:
        event = session.get(db_models.EmergencyEvent, event_id)
        if not event:
            return False
        event.status = status
        if action:
            event.action = action
        if status in {"resolved", "cancelled"}:
            event.resolved_at = _now()
        session.add(event)
        session.commit()
        return True


def list_emergencies(
    device_id: str, status: Optional[str] = None, limit: int = 50, session_id: Optional[str] = None
) -> List[db_models.EmergencyEvent]:
    with _session() as session:
        stmt = select(db_models.EmergencyEvent).where(db_models.EmergencyEvent.device_id == device_id)
        if status:
            stmt = stmt.where(db_models.EmergencyEvent.status == status)
        if session_id:
            stmt = stmt.where(db_models.EmergencyEvent.session_id == session_id)
        stmt = stmt.order_by(db_models.EmergencyEvent.created_at.desc()).limit(limit)
        return list(session.exec(stmt))


def get_latest_open_emergency(device_id: str, session_id: Optional[str] = None) -> Optional[db_models.EmergencyEvent]:
    with _session() as session:
        stmt = select(db_models.EmergencyEvent).where(db_models.EmergencyEvent.device_id == device_id)
        if session_id:
            stmt = stmt.where(db_models.EmergencyEvent.session_id == session_id)
        stmt = stmt.where(db_models.EmergencyEvent.status.notin_(["resolved", "cancelled"]))
        stmt = stmt.order_by(db_models.EmergencyEvent.created_at.desc()).limit(1)
        return session.exec(stmt).first()


def get_emergency_by_id(event_id: str) -> Optional[db_models.EmergencyEvent]:
    with _session() as session:
        return session.get(db_models.EmergencyEvent, event_id)


def get_active_emergency_for_device(device_id: str) -> Optional[Dict[str, Any]]:
    """
    Obtiene la emergencia activa más reciente para un dispositivo.
    Devuelve un dict con info de emergencia + contacto de emergencia del dispositivo.
    Retorna None si no hay emergencia activa.
    """
    emergency = get_latest_open_emergency(device_id)
    if not emergency:
        return None
    
    device = get_device(device_id)
    protocol = emergency.action or "escalate"
    
    return {
        "emergency_id": emergency.id,
        "status": "emergency",
        "protocol": protocol,
        "reason": emergency.reason,
        "contact_name": emergency.contact_name or (device.contact_name if device else None),
        "contact_phone": device.contact_phone if device else None,
    }
