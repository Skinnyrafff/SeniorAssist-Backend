from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ReminderStatus(str, Enum):
    draft = "draft"
    confirmed = "confirmed"
    cancelled = "cancelled"
    done = "done"


class ReminderCreate(BaseModel):
    device_id: str
    title: str
    due_at: Optional[datetime] = None
    timezone: Optional[str] = None
    status: ReminderStatus = Field(default=ReminderStatus.draft)


class ReminderUpdateStatus(BaseModel):
    status: ReminderStatus


class ReminderUpdate(BaseModel):
    title: Optional[str] = None
    due_at: Optional[datetime] = None
    timezone: Optional[str] = None
    status: Optional[ReminderStatus] = None


class ReminderResponse(BaseModel):
    """
    Respuesta de GET /reminders (optimizada para frontend Android).
    Solo incluye los 4 campos que la app Android necesita:
    - id: Identificador único del recordatorio
    - title: Título/descripción del recordatorio
    - due_at: Fecha/hora en ISO 8601 cuando se debe activar
    - status: Estado (draft, confirmed, done, cancelled)
    """
    id: str = Field(..., description="Identificador único")
    title: str = Field(..., description="Título del recordatorio")
    due_at: Optional[str] = Field(None, description="ISO 8601 datetime string")
    status: str = Field(..., description="Estado: draft, confirmed, done, cancelled")
