from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class EmergencyStatus(str, Enum):
    detected = "detected"
    confirming = "confirming"
    escalated = "escalated"
    cancelled = "cancelled"
    resolved = "resolved"


class EmergencyStatusResponse(BaseModel):
    """Response para GET /emergency-status/{device_id}"""
    status: str = Field(..., description="'ok' o 'emergency'")
    protocol: Optional[str] = Field(None, description="Protocolo a seguir (call_family, call_ambulance, etc)")
    contact_name: Optional[str] = Field(None, description="Nombre del contacto de emergencia")
    contact_phone: Optional[str] = Field(None, description="Teléfono del contacto")
    emergency_id: Optional[str] = Field(None, description="ID de la emergencia activa")
    reason: Optional[str] = Field(None, description="Razón de la emergencia")


class TriggerEmergencyRequest(BaseModel):
    """Request para POST /trigger-emergency"""
    device_id: str
    protocol: str = Field(..., description="call_family o call_ambulance")
    reason: Optional[str] = Field(None, description="Razón de la emergencia")


class EmergencyCreate(BaseModel):
    device_id: str
    status: EmergencyStatus = Field(default=EmergencyStatus.detected)
    session_id: Optional[str] = None
    reason: Optional[str] = None
    action: Optional[str] = None
    contact_name: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    source_message_id: Optional[str] = None


class EmergencyUpdateStatus(BaseModel):
    status: EmergencyStatus
    action: Optional[str] = None
