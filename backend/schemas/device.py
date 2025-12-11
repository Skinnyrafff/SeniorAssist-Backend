
from typing import Any, Optional, List
from pydantic import BaseModel, Field

class DeviceProfileUpdate(BaseModel):
    medical_notes: Optional[str] = None
    conditions: Optional[Any] = None
    medications: Optional[Any] = None


class RegisterDeviceRequest(BaseModel):
    device_id: str = Field(..., min_length=1, description="UUID único del dispositivo")
    user_id: str = Field(..., min_length=1, description="ID del usuario vinculado")
    emergency_contact: str = Field(..., min_length=1, description="Nombre del contacto de emergencia")
    emergency_phone: str = Field(..., min_length=1, description="Teléfono del contacto de emergencia")
    medical_notes: Optional[str] = Field(None, description="Notas médicas (alergias, enfermedades crónicas, etc.)")
    conditions: Optional[List[str]] = Field(None, description="Lista de condiciones médicas (ej: ['diabetes', 'hipertensión'])")


class RegisterDeviceResponse(BaseModel):
    """Response después de registrar dispositivo y usuario."""
    success: bool
    device_id: str
    user_id: str
    message: str


class DeviceProfile(BaseModel):
    device_id: str = Field(..., min_length=1)
    user_id: Optional[str] = None
    contact_name: Optional[str] = None
    contact_phone: Optional[str] = None
    medical_notes: Optional[str] = None
    conditions: Optional[Any] = None
    medications: Optional[Any] = None
