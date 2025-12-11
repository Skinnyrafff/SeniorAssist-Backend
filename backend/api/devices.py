import logging

from fastapi import APIRouter, HTTPException, status

from ..repositories import repository
from ..schemas import DeviceProfile, RegisterDeviceRequest, RegisterDeviceResponse, DeviceProfileUpdate

router = APIRouter()
log = logging.getLogger(__name__)


@router.post("/devices/register", response_model=RegisterDeviceResponse, status_code=status.HTTP_201_CREATED)
def register_device(payload: RegisterDeviceRequest):
    """
    Registra un nuevo usuario Y dispositivo en el primer inicio de la app.
    
    Si el device_id ya existe:
    - Actualiza los datos del Usuario existente
    - Actualiza los datos del Dispositivo existente
    - Retorna el mismo user_id (no crea uno nuevo)
    
    Crea automáticamente:
    1. Un usuario (User) con datos de la persona mayor
    2. Un dispositivo (Device) vinculado al usuario
    3. Vinculación mediante user_id
    
    Campos OBLIGATORIOS:
    - device_id, owner_name, emergency_contact, emergency_phone
    
    Campos OPCIONALES:
    - email, phone, date_of_birth, language, timezone, consent_accepted
    
    Request:
    {
        "device_id": "uuid-123",
        "owner_name": "Juan García López",
        "emergency_contact": "Hija María",
        "emergency_phone": "+34-655-123-456",
        "email": "juan@example.com",
        "phone": "+34-987-654-321",
        "language": "es",
        "timezone": "Europe/Madrid",
        "consent_accepted": true
    }
    
    Response (201):
    {
        "success": true,
        "device_id": "uuid-123",
        "user_id": "user-uuid",
        "message": "Usuario y dispositivo registrados"
    }
    """
    try:
        # 1. Verificar si el dispositivo ya existe
        log.info(f"Payload recibido en /devices/register: {payload}")
        existing_device = repository.get_device(payload.device_id)
        if existing_device and existing_device.user_id:
            user_id = existing_device.user_id
            repository.update_device(
                device_id=payload.device_id,
                user_id=payload.user_id,
                contact_name=payload.emergency_contact,
                contact_phone=payload.emergency_phone
            )
            log.info("Dispositivo re-registrado: user_id=%s, device_id=%s", user_id, payload.device_id)
        else:
            user_id = payload.user_id
            device_id = repository.create_device(
                device_id=payload.device_id,
                user_id=user_id,
                owner_name=payload.emergency_contact,
                emergency_phone=payload.emergency_phone,
                emergency_contact=payload.emergency_contact,
                medical_notes=payload.medical_notes,
                conditions=payload.conditions
            )
            log.info("Usuario y dispositivo registrados: user_id=%s, device_id=%s", user_id, device_id)
        return RegisterDeviceResponse(
            success=True,
            device_id=payload.device_id,
            user_id=user_id,
            message=f"Usuario y dispositivo registrados exitosamente",
        )
    except Exception as exc:
        log.exception("Error registrando usuario/dispositivo: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al registrar: {str(exc)}"
        )


@router.get("/devices/{device_id}", response_model=DeviceProfile)
def get_device(device_id: str):
    dev = repository.get_device(device_id)
    if not dev:
        raise HTTPException(status_code=404, detail="Device not found")
    return dev


@router.put("/devices/{device_id}")
def update_device(device_id: str, payload: DeviceProfileUpdate):
    ok = repository.update_device(
        device_id=device_id,
        medical_notes=payload.medical_notes,
        conditions=payload.conditions,
        medications=payload.medications,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Device not found")
    return {"status": "ok"}
