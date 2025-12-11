import logging
from fastapi import APIRouter, HTTPException, Query, status

from ..repositories import repository
from ..schemas import HealthMetricCreate, HealthMetricRecord

router = APIRouter()
log = logging.getLogger(__name__)


@router.post("/health", response_model=HealthMetricRecord, status_code=status.HTTP_201_CREATED)
def create_health_metric(payload: HealthMetricCreate):
    user = repository.get_user(payload.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    device = None
    if payload.device_id:
        device = repository.get_device(payload.device_id)
        if not device:
            raise HTTPException(status_code=404, detail="Device not found")
        if device.user_id and device.user_id != payload.user_id:
            raise HTTPException(status_code=400, detail="Device does not belong to user")
    record_id = repository.create_health_metric(
        user_id=payload.user_id,
        device_id=payload.device_id,
        metric=payload.metric,
        value=payload.value,
        unit=payload.unit,
        value_text=payload.value_text,
        meta=payload.meta,
        measured_at=payload.measured_at,
    )
    record = repository.get_health_metric(record_id)
    return record


@router.get("/health", response_model=list[HealthMetricRecord])
def list_health_metrics(
    user_id: str = Query(..., description="User owner of the records"),
    metric: str | None = Query(default=None, description="Filter by metric name"),
    device_id: str | None = Query(default=None, description="Filter by device id"),
    limit: int = Query(default=100, ge=1, le=500),
):
    user = repository.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if device_id:
        device = repository.get_device(device_id)
        if not device:
            raise HTTPException(status_code=404, detail="Device not found")
        if device.user_id and device.user_id != user_id:
            raise HTTPException(status_code=400, detail="Device does not belong to user")
    return repository.list_health_metrics(user_id=user_id, metric=metric, device_id=device_id, limit=limit)
