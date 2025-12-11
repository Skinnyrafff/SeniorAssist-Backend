from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, ConfigDict


class HealthMetricBase(BaseModel):
    user_id: str = Field(..., description="User owning the measurement")
    device_id: Optional[str] = Field(default=None, description="Optional device source")
    metric: str = Field(..., min_length=1, description="Metric name, e.g., heart_rate, bp_systolic")
    value: Optional[float] = Field(default=None, description="Numeric value when applicable")
    unit: Optional[str] = Field(default=None, description="Unit of measure")
    value_text: Optional[str] = Field(default=None, description="Textual value if not numeric or to complement")
    meta: Optional[Any] = None
    measured_at: Optional[datetime] = Field(default=None, description="When the measurement occurred")


class HealthMetricCreate(HealthMetricBase):
    pass


class HealthMetricRecord(HealthMetricBase):
    id: str
    created_at: datetime
    measured_at: datetime
    model_config = ConfigDict(from_attributes=True)
