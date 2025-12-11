from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class UserBase(BaseModel):
    full_name: Optional[str] = Field(default=None, max_length=255)


class UserCreate(UserBase):
    full_name: str = Field(..., min_length=1, max_length=255)


class UserUpdate(UserBase):
    pass


class UserProfile(UserBase):
    id: str
    model_config = ConfigDict(from_attributes=True)
