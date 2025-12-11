"""
SQLModel table definitions for persistence.
"""

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


class User(SQLModel, table=True):
    id: str = Field(primary_key=True)
    full_name: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Device(SQLModel, table=True):
    device_id: str = Field(primary_key=True)
    user_id: Optional[str] = Field(default=None, foreign_key="user.id", index=True)
    contact_name: Optional[str] = None
    contact_phone: Optional[str] = None
    medical_notes: Optional[str] = None
    conditions: Optional[Any] = Field(default=None, sa_column=Column(JSON))
    medications: Optional[Any] = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Session(SQLModel, table=True):
    id: str = Field(primary_key=True)
    device_id: str = Field(index=True)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    last_flow: Optional[str] = None
    context: Optional[Any] = Field(default=None, sa_column=Column(JSON))


class Message(SQLModel, table=True):
    id: str = Field(primary_key=True)
    device_id: str = Field(index=True)
    session_id: Optional[str] = Field(default=None, index=True)
    role: str
    text: str
    # Unified ML analysis metadata
    ml_analysis: Optional[Any] = Field(default=None, sa_column=Column(JSON))
    # Unified flow decision metadata
    flow_metadata: Optional[Any] = Field(default=None, sa_column=Column(JSON))
    entities: Optional[Any] = Field(default=None, sa_column=Column(JSON))
    emergency: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Reminder(SQLModel, table=True):
    id: str = Field(primary_key=True)
    device_id: str = Field(index=True)
    title: Optional[str] = None
    due_at: Optional[datetime] = None
    timezone: Optional[str] = None
    status: str = Field(default="draft", index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class EmergencyEvent(SQLModel, table=True):
    id: str = Field(primary_key=True)
    device_id: str = Field(index=True)
    status: str = Field(default="detected", index=True)
    contact_name: Optional[str] = None
    reason: Optional[str] = None
    action: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
