from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    Request para /chat endpoint - Requerido por el Frontend Android.
    
    device_id: Identificador persistente del dispositivo (enviado en TODAS las requests)
    session_id: Identificador efímero de la sesión de chat (nuevo al entrar en pantalla de chat)
    text: Mensaje del usuario en español
    """
    text: str = Field(..., min_length=1, description="User message in Spanish")
    device_id: str = Field(..., min_length=1, description="Persistent device identifier")
    session_id: str = Field(..., min_length=1, description="Ephemeral chat session identifier")


class LabeledScore(BaseModel):
    label: str
    score: float


class LabeledScoreWithTopK(LabeledScore):
    top_k: List[LabeledScore] | None = None


class Entity(BaseModel):
    type: str
    value: str
    score: float


class ChatResponse(BaseModel):
    """
    Respuesta de conversación para el Frontend Android.
    
    ⚠️ CAMPOS CRÍTICOS para Frontend:
    - reply: Texto a mostrar y reproducir con TTS
    - emergency: Booleano que dispara alerta visual de emergencia
    - flow: Tipo de intención detectada
    - emergency_event_id: ID de evento si emergency=true
    
    ℹ️ CAMPOS OPCIONALES para enriquecer UI:
    - intent, sentiment, emotion, entities: Análisis lingüístico
    
    📊 CAMPOS INTERNOS (debug/analytics):
    - processing_ms, gate, flow_source, flow_reason, next_prompt
    """
    # === CAMPOS CRÍTICOS ===
    reply: str
    emergency: bool
    flow: Optional[str] = None
    emergency_event_id: Optional[str] = None
    
    # === CAMPOS OPCIONALES PARA FRONTEND ===
    intent: LabeledScoreWithTopK
    sentiment: LabeledScoreWithTopK
    emotion: LabeledScoreWithTopK
    entities: List[Entity]
    
    # === CAMPOS INTERNOS (DEBUG/ANALYTICS) ===
    gate: str
    flow_source: Optional[str] = None
    flow_reason: Optional[str] = None
    reminder_ids: Optional[List[str]] = None
    next_prompt: Optional[str] = None


class HistoryItem(BaseModel):
    id: str
    device_id: str
    session_id: Optional[str] = None
    role: str
    text: str
    # Unified ML analysis
    ml_analysis: Optional[dict] = None  # {intent: {label, score}, sentiment: {label, score}, emotion: {label, score}}
    # Unified flow metadata
    flow_metadata: Optional[dict] = None  # {assigned, reason, source}
    entities: Optional[Any] = None
    emergency: bool = False
    created_at: datetime
