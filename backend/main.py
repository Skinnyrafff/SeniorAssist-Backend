"""
FastAPI backend for the elder assistant demo (mocked).
Pipeline uses mock predictors from predictors.py; swap them for real models later.
Run: uvicorn backend.main:app --reload --port 8000
"""

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import predictors


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Assistant Demo", version="0.1.0")


class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1, description="User message in Spanish")


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
    intent: LabeledScoreWithTopK
    sentiment: LabeledScoreWithTopK
    emotion: LabeledScoreWithTopK
    entities: List[Entity]
    reply: str
    decoder: str
    emergency: bool
    gate: str
    flow: Optional[str] = None
    next_prompt: Optional[str] = None
    processing_ms: int


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> Any:
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text is not allowed")

    start = time.perf_counter()
    log.info("Received text: %s", req.text)

    gate = predictors.safety_gate(req.text)

    try:
        intent = predictors.predict_intent(req.text)
        sentiment = predictors.predict_sentiment(req.text)
        emotion = predictors.predict_emotion(req.text)
        entities = predictors.predict_ner(req.text)
        # Heuristic: si el intent viene bajo y el texto sugiere recordatorio, forzar recordatorio
        text_low = req.text.lower()
        if intent.get("label") != "recordatorio" and float(intent.get("score", 0)) < 0.7:
            if any(keyword in text_low for keyword in ["recordatorio", "recordar", "anotar", "cita", "alarma", "recordarme"]):
                intent = {
                    "label": "recordatorio",
                    "score": 0.7,
                    "top_k": intent.get("top_k") or [{"label": "recordatorio", "score": 0.7}],
                }
        context = {
            "intent": intent,
            "sentiment": sentiment,
            "emotion": emotion,
            "entities": entities,
            "text": req.text,
        }
        reply, decoder_source = predictors.generate_reply(context)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(exc))

    emergency_flag = intent.get("label") in {"emergencia_medica", "alerta_medica"}
    if gate.get("emergency"):
        emergency_flag = True

    flow = None
    next_prompt = None
    intent_label = intent.get("label")
    intent_score = float(intent.get("score", 0))
    if emergency_flag:
        flow = "emergencia"
        next_prompt = "⚠ Emergencia detectada. ¿Llamo a tu contacto de emergencia o a servicios de urgencia?"
    elif intent_label == "recordatorio" and intent_score >= 0.6:
        flow = "recordatorio"
        next_prompt = "Entendido. ¿Qué debo recordar y a qué hora?"

    processing_ms = int((time.perf_counter() - start) * 1000)
    log.info(
        "Processed in %sms intent=%s sentiment=%s emotion=%s entities=%d emergency=%s decoder=%s",
        processing_ms,
        intent["label"],
        sentiment["label"],
        emotion["label"],
        len(entities),
        emergency_flag,
        decoder_source,
    )

    return {
        "intent": intent,
        "sentiment": sentiment,
        "emotion": emotion,
        "entities": entities,
        "reply": reply,
        "decoder": decoder_source,
        "emergency": emergency_flag,
        "gate": gate.get("reason", ""),
        "flow": flow,
        "next_prompt": next_prompt,
        "processing_ms": processing_ms,
    }
