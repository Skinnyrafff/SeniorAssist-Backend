"""
Conversation orchestrator that combines local predictors with optional LLM assist
to decide flows (emergencia, recordatorio, acompanamiento_social, consulta_informacion).
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from ..adapters import predictors
from ..repositories import repository
from ..core.config import settings
from .reminder_extractor import extract_reminders

logger = logging.getLogger(__name__)

EMERGENCY_LABELS = {"emergencia_medica", "alerta_medica"}
RECORDATORIO_KEYWORDS = ["recordatorio", "recordar", "anotar", "cita", "alarma", "recordarme"]
ACOMP_LABELS = {
    "conversacion_social",
    "saludo",
    "despedida",
    "motivacion_personal",
    "reporte_emocional",
    "agradecimiento",
    "no_entendido",
}
CONSULTA_LABELS = {
    "consulta_informacion",
    "informacion_personal",
    "monitoreo_salud",
    "configuracion_asistente",
    "comando_dispositivo",
}
ALLOWED_FLOWS = {"emergencia", "recordatorio", "acompanamiento_social", "consulta_informacion"}

FLOW_USE_LLM = settings.flow_use_llm
FLOW_LLM_MODEL = settings.flow_llm_model or settings.openai_model
FLOW_LLM_THRESHOLD = settings.flow_llm_threshold


def _keyword_recordatorio(text: str) -> bool:
    text_low = text.lower()
    return any(k in text_low for k in RECORDATORIO_KEYWORDS)


def _is_cancel(text: str) -> bool:
    txt = text.lower()
    cancel_words = ["cancelar", "cancela", "olvida", "olvidalo", "anular", "no gracias", "no, gracias"]
    return any(cw in txt for cw in cancel_words)


def _is_confirm(text: str) -> bool:
    txt = text.lower()
    confirm_words = ["confirmo", "confirma", "confirmar", "si", "sí", "dale", "ok", "okey", "vale", "de acuerdo"]
    return any(cw in txt for cw in confirm_words)


def _reminder_slots(text: str) -> Dict[str, bool]:
    """Heuristica ligera para saber si el usuario ya dio contenido o tiempo."""
    txt = text.lower()
    has_time = any(token in txt for token in ["manana", "hoy", "tarde", "noche", "am", "pm", "minuto", "hora", "horas"])
    import re

    if re.search(r"\b\d{1,2}:\d{2}\b", txt) or re.search(r"\b\d{1,2}\s*(am|pm)\b", txt):
        has_time = True
    words = [w for w in txt.replace(",", " ").split() if w not in RECORDATORIO_KEYWORDS]
    has_content = len(words) >= 4  # algo mas que "necesito un recordatorio"
    return {"has_time": has_time, "has_content": has_content}


@dataclass
class FlowDecision:
    flow: str
    next_prompt: Optional[str]
    reason: str
    source: str = "local"


class FlowAssistant:
    """Optional LLM-based assistant to decide flow independently from local ML."""

    def __init__(self, enabled: bool = FLOW_USE_LLM, threshold: float = FLOW_LLM_THRESHOLD):
        self.enabled = enabled and bool(predictors.OPENAI_CLIENT)
        self.threshold = threshold
        self.llm_timeout = 5.0  # Timeout: si demora más, usa fallback

    def suggest(
        self,
        text: str,
        intent: Dict[str, Any],
        sentiment: Dict[str, Any],
        emotion: Dict[str, Any],
        entities: Sequence[Dict[str, Any]],
        recent_messages: Optional[List[Any]] = None,
    ) -> Optional[FlowDecision]:
        """
        LLM valida la decisión del modelo ML con contexto conversacional.
        Si local y LLM discrepan → LLM corrige (por overfitting en local).
        Si LLM timeout → mantiene local.
        
        Esto preserva los modelos ML en tesis pero protege de errores.
        """
        if not self.enabled:
            return None

        try:
            # LLM revisa el texto crudo y la decisión de local
            local_intent = intent.get("label", "")
            local_score = float(intent.get("score", 0))
            
            # Construir contexto conversacional
            context_str = ""
            if recent_messages:
                context_lines = []
                for msg in recent_messages[-5:]:  # Últimos 5 mensajes max
                    role = msg.role if hasattr(msg, 'role') else 'unknown'
                    msg_text = msg.text if hasattr(msg, 'text') else str(msg)
                    context_lines.append(f"{role}: {msg_text}")
                if context_lines:
                    context_str = "Contexto previo:\n" + "\n".join(context_lines) + "\n\n"
            
            prompt = (
                f"{context_str}"
                f"Mensaje actual: '{text}'\n"
                f"ML detectó: {local_intent} (confianza {local_score:.2f})\n"
                f"Sentimiento: {sentiment.get('label')}, Emoción: {emotion.get('label')}\n\n"
                "Valida si es correcto usando el contexto. Flujos:\n"
                "- emergencia: peligro inmediato, dolor severo, caída, dificultad respiratoria\n"
                "- recordatorio: usuario menciona tomar/hacer algo + hora/momento específico (incluso en mensajes separados)\n"
                "- consulta_informacion: preguntas sobre salud, consejos, información general\n"
                "- acompanamiento_social: charla, emociones, compañía, saludos\n\n"
                "IMPORTANTE: Si en mensajes previos mencionó una acción (ej: tomar medicina) y ahora da hora/momento, "
                "es un RECORDATORIO aunque no lo diga explícitamente.\n"
                "JSON: {\"flow\": \"...\", \"next_prompt\": \"...\", \"corrected\": bool}"
            )
            res = predictors.OPENAI_CLIENT.chat.completions.create(
                model=FLOW_LLM_MODEL,
                messages=[
                    {"role": "system", "content": "Eres validador de intención para adultos mayores. Usa contexto para detectar recordatorios implícitos. Responde SOLO JSON."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=100,
                temperature=0.1,
                timeout=self.llm_timeout,
            )
            content = (res.choices[0].message.content or "").strip()
            
            # Extraer JSON si viene con markdown ```json...```
            if content.startswith("```"):
                lines = content.split("\n")
                json_lines = [line for line in lines[1:-1] if not line.startswith("```")]
                content = "\n".join(json_lines).strip()
            
            data = json.loads(content)
            flow = str(data.get("flow", "")).strip().lower()
            if flow not in ALLOWED_FLOWS:
                return None
            next_prompt = data.get("next_prompt")
            corrected = bool(data.get("corrected", False))
            
            if corrected:
                logger.info("LLM CORRECTED local: %s → %s (overfitting detected)", local_intent, flow)
            else:
                logger.info("LLM validated local: %s confirmed", local_intent)
            
            reason = f"llm_validated (corrected={corrected})"
            return FlowDecision(flow=flow, next_prompt=next_prompt, reason=reason, source="llm")
        except Exception as exc:  # pragma: no cover
            logger.warning("LLM validation failed, keeping local ML: %s", exc)
            return None  # None → merge() usará solo local


class ConversationOrchestrator:
    def __init__(self, assistant: Optional[FlowAssistant] = None):
        self.assistant = assistant or FlowAssistant()

    def decide_flow_local(
        self, text: str, intent: Dict[str, Any], gate: Dict[str, Any], entities: Sequence[Dict[str, Any]]
    ) -> FlowDecision:
        label = intent.get("label") or "no_entendido"
        score = float(intent.get("score", 0))
        gate_emergency = bool(gate.get("emergency"))
        has_emergency_intent = label in EMERGENCY_LABELS
        recordatorio_by_keyword = _keyword_recordatorio(text)

        if gate_emergency or has_emergency_intent:
            return FlowDecision(
                flow="emergencia",
                next_prompt="Emergencia detectada. Llamo a tu contacto de emergencia o a servicios de urgencia?",
                reason="gate_emergency" if gate_emergency else "intent_emergency",
            )

        if label == "recordatorio" and score >= 0.6:
            return FlowDecision(
                flow="recordatorio",
                next_prompt="Entendido. Que debo recordar y a que hora?",
                reason="intent_recordatorio",
            )

        if recordatorio_by_keyword and score < 0.7:
            return FlowDecision(
                flow="recordatorio",
                next_prompt="Te ayudo a guardar un recordatorio. Que debo recordar y cuando?",
                reason="keyword_recordatorio",
            )

        if label in CONSULTA_LABELS:
            return FlowDecision(
                flow="consulta_informacion",
                next_prompt="Claro, dime que informacion necesitas y te apoyo.",
                reason="intent_consulta",
            )

        if label in ACOMP_LABELS:
            return FlowDecision(
                flow="acompanamiento_social",
                next_prompt="Estoy aqui contigo. Quieres contarme mas o necesitas apoyo en algo?",
                reason="intent_acompanamiento",
            )

        return FlowDecision(
            flow="acompanamiento_social",
            next_prompt="Estoy aqui para ayudarte. Como puedo apoyarte?",
            reason="fallback_acompanamiento",
        )

    def merge_flow_decisions(
        self,
        local_decision: FlowDecision,
        llm_decision: Optional[FlowDecision],
        intent_score: float,
    ) -> FlowDecision:
        """
        Combina: Local ML (tu tesis) + LLM (validador/corrector de overfitting).
        
        Estrategia:
        - Local ML es la base (preserva tu trabajo)
        - LLM valida y corrige si detecta overfitting
        - Si LLM falla (timeout) → mantiene local
        """
        if not llm_decision:
            # LLM no respondió o timeout → usa local
            return local_decision

        local_flow = local_decision.flow
        llm_flow = llm_decision.flow

        # Caso 1: LLM validó local → Consenso fuerte
        if local_flow == llm_flow:
            return FlowDecision(
                flow=local_flow,
                next_prompt=local_decision.next_prompt,
                reason=f"{local_decision.reason}+llm_validated",
                source="hybrid+validated",
            )

        # Caso 2: EMERGENCIA detectada por cualquiera → Escala (seguridad)
        if local_flow == "emergencia" or llm_flow == "emergencia":
            return FlowDecision(
                flow="emergencia",
                next_prompt=llm_decision.next_prompt or local_decision.next_prompt,
                reason=f"emergencia_{llm_flow if llm_flow == 'emergencia' else 'local'}",
                source="hybrid+safety",
            )

        # Caso 3: RECORDATORIO detectado por cualquiera → Preserva datos
        if local_flow == "recordatorio" or llm_flow == "recordatorio":
            return FlowDecision(
                flow="recordatorio",
                next_prompt=llm_decision.next_prompt or local_decision.next_prompt,
                reason=f"recordatorio_{llm_flow if llm_flow == 'recordatorio' else 'local'}",
                source="hybrid+preservation",
            )

        # Caso 4: LLM corrigió local (por overfitting probable)
        # Confía en la corrección LLM
        return FlowDecision(
            flow=llm_flow,
            next_prompt=llm_decision.next_prompt,
            reason=f"llm_corrected_local (local={local_flow}, llm={llm_flow})",
            source="hybrid+corrected",
        )

    def _handle_emergency(self, text: str, decision: FlowDecision) -> FlowDecision:
        txt = text.lower()
        cancel_words = ["falsa alarma", "ya estoy bien", "no llames", "no llames a nadie", "no es necesario", "todo bien"]
        confirm_words = ["llama", "llamar", "contacta", "contactar", "ambulancia", "urgencias", "emergencia", "911", "112"]
        if any(cw in txt for cw in cancel_words):
            decision.next_prompt = "Entendido, cancelo la alerta. Si vuelves a sentirte mal, avisa de inmediato."
            decision.reason = "emergencia_cancel"
        elif any(cw in txt for cw in confirm_words):
            decision.next_prompt = "Procedo a avisar a tu contacto de emergencia o servicios de urgencia. Confirmas?"
            decision.reason = "emergencia_confirm"
        else:
            decision.next_prompt = "Es una emergencia? Llamo a tu contacto de emergencia o a servicios de urgencia."
            decision.reason = "emergencia_check"
        return decision

    def process(
        self,
        text: str,
        *,
        device_id: str,
        session_id: Optional[str] = None,
        source_message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        gate = predictors.safety_gate(text)
        
        # Recuperar contexto conversacional temprano (para todos los flujos)
        recent_messages = []
        try:
            recent_messages = repository.list_messages(device_id=device_id, limit=10, session_id=session_id)
        except Exception as exc:  # pragma: no cover
            logger.warning("No se pudo recuperar historial: %s", exc)
        
        # Recuperar datos de salud del dispositivo para contexto de OpenAI
        medical_notes = None
        conditions = None
        try:
            device = repository.get_device(device_id)
            if device:
                medical_notes = device.medical_notes
                conditions = device.conditions
        except Exception as exc:  # pragma: no cover
            logger.warning("No se pudo recuperar datos de salud del dispositivo: %s", exc)

        if not gate.get("allow", True):
            processing_ms = int((time.perf_counter() - start) * 1000)
            logger.warning("Safety gate marcó como no permitido, pero se continuará para emergencia.")
            if gate.get("emergency"):
                # Forzar flujo de emergencia en lugar de bloquear
                intent = {"label": "emergencia_medica", "score": 1.0, "top_k": [{"label": "emergencia_medica", "score": 1.0}]}
                try:
                    sentiment = predictors.predict_sentiment(text)
                    emotion = predictors.predict_emotion(text)
                    entities = predictors.predict_ner(text)
                except Exception:
                    sentiment = {"label": "NEU", "score": 0.0, "top_k": []}
                    emotion = {"label": "neutral", "score": 0.0, "top_k": []}
                    entities = []
                decision = FlowDecision(
                    flow="emergencia",
                    next_prompt="Emergencia detectada. Llamo a tu contacto de emergencia o a servicios de urgencia?",
                    reason="gate_emergency",
                    source="gate",
                )
                reply, decoder_source = predictors.generate_reply(
                    {"intent": intent, "sentiment": sentiment, "emotion": emotion, "entities": entities, "text": text},
                    recent_messages=recent_messages,
                    medical_notes=medical_notes,
                    conditions=conditions
                )
                emergency_event_id = None
                try:
                    emergency_event_id = repository.create_emergency_event(
                        device_id=device_id,
                        session_id=session_id,
                        status="detected",
                        reason=decision.reason,
                        action=None,
                        contact_name=None,
                        meta=None,
                        source_message_id=source_message_id,
                    )
                except Exception as exc:  # pragma: no cover
                    logger.warning("No se pudo persistir evento de emergencia (gate): %s", exc)
                return {
                    "intent": intent,
                    "sentiment": sentiment,
                    "emotion": emotion,
                    "entities": entities,
                    "reply": reply,
                    "decoder": decoder_source,
                    "emergency": True,
                    "gate": gate.get("reason", ""),
                    "flow": decision.flow,
                    "next_prompt": decision.next_prompt,
                    "flow_source": decision.source,
                    "flow_reason": decision.reason,
                    "emergency_event_id": emergency_event_id,
                    "processing_ms": processing_ms,
                }
            else:
                return {
                    "intent": {"label": "bloqueado", "score": 0.0, "top_k": []},
                    "sentiment": {"label": "NEU", "score": 0.0, "top_k": []},
                    "emotion": {"label": "neutral", "score": 0.0, "top_k": []},
                    "entities": [],
                    "reply": "Mensaje bloqueado por seguridad. Si necesitas ayuda urgente, contacta a un servicio de emergencia.",
                    "decoder": "gate",
                    "emergency": gate.get("emergency", False),
                    "gate": gate.get("reason", ""),
                    "flow": "bloqueado",
                    "next_prompt": None,
                    "flow_source": "gate",
                    "flow_reason": "gate_block",
                    "processing_ms": processing_ms,
                }

        intent = predictors.predict_intent(text)
        sentiment = predictors.predict_sentiment(text)
        emotion = predictors.predict_emotion(text)
        entities = predictors.predict_ner(text)

        # Decisión local (rápida)
        local_decision = self.decide_flow_local(text, intent, gate, entities)

        # Decisión LLM en paralelo (si está habilitado)
        # LLM lee el texto RAW sin influencia de predicciones ML + contexto conversacional
        llm_decision = None
        if self.assistant.enabled:
            llm_decision = self.assistant.suggest(
                text, intent, sentiment, emotion, entities, recent_messages=recent_messages
            )

        # Combinar decisiones: local + LLM se complementan
        # LLM es independiente, detecta sesgos de overfitting en local
        intent_score = float(intent.get("score", 0))
        decision = self.merge_flow_decisions(local_decision, llm_decision, intent_score)
        flow_source = decision.source

        # Log detallado de la decisión
        if llm_decision:
            if llm_decision.source == "fallback":
                logger.warning(
                    "Using FALLBACK flow (LLM unavailable): %s [%s] - local=%s (score=%.2f)",
                    llm_decision.flow,
                    llm_decision.reason,
                    local_decision.flow,
                    intent_score,
                )
            elif llm_decision.flow != local_decision.flow:
                logger.info(
                    "Flow decision: local=%s (score=%.2f) vs llm=%s → final=%s [%s]",
                    local_decision.flow,
                    intent_score,
                    llm_decision.flow,
                    decision.flow,
                    decision.reason,
                )
            else:
                logger.info("Flow consensus: %s (local+llm agree)", decision.flow)
        else:
            logger.info("Using LOCAL flow only (LLM disabled): %s", local_decision.flow)

        reminder_ids: List[str] = []
        emergency_event_id: Optional[str] = None
        emergency_action: Optional[str] = None
        emergency_contact_name: Optional[str] = None

        if decision.flow == "emergencia":
            # Checar contacto de emergencia
            contact_phone = None
            try:
                dev = repository.get_device(device_id)
                if dev:
                    emergency_contact_name = dev.contact_name
                    contact_phone = dev.contact_phone
            except Exception as exc:  # pragma: no cover
                logger.warning("No se pudo obtener contacto de emergencia: %s", exc)

            if contact_phone:
                decision.next_prompt = (
                    f"Emergencia detectada. Llamo a tu contacto de emergencia ({emergency_contact_name or 'contacto'} al {contact_phone})?"
                )
                decision.reason = "emergencia_contacto"
                emergency_action = "contact_family"
            else:
                decision.next_prompt = "Emergencia detectada. Llamo a servicios de emergencia?"
                decision.reason = "emergencia_servicios"
                emergency_action = "call_services"

            decision = self._handle_emergency(text, decision)
        elif decision.flow == "recordatorio":
            if _is_cancel(text):
                decision.next_prompt = "Entendido, cancelo el recordatorio. Te ayudo con algo mas?"
                decision.reason = "recordatorio_cancel"
            else:
                slots = _reminder_slots(text)
                if slots["has_content"] and slots["has_time"]:
                    decision.next_prompt = "Listo. Confirmo el recordatorio asi? Si no, dime cancelar."
                    decision.reason = "recordatorio_confirm"
                elif slots["has_content"] and not slots["has_time"]:
                    decision.next_prompt = "Cuando debo recordartelo? (ej. hoy 5pm, manana 9:00)"
                    decision.reason = "recordatorio_pedir_hora"
                else:
                    decision.next_prompt = "Que debo recordar y cuando? (ej. tomar medicina a las 9am)"
                    decision.reason = "recordatorio_pedir_contenido"

        # Si es emergencia, NO generar respuesta con OpenAI, solo activar protocolo
        if decision.flow == "emergencia":
            reply = "Protocolo de emergencia activado."
            decoder_source = "emergency_protocol"
        else:
            reply, decoder_source = predictors.generate_reply(
                {"intent": intent, "sentiment": sentiment, "emotion": emotion, "entities": entities, "text": text},
                recent_messages=recent_messages,
                medical_notes=medical_notes,
                conditions=conditions
            )

        # Persist flows for reminders/emergencies
        try:
            if decision.flow == "recordatorio":
                status_map = {
                    "recordatorio_cancel": "cancelled",
                    "recordatorio_confirm": "draft",
                    "recordatorio_pedir_hora": "draft",
                    "recordatorio_pedir_contenido": "draft",
                    "intent_recordatorio": "draft",
                    "keyword_recordatorio": "draft",
                }
                target_status = status_map.get(decision.reason, "draft")
                if _is_confirm(text):
                    target_status = "confirmed"
                reminders = extract_reminders(text)
                if not reminders:
                    reminders = [{"title": text, "due_at": None, "timezone": None}]

                if target_status == "cancelled":
                    existing = repository.get_latest_reminder(device_id, session_id=session_id)
                    if existing:
                        repository.update_reminder_status(existing.id, target_status)
                        reminder_ids.append(existing.id)
                else:
                    for rem in reminders:
                        title = rem.get("title") or text
                        due_at = rem.get("due_at")
                        existing = repository.find_similar_reminder(
                            device_id=device_id, title=title, due_at=due_at, session_id=session_id
                        )
                        if existing:
                            repository.update_reminder_status(existing.id, target_status)
                            keep_id = existing.id
                        else:
                            keep_id = repository.create_reminder(
                                device_id=device_id,
                                session_id=session_id,
                                title=title,
                                due_at=due_at,
                                timezone=rem.get("timezone"),
                                status=target_status,
                                notes=None,
                                meta=None,
                                source_message_id=source_message_id,
                            )
                        reminder_ids.append(keep_id)
                        try:
                            repository.cleanup_reminder_duplicates(
                                device_id=device_id, title=title, due_at=due_at, keep_id=keep_id
                            )
                        except Exception as exc:  # pragma: no cover
                            logger.warning("No se pudo limpiar duplicados de recordatorio: %s", exc)
            elif decision.flow == "emergencia":
                status_map = {
                    "emergencia_cancel": "cancelled",
                    "emergencia_confirm": "escalated",
                    "emergencia_check": "confirming",
                    "emergencia_contacto": "confirming",
                    "emergencia_servicios": "confirming",
                    "gate_emergency": "detected",
                    "intent_emergency": "detected",
                }
                target_status = status_map.get(decision.reason, "detected")
                existing_event = repository.get_latest_open_emergency(device_id, session_id=session_id)
                if existing_event:
                    repository.update_emergency_status(existing_event.id, target_status, emergency_action)
                    emergency_event_id = existing_event.id
                else:
                    emergency_event_id = repository.create_emergency_event(
                        device_id=device_id,
                        session_id=session_id,
                        status=target_status,
                        reason=decision.reason,
                        action=emergency_action,
                        contact_name=emergency_contact_name,
                        meta=None,
                        source_message_id=source_message_id,
                    )
        except Exception as exc:  # pragma: no cover
            logger.warning("No se pudo persistir flujo %s: %s", decision.flow, exc)

        processing_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "Orchestrated flow=%s source=%s intent=%s decoder=%s T=%sms",
            decision.flow,
            flow_source,
            intent.get("label"),
            decoder_source,
            processing_ms,
        )

        return {
            "intent": intent,
            "sentiment": sentiment,
            "emotion": emotion,
            "entities": entities,
            "reply": reply,
            "decoder": decoder_source,
            "emergency": decision.flow == "emergencia" or bool(gate.get("emergency")),
            "gate": gate.get("reason", ""),
            "flow": decision.flow,
            "next_prompt": decision.next_prompt,
            "flow_source": flow_source,
            "flow_reason": decision.reason,
            "reminder_ids": reminder_ids or None,
            "emergency_event_id": emergency_event_id,
        }
