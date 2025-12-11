"""
Predictors for intent, sentiment, emotion, NER, plus safety gate and decoder.
Uses local transformers/spacy models and optional OpenAI for generation.
"""

import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import spacy
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from transformers import pipeline

from ..core.config import settings

load_dotenv()

logger = logging.getLogger(__name__)

OPENAI_MODEL = settings.openai_model
OPENAI_API_KEY = settings.openai_api_key
OPENAI_CLIENT: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
DECODER_USE_OPENAI = settings.decoder_use_openai
INTENT_MODEL_PATH = settings.intent_model_path
SENTIMENT_MODEL_PATH = settings.sentiment_model_path
EMOTION_MODEL_PATH = settings.emotion_model_path
NER_MODEL_PATH = settings.ner_model_path

INTENT_PIPELINE = None
SENTIMENT_PIPELINE = None
EMOTION_PIPELINE = None
NER_MODEL = None

INTENT_LABELS = {
    0: "agradecimiento",
    1: "alerta_medica",
    2: "comando_dispositivo",
    3: "configuracion_asistente",
    4: "consulta_informacion",
    5: "conversacion_social",
    6: "despedida",
    7: "emergencia_medica",
    8: "informacion_personal",
    9: "monitoreo_salud",
    10: "motivacion_personal",
    11: "no_entendido",
    12: "recordatorio",
    13: "reporte_emocional",
    14: "saludo",
}

SENTIMENT_LABELS = {0: "NEG", 1: "NEU", 2: "POS"}

EMOTION_LABELS = {0: "alegria", 1: "asco", 2: "enojo", 3: "miedo", 4: "sorpresa", 5: "tristeza"}


def _rng(seed: str) -> random.Random:
    return random.Random(seed)


def _load_hf_pipeline(model_path: str, task: str):
    if not model_path or not os.path.isdir(model_path):
        raise RuntimeError(f"{task}: ruta no valida: {model_path}")
    pl = pipeline(task, model=model_path, tokenizer=model_path, top_k=None)
    logger.info("%s: modelo cargado desde %s", task, model_path)
    return pl


def _load_intent_pipeline() -> None:
    global INTENT_PIPELINE
    INTENT_PIPELINE = _load_hf_pipeline(INTENT_MODEL_PATH, "text-classification")


def _load_sentiment_pipeline() -> None:
    global SENTIMENT_PIPELINE
    SENTIMENT_PIPELINE = _load_hf_pipeline(SENTIMENT_MODEL_PATH, "text-classification")


def _load_emotion_pipeline() -> None:
    global EMOTION_PIPELINE
    EMOTION_PIPELINE = _load_hf_pipeline(EMOTION_MODEL_PATH, "text-classification")


def _load_ner_model() -> None:
    global NER_MODEL
    if not NER_MODEL_PATH or not os.path.isdir(NER_MODEL_PATH):
        raise RuntimeError(f"NER: ruta no valida: {NER_MODEL_PATH}")
    NER_MODEL = spacy.load(NER_MODEL_PATH)
    logger.info("NER: modelo cargado desde %s", NER_MODEL_PATH)


def load_models() -> None:
    _load_intent_pipeline()
    _load_sentiment_pipeline()
    _load_emotion_pipeline()
    _load_ner_model()
    logger.info("Modelos cargados correctamente")


def _map_label(label: str, mapping: Dict[int, str]) -> str:
    if label.startswith("LABEL_"):
        idx = int(label.split("_")[1])
        return mapping.get(idx, label)
    return label


def _top_k_from_result(result, mapping: Dict[int, str], k: int = 3) -> List[Dict[str, float]]:
    if result and isinstance(result[0], list):
        result = result[0]
    sorted_res = sorted(result, key=lambda x: x["score"], reverse=True)
    top_items = sorted_res[:k]
    return [
        {"label": _map_label(item["label"], mapping), "score": round(float(item["score"]), 3)}
        for item in top_items
    ]


def _ensure_loaded() -> None:
    if not all([INTENT_PIPELINE, SENTIMENT_PIPELINE, EMOTION_PIPELINE, NER_MODEL]):
        raise RuntimeError("Model pipelines not loaded")


def predict_intent(text: str) -> Dict[str, float]:
    _ensure_loaded()
    result = INTENT_PIPELINE(text)
    top_k = _top_k_from_result(result, INTENT_LABELS, k=3)
    top = top_k[0]
    return {"label": top["label"], "score": top["score"], "top_k": top_k}


def predict_sentiment(text: str) -> Dict[str, float]:
    _ensure_loaded()
    result = SENTIMENT_PIPELINE(text)
    top_k = _top_k_from_result(result, SENTIMENT_LABELS, k=3)
    top = top_k[0]
    return {"label": top["label"], "score": top["score"], "top_k": top_k}


def _openai_emotion_guess(text: str) -> Optional[str]:
    """Use OpenAI as a lightweight validator for emotion before local model."""
    if not OPENAI_CLIENT:
        return None
    try:
        prompt = (
            "Clasifica la emocion principal del texto en: alegria, asco, enojo, miedo, sorpresa, tristeza."
            " Responde solo una palabra de esa lista."
            f"\nTexto: {text}"
        )
        res = OPENAI_CLIENT.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Responde solo con la emocion en minusculas, sin explicaciones."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=5,
            temperature=0,
        )
        content = (res.choices[0].message.content or "").strip().lower()
        word = content.split()[0].strip(".,;:") if content else ""
        if word in EMOTION_LABELS.values():
            return word
        return None
    except Exception as exc:  # pragma: no cover
        logger.warning("OpenAI emotion validator fallo, se usa modelo local: %s", exc)
        return None


def predict_emotion(text: str) -> Dict[str, float]:
    _ensure_loaded()
    validator_label = _openai_emotion_guess(text)
    result = EMOTION_PIPELINE(text)
    top_k_model = _top_k_from_result(result, EMOTION_LABELS, k=3)
    combined: List[Dict[str, float]] = []
    if validator_label:
        combined.append({"label": validator_label, "score": 0.85})
    for item in top_k_model:
        if any(c["label"] == item["label"] for c in combined):
            continue
        combined.append(item)
    top = combined[0]
    return {"label": top["label"], "score": round(float(top.get("score", 0)), 3), "top_k": combined[:3]}


def predict_ner(text: str) -> List[Dict[str, float]]:
    _ensure_loaded()
    doc = NER_MODEL(text)
    ents: List[Dict[str, float]] = []
    for ent in doc.ents:
        ents.append({"type": ent.label_, "value": ent.text, "score": 0.0})
    return ents


def safety_gate(text: str) -> Dict[str, Any]:
    if not DECODER_USE_OPENAI or not OPENAI_CLIENT:
        return {"allow": True, "emergency": False, "reason": "gate-bypass", "cls": "bypass"}
    try:
        prompt = (
            "Clasifica el texto en: normal, abuso/spam, emergencia_medica, autolesion."
            " Devuelve solo una palabra de la clase y una breve razon."
            f"\nTexto: {text}"
        )
        res = OPENAI_CLIENT.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Eres un clasificador breve. Responde en una sola linea."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=50,
            temperature=0,
        )
        content = (res.choices[0].message.content or "").strip()
        first_word = content.split()[0].lower().strip(".,;:") if content else "normal"
        cls = first_word if first_word in {"abuso", "spam", "emergencia_medica", "autolesion"} else "normal"
        if cls == "emergencia_medica":
            cls = "emergencia"
        allow = cls == "normal"
        emergency = cls in {"emergencia", "autolesion"}
        return {"allow": allow, "emergency": emergency, "reason": content, "cls": cls}
    except Exception as exc:  # pragma: no cover
        logger.warning("Gate fallback (bypass) por error: %s", exc)
        return {"allow": True, "emergency": False, "reason": "gate-error-bypass", "cls": "bypass"}


def generate_reply(
    context: Dict[str, object], 
    recent_messages=None, 
    medical_notes: Optional[str] = None,
    conditions: Optional[list] = None
) -> Tuple[str, str]:
    intent = context.get("intent", {}).get("label", "acompanamiento")
    intent_score = float(context.get("intent", {}).get("score", 0))
    sentiment = str(context.get("sentiment", {}).get("label", "NEU")).upper()
    emotion = context.get("emotion", {}).get("label", "neutral")
    entities = context.get("entities") or []
    user_text = context.get("text", "")
    names = [f"{e['type']}:{e['value']}" for e in entities] if entities else []
    ent_text = " Veo que mencionas " + ", ".join(names) + "." if names else ""

    if sentiment == "NEG":
        tone = "Estoy contigo; te apoyo"
    elif sentiment == "NEU":
        tone = "Estoy aqui para ayudarte"
    else:
        tone = "Estoy listo para ayudarte"

    if DECODER_USE_OPENAI and OPENAI_CLIENT:
        try:
            # Construir historial de conversación para contexto
            conversation_history = ""
            if recent_messages:
                history_lines = []
                for msg in recent_messages[-10:]:  # Últimos 10 mensajes para contexto
                    try:
                        role = msg.role if hasattr(msg, 'role') else 'unknown'
                        msg_text = msg.text if hasattr(msg, 'text') else str(msg)
                        if role in ('user', 'assistant'):
                            history_lines.append(f"{role.upper()}: {msg_text[:200]}")  # Limitar por largo
                    except:
                        pass
                if history_lines:
                    conversation_history = "\nHistorial de conversación:\n" + "\n".join(history_lines)
            
            # Construir contexto de salud
            health_context = ""
            if medical_notes or conditions:
                health_lines = []
                if medical_notes:
                    health_lines.append(f"NOTAS MÉDICAS: {medical_notes}")
                if conditions:
                    conditions_str = ", ".join(conditions) if isinstance(conditions, list) else str(conditions)
                    health_lines.append(f"CONDICIONES: {conditions_str}")
                if health_lines:
                    health_context = "\nPerfil de Salud del Usuario:\n" + "\n".join(health_lines)
            
            prompt = (
                "Eres un asistente virtual (estilo ChatGPT) para adultos mayores. Responde en espanol, 2-5 frases max,"
                " empatico, claro y util. Prioriza responder la peticion concreta del usuario."
                " Reglas: 1) No inventes datos; 2) Si hay riesgo/emergencia, sugiere contactar a alguien o servicios y haz 1-2 preguntas de verificacion;"
                " 3) Evita lenguaje tecnico y disculpas vacias; 4) La emocion solo ajusta el tono, no la accion;"
                " 5) Social/juego: conversa y sugiere 2-3 ideas sencillas; 6) Consulta/ayuda: responde directo y un paso siguiente;"
                " 7) Si intencion baja o 'no_entendido', usa el texto del usuario para cumplir lo que pide explicitamente."
                " 8) IMPORTANTE: Usa el contexto de mensajes previos para recordar información personal que el usuario mencionó."
                " 9) CRÍTICO: Si el usuario tiene condiciones médicas o notas especiales, ajusta tus respuestas considerando la seguridad y riesgos."
                f"\nContexto:"
                f"\n- Intencion: {intent} (score {intent_score})"
                f"\n- Sentimiento: {sentiment}"
                f"\n- Emocion: {emotion}"
                f"\n- Entidades: {', '.join(names) if names else 'ninguna'}"
                f"\n- Texto del usuario: {user_text}"
                f"{conversation_history}"
                f"{health_context}"
            )
            logger.info("Decoder: usando OpenAI model=%s con historial=%d y perfil_salud=%s", 
                       OPENAI_MODEL, len(recent_messages or []), bool(health_context))
            result = OPENAI_CLIENT.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Asistente empatico para adultos mayores. Se util, conversacional y claro."
                            " Prioriza seguridad si hay riesgo, y responde a la peticion de forma util."
                            " IMPORTANTE: Usa la información de mensajes previos para responder preguntas sobre el usuario"
                            " (ej: si el usuario dijo 'Me llamo Juan' antes, responde 'Te llamas Juan' cuando lo pregunte)."
                            " CRÍTICO: Si conoces sus condiciones médicas o notas de salud, usa esa información para dar respuestas más seguras"
                            " (ej: si es diabético, sé cauteloso con recomendaciones dietéticas)."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=settings.decoder_max_tokens,
                temperature=0.6,
            )
            return result.choices[0].message.content.strip(), "openai"
        except OpenAIError as exc:
            logger.warning("Fallo OpenAI, usando mock: %s", exc)
        except Exception as exc:  # pragma: no cover
            logger.warning("Fallo OpenAI inesperado, usando mock: %s", exc)
    else:
        logger.info("Decoder: usando mock (DECODER_USE_OPENAI desactivado o sin OPENAI_API_KEY)")

    return (
        f"{tone}. Puedo ayudarte con {intent}. {ent_text}Dime que necesitas y lo hacemos juntos."
    ), "mock"


if OPENAI_CLIENT:
    logger.info("OpenAI client inicializado con modelo %s", OPENAI_MODEL)
else:
    logger.info("OpenAI client no inicializado (sin OPENAI_API_KEY)")
