"""
Predictors for intent, sentiment, emotion, NER, plus safety gate and decoder.
Uses local transformers/spacy models and optional OpenAI for generation.
"""

import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from transformers import pipeline
import spacy

load_dotenv()

logger = logging.getLogger(__name__)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CLIENT: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
DECODER_USE_OPENAI = os.getenv("DECODER_USE_OPENAI", "false").lower() == "true"
INTENT_MODEL_PATH = os.getenv("INTENT_MODEL_PATH", "./models/intention/robertuito_finetuned")
SENTIMENT_MODEL_PATH = os.getenv("SENTIMENT_MODEL_PATH", "./models/sentiment/dccuchile_bert_spanish_finetuned")
EMOTION_MODEL_PATH = os.getenv(
    "EMOTION_MODEL_PATH", "./models/emotion/pysentimiento_robertuito_sentiment_analysis_finetuned"
)
NER_MODEL_PATH = os.getenv("NER_MODEL_PATH", "./models/ner/spacy/model_es_ner")

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


def predict_intent(text: str) -> Dict[str, float]:
    result = INTENT_PIPELINE(text)
    top_k = _top_k_from_result(result, INTENT_LABELS, k=3)
    top = top_k[0]
    return {"label": top["label"], "score": top["score"], "top_k": top_k}


def predict_sentiment(text: str) -> Dict[str, float]:
    result = SENTIMENT_PIPELINE(text)
    top_k = _top_k_from_result(result, SENTIMENT_LABELS, k=3)
    top = top_k[0]
    return {"label": top["label"], "score": top["score"], "top_k": top_k}


def predict_emotion(text: str) -> Dict[str, float]:
    result = EMOTION_PIPELINE(text)
    top_k = _top_k_from_result(result, EMOTION_LABELS, k=3)
    top = top_k[0]
    return {"label": top["label"], "score": top["score"], "top_k": top_k}


def predict_ner(text: str) -> List[Dict[str, float]]:
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
    except Exception as exc:
        logger.warning("Gate fallback (bypass) por error: %s", exc)
        return {"allow": True, "emergency": False, "reason": "gate-error-bypass", "cls": "bypass"}


def generate_reply(context: Dict[str, object]) -> Tuple[str, str]:
    intent = context.get("intent", {}).get("label", "acompanamiento")
    intent_score = float(context.get("intent", {}).get("score", 0))
    sentiment = context.get("sentiment", {}).get("label", "neu")
    emotion = context.get("emotion", {}).get("label", "neutral")
    entities = context.get("entities") or []
    user_text = context.get("text", "")
    names = [f"{e['type']}:{e['value']}" for e in entities] if entities else []
    ent_text = " Veo que mencionas " + ", ".join(names) + "." if names else ""
    tone = "Lamento lo que sientes" if sentiment == "neg" else "Me alegra escucharte"

    if DECODER_USE_OPENAI and OPENAI_CLIENT:
        try:
            prompt = (
                "Eres un asistente virtual para adultos mayores. Responde en espanol, breve (1-3 frases),"
                " con tono calido, claro y practico. Reglas: 1) No inventes datos; 2) Si detectas riesgo"
                " o peticion de ayuda urgente, sugiere avisar a un contacto o servicios de emergencia y"
                " preguntas de verificacion; 3) Evita lenguaje tecnico; 4) Evita medicalizar sin contexto,"
                " ofrece apoyo y pasos simples; 5) Si hay entidades relevantes, usalas para personalizar;"
                " 6) Solo usa la emocion como tono, no cambies la accion ni el contenido por la emocion;"
                " 7) Prioriza la intencion y el sentimiento; si la intencion es 'no_entendido' o score < 0.6,"
                " usa el texto del usuario para cumplir la peticion explicita."
                f"\nContexto:"
                f"\n- Intencion: {intent} (score {intent_score})"
                f"\n- Sentimiento: {sentiment}"
                f"\n- Emocion: {emotion}"
                f"\n- Entidades: {', '.join(names) if names else 'ninguna'}"
                f"\n- Texto del usuario: {user_text}"
            )
            logger.info("Decoder: usando OpenAI model=%s", OPENAI_MODEL)
            result = OPENAI_CLIENT.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Asistente empatico para adultos mayores. Se breve, calido, y practico."
                            " Prioriza seguridad y apoyo emocional."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=150,
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
        f"{tone}. Estoy aqui para ayudarte con {intent} y noto una emocion de {emotion}."
        f"{ent_text} ¿Quieres que te apoye en algo mas?"
    ), "mock"


if OPENAI_CLIENT:
    logger.info("OpenAI client inicializado con modelo %s", OPENAI_MODEL)
else:
    logger.info("OpenAI client no inicializado (sin OPENAI_API_KEY)")

_load_intent_pipeline()
_load_sentiment_pipeline()
_load_emotion_pipeline()
_load_ner_model()
