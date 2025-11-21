"""
Streamlit chat UI for the assistant demo.
Uses backend /chat and renders mensajes estilo chat (usuario derecha, asistente izquierda).
Run: streamlit run frontend/streamlit_app.py
"""

import os
from datetime import datetime
from typing import Any, Dict, List

import requests
import streamlit as st

# Opcional: TTS local
try:
    import pyttsx3

    HAS_TTS = True
except Exception:
    HAS_TTS = False
    pyttsx3 = None  # type: ignore

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/chat")


def post_chat(text: str) -> Dict[str, Any]:
    resp = requests.post(BACKEND_URL, json={"text": text}, timeout=15)
    resp.raise_for_status()
    return resp.json()


def pct(score: float) -> str:
    try:
        return f"{float(score) * 100:.1f}%"
    except Exception:
        return "--%"


st.set_page_config(page_title="Asistente Adultos Mayores", page_icon=":handshake:", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 860px; margin: 0 auto !important;}
    .stChatInput > div {max-width: 640px; margin: 18px auto 12px auto;}
    body {background-color:#0b0c0f; color:#e7e9ed;}
    .chat-bubble-user {background:#30343b; color:#f7f8fb; border-radius:16px; padding:12px 16px; border:1px solid #3d414a; max-width:62%; box-shadow:none; font-size:1.22rem;}
    .chat-bubble-bot {background:transparent; color:#e7e9ed; border-radius:16px; padding:6px 2px; max-width:78%; font-size:1.22rem;}
    .chat-meta {font-size:0.9em; color:#a8adb4; margin-top:8px; line-height:1.4;}
    .pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.85em; margin-right:6px;}
    .pill-intent {background:rgba(76, 130, 255, 0.14); color:#d7e3ff; border:1px solid rgba(76,130,255,0.3);}
    .pill-sent {background:rgba(52,211,153,0.14); color:#cef8e6; border:1px solid rgba(52,211,153,0.3);}
    .pill-emo {background:rgba(249,115,22,0.14); color:#fde7d5; border:1px solid rgba(249,115,22,0.3);}
    .pill-topk {color:#cfd4db;}
    .chat-row {display:flex; margin:18px 0;}
    .chat-row.user {justify-content:flex-end;}
    .chat-row.bot {justify-content:flex-start;}
    .tiny {font-size:0.85em; color:#9da3ab;}
    /* Barra de input refinada */
    .stChatInput > div {
        background:#181c23 !important;
        border:1.5px solid #23283a !important;
        border-radius:22px !important;
        padding:7px 18px 7px 18px !important;
        box-shadow:0 2px 8px rgba(0,0,0,0.10) inset, 0 1px 0 rgba(255,255,255,0.03);
        transition:border-color 0.18s cubic-bezier(.4,0,.2,1), box-shadow 0.18s cubic-bezier(.4,0,.2,1);
    }
    .stTextInput textarea, .stChatInput textarea {
        background:transparent !important;
        color:#e7e9ed !important;
        border:none !important;
        border-radius:16px !important;
        padding:5px 2px 5px 2px !important;
        font-size:1.08rem !important;
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif !important;
        resize:none !important;
        min-height:38px !important;
        max-height:80px !important;
    }
    .stChatInput textarea::placeholder {
        color:#a3a8b8 !important;
        opacity:0.7 !important;
        font-size:0.98rem !important;
    }
    .stChatInput textarea:focus {
        outline:none !important;
        box-shadow:none !important;
    }
    .stChatInput > div:focus-within {
        border-color:#4f8cff !important;
        box-shadow:0 0 0 2px rgba(79,140,255,0.18), 0 2px 8px rgba(0,0,0,0.13) inset;
    }
    .topk-line {font-size:0.85em; color:#cfd4db; margin-left:8px;}
    .voice-box {border:1px solid #23283a; border-radius:12px; padding:10px 12px; margin:8px 0; background:#11141a;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Asistente para Adultos Mayores")
st.caption("Flujo: intencion, sentimiento, emocion, NER y respuesta empatica.")

# Historial de chat
if "messages" not in st.session_state:
    st.session_state["messages"]: List[Dict[str, Any]] = []
if "history" not in st.session_state:
    st.session_state["history"]: List[Dict[str, Any]] = []
def now_hhmm() -> str:
    return datetime.now().strftime("%H:%M")


def render_topk(items: List[Dict[str, Any]]) -> str:
    if not items:
        return ""
    return " · " + " / ".join(
        f"<span class='pill-topk'>{it.get('label')} {pct(it.get('score', 0))}</span>" for it in items
    )


def tts_play(text: str) -> bool:
    if not HAS_TTS or not text:
        return False
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        engine.say(text)
        engine.runAndWait()
        return True
    except Exception:
        return False


class SpeechToTextProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer() if HAS_VOICE else None
        self.transcript = ""
        self._buffer = bytes()

    def recv_audio(self, frame):
        # frame is av.AudioFrame
        pcm = frame.to_ndarray().tobytes()
        self._buffer += pcm
        # Simple buffer length check (~0.5s chunks)
        if len(self._buffer) >= 16000 * 2 * 1:  # 1 second @16kHz, 16-bit
            self._transcribe()
            self._buffer = bytes()
        return frame

    def _transcribe(self):
        if not (self.recognizer and self._buffer):
            return
        audio_data = sr.AudioData(self._buffer, 16000, 2)
        try:
            result = self.recognizer.recognize_google(audio_data, language="es-ES")
            if result:
                self.transcript = result
        except Exception:
            pass

    def flush(self):
        if self._buffer:
            self._transcribe()
            self._buffer = bytes()



# layout centrado y estrecho
for idx, msg in enumerate(st.session_state["messages"]):
    role = msg.get("role", "assistant")
    content = msg.get("content", "")
    meta = msg.get("meta") or {}
    ts = msg.get("ts", "")
    if role == "user":
        st.markdown(
            f"""
            <div class='chat-row user'>
              <div class='chat-bubble-user'>{content}</div>
            </div>
            """,
            unsafe_allow_html=True,
            )
        if ts:
            st.markdown(f"<div class='tiny' style='text-align:right;'>{ts}</div>", unsafe_allow_html=True)
    else:
        intent = meta.get("intent", {}) if meta else {}
        sent = meta.get("sentiment", {}) if meta else {}
        emo = meta.get("emotion", {}) if meta else {}
        ents = meta.get("entities") if meta else []

        aux_html = ""
        if meta:
            aux_html = (
                "<div class='chat-meta'>"
                f"<span class='pill pill-intent'>Intent: {intent.get('label', '--')} {pct(intent.get('score', 0))}"
                f"{render_topk(intent.get('top_k', []))}</span><br>"
                f"<span class='pill pill-sent'>Sent: {sent.get('label', '--')} {pct(sent.get('score', 0))}"
                f"{render_topk(sent.get('top_k', []))}</span><br>"
                f"<span class='pill pill-emo'>Emocion: {emo.get('label', '--')} {pct(emo.get('score', 0))}"
                f"{render_topk(emo.get('top_k', []))}</span>"
            )
            if ents:
                aux_html += "<br><span class='tiny'>Entidades:</span> " + ", ".join(
                    f"{e.get('type')}:{e.get('value')}" for e in ents
                )
            aux_html += "<br><span class='tiny'>Decoder: GPT | T=" + str(meta.get("processing_ms", 0)) + " ms</span>"
            if meta.get("emergency"):
                aux_html += "<br><span style='color:#f87171;'>EMERGENCIA detectada</span>"
            aux_html += "</div>"

        st.markdown(
            f"""
            <div class='chat-row bot'>
              <div class='chat-bubble-bot'>
                <div style='white-space:pre-wrap;'>{content}</div>
                {aux_html}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_ts1, col_ts2, col_ts3 = st.columns([0.2, 0.5, 0.3])
        with col_ts1:
            if ts:
                st.markdown(f"<div class='tiny'>{ts}</div>", unsafe_allow_html=True)
        with col_ts2:
            if HAS_TTS and st.button("🔊 Reproducir", key=f"tts-{idx}"):
                tts_play(content)
        with col_ts3:
            pass
        # Bloque visual para resumen de clasificación (si existe)
        resumen = meta.get("resumen") if meta else None
        if resumen:
            st.markdown(f"""
            <div style='background-color:#f4f4f4; padding:18px 22px; border-radius:12px; margin:10px 0 18px 48px; max-width:600px; box-shadow:0 2px 8px rgba(0,0,0,0.10);'>
                <b>Resumen de clasificación:</b><br>
                <ul style='margin-left:18px; margin-bottom:0;'>
                    <li><b>Término:</b> {resumen.get('termino','--')}</li>
                    <li><b>Etiqueta:</b> {resumen.get('etiqueta','--')}</li>
                    <li><b>Macro:</b> {resumen.get('macro','--')}</li>
                    <li><b>Domina:</b> <span style='color:#1976d2;'>{resumen.get('domina','--')}</span></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

prompt = st.chat_input("Escribe tu mensaje...")
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt, "ts": now_hhmm()})
    try:
        with st.spinner("Procesando..."):
            data = post_chat(prompt.strip())
    except requests.RequestException as exc:
        st.error(f"No se pudo conectar con el backend: {exc}")
    else:
        reply_text = data.get("reply", "")
        st.session_state["messages"].append(
            {"role": "assistant", "content": reply_text, "meta": data, "ts": now_hhmm()}
        )
        st.session_state["history"].append(data)
    st.rerun()

# Botón limpiar
if st.button("Limpiar chat"):
    st.session_state["messages"] = []
    st.session_state["history"] = []
    st.rerun()

# Sidebar: historial y monitor de sentimiento
with st.sidebar:
    st.markdown("### Historial")
    if not st.session_state["history"]:
        st.caption("Sin registros")
    else:
        for item in st.session_state["history"][-5:][::-1]:
            st.markdown(
                f"- {item.get('intent', {}).get('label','--')} | "
                f"{item.get('sentiment', {}).get('label','--')} | "
                f"{item.get('emotion', {}).get('label','--')}"
            )
    # Monitor de sentimiento predominante
    sentiments = [h.get("sentiment", {}).get("label") for h in st.session_state["history"] if h.get("sentiment")]
    if sentiments:
        counts = {}
        for s in sentiments:
            counts[s] = counts.get(s, 0) + 1
        dominant = max(counts, key=counts.get)
        st.markdown("### Sentimiento predominante")
        st.metric("Predomina", dominant, delta=None)
    else:
        st.markdown("### Sentimiento predominante")
        st.caption("Sin datos")
