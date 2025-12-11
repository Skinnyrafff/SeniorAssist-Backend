"""
Heuristics to extract reminder items (title, due_at) from a text.
Uses spaCy NER (already loaded in predictors), regex, and dateparser.
"""

import re
from datetime import datetime
from typing import Dict, List, Optional

import dateparser

from ..adapters import predictors
from ..core.config import settings


def _segments_by_connectors(text: str) -> List[str]:
    segs = re.split(r"\b(?:y|ademas|adem\u00e1s|tambien|tamb\u00e9n|;|,)\b", text, flags=re.I)
    return [s.strip() for s in segs if s.strip()]


def _extract_times(text: str) -> List[str]:
    times = []
    times += re.findall(r"\b(\d{1,2}:\d{2})\b", text)
    times += [f"{h} {ampm}" for h, ampm in re.findall(r"\b(\d{1,2})\s*(am|pm)\b", text, flags=re.I)]
    return times


def _parse_due(text: str) -> Optional[datetime]:
    dt = dateparser.parse(
        text,
        settings={"RETURN_AS_TIMEZONE_AWARE": False, "PREFER_DATES_FROM": "future", "RELATIVE_BASE": datetime.utcnow()},
        languages=["es", "en"],
    )
    return dt


def extract_reminders(text: str) -> List[Dict[str, Optional[str]]]:
    items: List[Dict[str, Optional[str]]] = []

    # NER time/date entities
    entities = predictors.predict_ner(text)
    ner_times = [e for e in entities if e.get("type", "").lower() in {"time", "date"}]

    segments = _segments_by_connectors(text) if len(ner_times) > 1 else [text]

    for seg in segments:
        seg_times = _extract_times(seg)
        ner_in_seg = [e for e in ner_times if e.get("value") and e["value"] in seg]

        # Build time hint
        time_hint = None
        if seg_times:
            time_hint = seg_times[0]
        elif ner_in_seg:
            time_hint = ner_in_seg[0].get("value")

        due_at = _parse_due(seg if time_hint is None else time_hint)

        title = seg
        for token in seg_times:
            title = re.sub(re.escape(token), "", title, flags=re.I)
        for ent in ner_in_seg:
            title = title.replace(ent.get("value", ""), "")
        title = re.sub(r"\b(?:am|pm|a\.m\.|p\.m\.)\b", "", title, flags=re.I)
        title = re.sub(r"\s+", " ", title).strip(" ,.;")

        # Quitar verbos iniciales comunes
        title = re.sub(
            r"^(recuerda|recordar|recordatorio|necesito|quiero|favor|debo|pon|configura)\s+",
            "",
            title,
            flags=re.I,
        ).strip(" ,.;")

        items.append({"title": title or seg, "due_at": due_at, "timezone": settings.timezone})

    return items
