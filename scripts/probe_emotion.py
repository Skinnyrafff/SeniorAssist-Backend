"""
Quick probe for the emotion model to inspect label IDs and scores.
Usage: python scripts/probe_emotion.py
Uses EMOTION_MODEL_PATH from .env (default: ./models/emotion/pysentimiento_robertuito_sentiment_analysis_finetuned).
"""

import os
from typing import Dict, List

from dotenv import load_dotenv
from transformers import pipeline


def main() -> None:
    load_dotenv(".env")
    emotion_path = os.getenv(
        "EMOTION_MODEL_PATH", "./models/emotion/pysentimiento_robertuito_sentiment_analysis_finetuned"
    )
    clf = pipeline("text-classification", model=emotion_path, tokenizer=emotion_path, top_k=None)

    samples: Dict[str, List[str]] = {
        "alegria": [
            "me siento feliz",
            "estoy contento y agradecido",
            "hoy fue un gran dia",
        ],
        "asco": [
            "me da asco esta comida",
            "siento repulsion al verlo",
        ],
        "enojo": [
            "estoy muy enojado",
            "me molesta mucho lo que paso",
        ],
        "miedo": [
            "tengo miedo de caerme",
            "estoy asustado por la noche",
        ],
        "soledad": [
            "me siento solo",
            "no tengo con quien hablar",
        ],
        "sorpresa": [
            "estoy sorprendido por la noticia",
            "no me lo esperaba",
        ],
        "tristeza": [
            "estoy muy triste",
            "me siento deprimido",
        ],
    }

    for target, utterances in samples.items():
        print(f"\n### Target: {target}")
        for txt in utterances:
            res = clf(txt)
            items = res[0] if res and isinstance(res[0], list) else res
            top = max(items, key=lambda x: x["score"])
            top_two = sorted(items, key=lambda x: x["score"], reverse=True)[:2]
            print(f"- {txt}")
            for item in top_two:
                print(f"  {item['label']}: {item['score']:.4f}")
            print(f"  -> predicted: {top['label']}")


if __name__ == "__main__":
    main()
