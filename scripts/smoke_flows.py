"""
Smoke test for API flows with mocked predictors (no real models required).
Runs a few calls against the FastAPI app using TestClient.
"""

from contextlib import nullcontext
from typing import Dict

from fastapi.testclient import TestClient

import backend.adapters.predictors as predictors
from backend.main import app


def _mock_predictors():
    def mock_load_models():
        return None

    def mk(label, score=0.9):
        return {"label": label, "score": score, "top_k": [{"label": label, "score": score}]}

    predictors.load_models = mock_load_models  # type: ignore
    predictors.predict_intent = lambda text: mk("recordatorio" if "recordar" in text else "emergencia_medica")  # type: ignore
    predictors.predict_sentiment = lambda text: mk("NEU")  # type: ignore
    predictors.predict_emotion = lambda text: mk("neutral")  # type: ignore
    predictors.predict_ner = lambda text: []  # type: ignore
    predictors.safety_gate = lambda text: {"allow": True, "emergency": "emergencia" in text, "reason": "mock"}  # type: ignore
    predictors.generate_reply = lambda ctx: ("respuesta mock", "mock")  # type: ignore


def run_smoke():
    _mock_predictors()
    with TestClient(app) as client:
        resp = client.post("/chat", json={"text": "necesito recordar pagar la luz manana 9am", "device_id": "dev-smoke"})
        assert resp.status_code == 200, resp.text
        data: Dict = resp.json()
        assert data["flow"] == "recordatorio"

        resp2 = client.post("/chat", json={"text": "me duele el pecho es una emergencia", "device_id": "dev-smoke"})
        assert resp2.status_code == 200, resp2.text
        data2: Dict = resp2.json()
        assert data2["flow"] == "emergencia"

        hist = client.get("/history", params={"device_id": "dev-smoke", "limit": 5})
        assert hist.status_code == 200, hist.text
        print("Smoke OK: chat and history")


if __name__ == "__main__":
    run_smoke()
