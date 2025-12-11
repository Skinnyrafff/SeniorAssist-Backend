"""
Quick CLI to smoke-test the backend without frontend.
Usage:
  BASE_URL=http://localhost:8000 DEVICE_ID=dev1 python scripts/cli_test.py
"""

import os
import sys
from typing import Any, Dict

import requests

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
DEVICE_ID = os.getenv("DEVICE_ID", "dev1")


def post_chat(text: str) -> Dict[str, Any]:
    resp = requests.post(f"{BASE_URL}/chat", json={"text": text, "device_id": DEVICE_ID}, timeout=20)
    resp.raise_for_status()
    return resp.json()


def get_json(path: str) -> Any:
    resp = requests.get(f"{BASE_URL}{path}", timeout=20)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    print(f"Testing backend at {BASE_URL} with device_id={DEVICE_ID}")

    recordatorio_text = "recuerda pagar la luz manana 9am"
    emergencia_text = "me cai y sangro, es emergencia"

    try:
        print("\n=> POST /chat (recordatorio)")
        res1 = post_chat(recordatorio_text)
        print(f"flow={res1.get('flow')} next_prompt={res1.get('next_prompt')}")

        print("\n=> POST /chat (emergencia)")
        res2 = post_chat(emergencia_text)
        print(f"flow={res2.get('flow')} next_prompt={res2.get('next_prompt')}")

        print("\n=> GET /reminders")
        rems = get_json(f"/reminders?device_id={DEVICE_ID}")
        print(rems)

        print("\n=> GET /emergencies")
        ems = get_json(f"/emergencies?device_id={DEVICE_ID}")
        print(ems)

        print("\n=> GET /history")
        hist = get_json(f"/history?device_id={DEVICE_ID}&limit=5")
        print(hist)

    except requests.HTTPError as exc:
        print(f"HTTP error: {exc} => {getattr(exc.response, 'text', '')}")
        sys.exit(1)
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
