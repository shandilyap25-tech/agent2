from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

LEAD_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "captured_leads.jsonl"


def mock_lead_capture(name: str, email: str, platform: str) -> str:
    message = f"Lead captured successfully: {name}, {email}, {platform}"
    print(message)

    LEAD_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "email": email,
        "platform": platform,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    with LEAD_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")

    return message

