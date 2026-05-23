"""Telegram notification helper — best-effort, never breaks the caller.

Set two environment variables (e.g. in ~/.telegram_env):
    export TELEGRAM_BOT_TOKEN="123456:ABCdef..."
    export TELEGRAM_CHAT_ID="987654321"

If either variable is absent the send() call is a silent no-op, so training
runs fine on machines where notifications are not configured.
"""
from __future__ import annotations

import json
import os
import urllib.request


def send(message: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return
    try:
        data = json.dumps({"chat_id": chat_id, "text": message}).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:  # noqa: BLE001
        pass  # notifications are best-effort; never interrupt training
