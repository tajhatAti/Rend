"""
Run this ONCE after your Render service is deployed and live, to tell
Telegram where to send updates.

Usage:
    python set_webhook.py https://your-app-name.onrender.com

Reads TELEGRAM_BOT_TOKEN from the environment (or a local .env file).
"""

import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

if not TOKEN:
    raise ValueError("Set TELEGRAM_BOT_TOKEN in your environment or .env file first.")

if len(sys.argv) != 2:
    print("Usage: python set_webhook.py https://your-app-name.onrender.com")
    sys.exit(1)

base_url = sys.argv[1].rstrip("/")
webhook_url = f"{base_url}/webhook/{TOKEN}"

resp = requests.post(
    f"https://api.telegram.org/bot{TOKEN}/setWebhook",
    json={"url": webhook_url},
)

print("Status code:", resp.status_code)
print("Response:", resp.json())
