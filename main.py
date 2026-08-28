"""
Telegram bot on Render that calls a Hugging Face Space (the actual model).

Why Render sits in the middle
-----------------------------
Telegram talks to *your* server. Your server talks to the Space.
Users never call Hugging Face from Telegram, so Space URLs / tokens stay
server-side. This is the normal pattern — it does not skip Hugging Face
rate limits or billing; the Space still enforces its own queue.

Flow:
    Telegram user
      -> Telegram servers
      -> Render (this webhook)
      -> Hugging Face Space (Gradio API: chat / image / audio / …)
      -> Render
      -> Telegram
      -> user

Optional: GROQ_API_KEY is only a text-chat fallback if no Space is set.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Request
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
# Space id like "username/my-space"  OR a full https://….hf.space URL
HF_SPACE_ID = (os.environ.get("HF_SPACE_ID") or os.environ.get("HF_SPACE_URL") or "").strip()
HF_SPACE_API = (os.environ.get("HF_SPACE_API") or "/predict").strip()
HF_API_TOKEN = (os.environ.get("HF_API_TOKEN") or "").strip() or None
GROQ_API_KEY = (os.environ.get("GROQ_API_KEY") or "").strip()
GROQ_MODEL = (os.environ.get("GROQ_MODEL") or "openai/gpt-oss-20b").strip()
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL", "")

_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
_AUDIO_EXT = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".opus"}

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("Missing TELEGRAM_BOT_TOKEN environment variable.")
if not HF_SPACE_ID and not GROQ_API_KEY:
    raise ValueError(
        "Set HF_SPACE_ID to your Hugging Face Space (username/space-name) "
        "or GROQ_API_KEY as a text-chat fallback."
    )

_space_client = None


def _get_space_client():
    """Connect once. Hugging Face Spaces sleep; first connect may take a minute."""
    global _space_client
    if _space_client is not None:
        return _space_client
    from gradio_client import Client

    logger.info("Connecting to Hugging Face Space %s …", HF_SPACE_ID)
    _space_client = Client(HF_SPACE_ID, hf_token=HF_API_TOKEN, verbose=False)
    try:
        logger.info("Space API:\n%s", _space_client.view_api(return_format="str"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not list Space API: %s", exc)
    return _space_client


def _call_space(user_text: str) -> Any:
    client = _get_space_client()
    return client.predict(user_text, api_name=HF_SPACE_API)


def _classify_payload(result: Any) -> tuple[str, str]:
    """Turn a Gradio return value into ('text'|'photo'|'audio', value)."""
    if isinstance(result, (list, tuple)):
        result = next((item for item in result if item not in (None, "")), result[0] if result else "")
    if isinstance(result, dict):
        result = (
            result.get("url")
            or result.get("path")
            or result.get("value")
            or result.get("text")
            or result.get("content")
            or str(result)
        )

    value = result if isinstance(result, str) else str(result)
    lower = value.lower().split("?", 1)[0]
    path = Path(value)

    if path.exists():
        ext = path.suffix.lower()
        if ext in _IMAGE_EXT:
            return "photo", value
        if ext in _AUDIO_EXT:
            return "audio", value
        return "text", value

    if any(lower.endswith(ext) for ext in _IMAGE_EXT):
        return "photo", value
    if any(lower.endswith(ext) for ext in _AUDIO_EXT):
        return "audio", value
    return "text", value


def _call_groq(user_text: str) -> str:
    from huggingface_hub import InferenceClient

    client = InferenceClient(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY,
        timeout=30,
    )
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": user_text}],
        max_tokens=300,
    )
    return response.choices[0].message.content or ""


def run_backend(user_text: str) -> tuple[str, str]:
    """Preferred: Hugging Face Space. Fallback: Groq text chat."""
    if HF_SPACE_ID:
        try:
            raw = _call_space(user_text)
            kind, payload = _classify_payload(raw)
            logger.info("Space reply ok (kind=%s)", kind)
            return kind, payload
        except Exception as exc:  # noqa: BLE001
            logger.error("Hugging Face Space call failed: %s", exc, exc_info=True)
            if not GROQ_API_KEY:
                return (
                    "text",
                    "Could not reach the Hugging Face Space. It may be sleeping "
                    "or HF_SPACE_API does not match the Space's endpoint. "
                    f"Check Render logs. ({exc.__class__.__name__})",
                )

    if GROQ_API_KEY:
        try:
            return "text", _call_groq(user_text)
        except Exception as exc:  # noqa: BLE001
            logger.error("Groq call failed: %s", exc, exc_info=True)
            return "text", "Sorry, I couldn't reach the AI model just now. Please try again."

    return "text", "No backend configured. Set HF_SPACE_ID on Render."


# ---------------------------------------------------------------------------
# Telegram handlers
# ---------------------------------------------------------------------------


async def _send_result(update: Update, kind: str, payload: str) -> None:
    message = update.message
    if kind == "photo":
        path = Path(payload)
        if path.exists():
            with path.open("rb") as handle:
                await message.reply_photo(photo=handle)
            return
        await message.reply_photo(photo=payload)
        return
    if kind == "audio":
        path = Path(payload)
        if path.exists():
            with path.open("rb") as handle:
                await message.reply_voice(voice=handle)
            return
        await message.reply_voice(voice=payload)
        return
    await message.reply_text(payload[:4000] or "(empty response)")


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    backend = f"Hugging Face Space `{HF_SPACE_ID}`" if HF_SPACE_ID else "Groq"
    await update.message.reply_text(
        f"Hi! Send me a message and I'll run it on {backend}."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_text = update.message.text
    logger.info("Received message: %s", user_text)
    await update.message.chat.send_action(action="typing")
    kind, payload = await asyncio.to_thread(run_backend, user_text)
    await _send_result(update, kind, payload)


# ---------------------------------------------------------------------------
# python-telegram-bot + FastAPI webhook
# ---------------------------------------------------------------------------

telegram_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
telegram_app.add_handler(CommandHandler("start", start_command))
telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

app = FastAPI()


@app.on_event("startup")
async def on_startup() -> None:
    await telegram_app.initialize()
    await telegram_app.start()
    logger.info(
        "Telegram application started. space=%s api=%s groq=%s",
        HF_SPACE_ID or "(none)",
        HF_SPACE_API,
        bool(GROQ_API_KEY),
    )
    if RENDER_EXTERNAL_URL:
        logger.info(
            "Remember to set your webhook to: %s/webhook/%s",
            RENDER_EXTERNAL_URL,
            TELEGRAM_BOT_TOKEN,
        )


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await telegram_app.stop()
    await telegram_app.shutdown()


@app.get("/")
async def health_check():
    return {
        "status": "ok",
        "space": HF_SPACE_ID or None,
        "space_api": HF_SPACE_API if HF_SPACE_ID else None,
        "groq_fallback": bool(GROQ_API_KEY),
    }


@app.post("/webhook/{token}")
async def telegram_webhook(token: str, request: Request):
    """
    Return 200 immediately so Telegram does not retry while the Space
    is waking up (cold start can take 30–60s).
    """
    if token != TELEGRAM_BOT_TOKEN:
        return {"error": "unauthorized"}, 403

    data = await request.json()
    update = Update.de_json(data, telegram_app.bot)

    async def _process() -> None:
        try:
            await telegram_app.process_update(update)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to process Telegram update")

    asyncio.create_task(_process())
    return {"status": "ok"}
