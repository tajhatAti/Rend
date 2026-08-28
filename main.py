"""
Telegram bot: Render handles webhooks, Groq runs the model (free tier).

Flow:
    Telegram user -> Telegram -> Render (this app) -> Groq API
    -> Render -> Telegram -> user

Why this, not Hugging Face or self-hosting?
- Hugging Face free credits are ~$0.10/month and already exhausted (402).
- Render free has no GPU and ~512MB RAM — it cannot run a 7B model.
- Hugging Face Inference Endpoints / GPU Spaces cost real money.
- Groq gives a real free API (no GPU needed on our side).

Why webhooks instead of polling?
Render's free web service sleeps when idle. Webhooks wake it only when
Telegram POSTs a new message.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import FastAPI, Request
from huggingface_hub import InferenceClient
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
GROQ_API_KEY = (os.environ.get("GROQ_API_KEY") or "").strip()
HF_API_TOKEN = (os.environ.get("HF_API_TOKEN") or "").strip()
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL", "")

# Groq shut down llama-3.1-8b-instant on 2026-08-16; gpt-oss-20b is current.
GROQ_MODEL = (os.environ.get("GROQ_MODEL") or "openai/gpt-oss-20b").strip()
HF_MODEL = (os.environ.get("HF_MODEL") or "openai/gpt-oss-20b").strip()
_CLIENT_TIMEOUT = 30

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("Missing TELEGRAM_BOT_TOKEN environment variable.")
if not GROQ_API_KEY and not HF_API_TOKEN:
    raise ValueError(
        "Set GROQ_API_KEY (recommended, free at https://console.groq.com/keys) "
        "or HF_API_TOKEN."
    )

# Groq is the primary brain. Hugging Face is only a leftover fallback —
# free HF credits are tiny and this project already hit 402 Payment Required.
groq_client: Optional[InferenceClient] = None
if GROQ_API_KEY:
    groq_client = InferenceClient(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY,
        timeout=_CLIENT_TIMEOUT,
    )

hf_client: Optional[InferenceClient] = None
if HF_API_TOKEN:
    hf_client = InferenceClient(
        provider="groq",
        api_key=HF_API_TOKEN,
        timeout=_CLIENT_TIMEOUT,
    )

# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _http_status(exc: BaseException) -> Optional[int]:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    if response is not None:
        code = getattr(response, "status_code", None)
        if isinstance(code, int):
            return code
    text = str(exc)
    for code in (401, 402, 403, 404, 429, 503):
        if str(code) in text:
            return code
    return None


def _complete(client: InferenceClient, model: str, user_text: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_text}],
        max_tokens=300,
    )
    content = response.choices[0].message.content
    return content or ""


def _user_error_message(status: Optional[int], *, used_groq: bool) -> str:
    if not GROQ_API_KEY:
        return (
            "The bot has no Groq API key yet, and Hugging Face free credits "
            "are used up. Add GROQ_API_KEY on Render:\n"
            "1) Open https://console.groq.com/keys (free Google login)\n"
            "2) Create an API key\n"
            "3) Render → your service → Environment → GROQ_API_KEY = that key\n"
            "4) Save and wait for the redeploy"
        )
    if status == 401:
        which = "Groq" if used_groq else "Hugging Face"
        return (
            f"{which} rejected the API key (401). Check GROQ_API_KEY on Render "
            "(https://console.groq.com/keys)."
        )
    if status == 429:
        return "The AI provider is rate-limiting us. Please try again in a moment."
    if status == 402:
        return (
            "Hugging Face credits are exhausted. This bot should be using Groq "
            "instead — make sure GROQ_API_KEY is set on Render."
        )
    return (
        "Sorry, I couldn't reach the AI model just now. Please try again in a moment."
    )


def generate_reply(user_text: str) -> str:
    last_status: Optional[int] = None
    used_groq = False

    if groq_client is not None:
        used_groq = True
        try:
            reply = _complete(groq_client, GROQ_MODEL, user_text)
            logger.info("Groq reply ok (model=%s)", GROQ_MODEL)
            return reply
        except Exception as exc:  # noqa: BLE001
            last_status = _http_status(exc) or last_status
            logger.error("Groq API call failed (model=%s): %s", GROQ_MODEL, exc, exc_info=True)

    if hf_client is not None:
        try:
            reply = _complete(hf_client, HF_MODEL, user_text)
            logger.info("HF reply ok (model=%s)", HF_MODEL)
            return reply
        except Exception as exc:  # noqa: BLE001
            last_status = _http_status(exc) or last_status
            logger.error("Hugging Face API call failed (model=%s): %s", HF_MODEL, exc, exc_info=True)

    return _user_error_message(last_status, used_groq=used_groq)


# ---------------------------------------------------------------------------
# Telegram handlers
# ---------------------------------------------------------------------------


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hi! Send me any message and I'll reply using an open-source AI model."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_text = update.message.text
    logger.info("Received message: %s", user_text)
    reply_text = generate_reply(user_text)
    await update.message.reply_text(reply_text)


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
        "Telegram application started. groq=%s hf=%s model=%s",
        bool(groq_client),
        bool(hf_client),
        GROQ_MODEL if groq_client else HF_MODEL,
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
        "backend": "groq" if groq_client else "huggingface",
        "model": GROQ_MODEL if groq_client else HF_MODEL,
        "groq_configured": bool(groq_client),
        "hf_configured": bool(hf_client),
    }


@app.post("/webhook/{token}")
async def telegram_webhook(token: str, request: Request):
    if token != TELEGRAM_BOT_TOKEN:
        return {"error": "unauthorized"}, 403

    data = await request.json()
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.process_update(update)
    return {"status": "ok"}
