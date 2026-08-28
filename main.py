"""
Telegram bot powered by Hugging Face's Inference API (Qwen2.5-0.5B-Instruct).
Hosted on Render as a webhook-based FastAPI service.

Flow:
    Telegram user -> Telegram servers -> Render (this app) -> Hugging Face Inference API
    -> Render (this app) -> Telegram servers -> user

Why webhooks instead of polling?
Render's free web service can sleep when idle and only stays "awake" while
handling HTTP requests. Long-polling (which keeps an open connection to
Telegram) fights with that. Webhooks work naturally: Telegram just POSTs to
our URL whenever there's a new message, which wakes the service if needed.
"""

import os
import logging

from fastapi import FastAPI, Request
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from huggingface_hub import InferenceClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
# The public URL Render gives your service, e.g. https://your-app.onrender.com
# Used only to log/confirm the webhook target; not required for the app to run.
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL", "")

HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("Missing TELEGRAM_BOT_TOKEN environment variable.")
if not HF_API_TOKEN:
    raise ValueError("Missing HF_API_TOKEN environment variable.")

# Hugging Face client. As of the newer "Inference Providers" system, HF
# routes chat_completion calls through one of several backend providers
# (e.g. hf-inference, together, novita). Passing provider="auto" lets HF
# pick a provider that actually supports the requested model - omitting
# it, or using a model no provider supports, causes failures.
hf_client = InferenceClient(provider="auto", api_key=HF_API_TOKEN)

# ---------------------------------------------------------------------------
# Telegram handlers
# ---------------------------------------------------------------------------


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /start command."""
    await update.message.reply_text(
        "Hi! I'm a test bot powered by Qwen2.5 (via Hugging Face). "
        "Send me any message and I'll reply using the AI model."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles any regular text message from the user."""
    user_text = update.message.text
    logger.info("Received message: %s", user_text)

    try:
        # chat.completions.create expects a list of {"role": ..., "content": ...}
        # messages, same shape as OpenAI-style chat APIs. `model` must be passed
        # here since the client itself no longer pins a single model.
        response = hf_client.chat.completions.create(
            model=HF_MODEL,
            messages=[{"role": "user", "content": user_text}],
            max_tokens=300,
        )
        reply_text = response.choices[0].message.content

    except Exception as exc:  # noqa: BLE001 - we want to catch and report any failure
        # Common causes: model still "warming up" on HF's servers, network hiccup,
        # rate limiting, an invalid/expired HF token, or a model not supported
        # by any inference provider. The full exception is logged so you can
        # check Render's logs (Render dashboard -> your service -> Logs) for
        # the exact reason.
        logger.error("Hugging Face API call failed: %s", exc, exc_info=True)
        reply_text = (
            "Sorry, I couldn't reach the AI model just now "
            "(it may still be loading or there was a network issue). "
            "Please try again in a moment."
        )

    await update.message.reply_text(reply_text)


# ---------------------------------------------------------------------------
# Build the python-telegram-bot Application
# ---------------------------------------------------------------------------

telegram_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
telegram_app.add_handler(CommandHandler("start", start_command))
telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# ---------------------------------------------------------------------------
# FastAPI app - receives Telegram's webhook POSTs
# ---------------------------------------------------------------------------

app = FastAPI()


@app.on_event("startup")
async def on_startup() -> None:
    """Initialize the PTB Application when the FastAPI server starts."""
    await telegram_app.initialize()
    await telegram_app.start()
    logger.info("Telegram application started.")
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
    """Simple health check endpoint so you can confirm the service is up."""
    return {"status": "ok", "model": HF_MODEL}


@app.post("/webhook/{token}")
async def telegram_webhook(token: str, request: Request):
    """
    Telegram sends updates here via HTTP POST.
    The token in the URL path acts as a simple secret so random requests
    can't feed fake updates into your bot.
    """
    if token != TELEGRAM_BOT_TOKEN:
        return {"error": "unauthorized"}, 403

    data = await request.json()
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.process_update(update)
    return {"status": "ok"}

