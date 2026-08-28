"""
Telegram bot powered by Hugging Face Inference Providers.
Hosted on Render as a webhook-based FastAPI service.

Flow:
    Telegram user -> Telegram servers -> Render (this app) -> Hugging Face router
    -> Render (this app) -> Telegram servers -> user

Why webhooks instead of polling?
Render's free web service can sleep when idle and only stays "awake" while
handling HTTP requests. Long-polling (which keeps an open connection to
Telegram) fights with that. Webhooks work naturally: Telegram just POSTs to
our URL whenever there's a new message, which wakes the service if needed.
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
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
# Optional Groq key. Groq has a real free tier; Hugging Face only gives a
# tiny monthly credit ($0.10) that runs out fast on a chat bot.
GROQ_API_KEY = (os.environ.get("GROQ_API_KEY") or "").strip()
# The public URL Render gives your service, e.g. https://your-app.onrender.com
# Used only to log/confirm the webhook target; not required for the app to run.
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL", "")

# Qwen2.5-7B-Instruct is NOT on Groq, and on this account it is "not supported
# by any provider you have enabled" (HF 400). Groq *is* enabled, and Groq's
# Hugging Face catalog is gpt-oss-20b / gpt-oss-120b.
_DEFAULT_MODEL = "openai/gpt-oss-20b"
_DEFAULT_PROVIDER = "groq"
# llama-3.1-8b-instant was shut down on Groq on 2026-08-16.
_DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"

HF_MODEL = (os.environ.get("HF_MODEL") or _DEFAULT_MODEL).strip()
HF_PROVIDER = (os.environ.get("HF_PROVIDER") or _DEFAULT_PROVIDER).strip()
GROQ_MODEL = (os.environ.get("GROQ_MODEL") or _DEFAULT_GROQ_MODEL).strip()

_HF_INFERENCE_ALIASES = {"hf-inference", "hf_inference", "huggingface"}
# Models we already know will 400 on this setup — skip them instead of wasting
# a webhook round-trip.
_UNAVAILABLE_MODELS = {
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct-1M",
}
_CLIENT_TIMEOUT = 30


def _routes() -> list[tuple[str, str]]:
    """(provider, model) pairs to try, in order."""
    routes: list[tuple[str, str]] = []

    requested_model = HF_MODEL
    requested_provider = HF_PROVIDER

    if requested_model in _UNAVAILABLE_MODELS:
        logger.warning(
            "HF_MODEL=%s is not served by any provider enabled on this Hugging "
            "Face account (and Groq does not host it). Using %s instead.",
            requested_model,
            _DEFAULT_MODEL,
        )
        requested_model = _DEFAULT_MODEL
        requested_provider = _DEFAULT_PROVIDER

    if requested_provider.lower() in _HF_INFERENCE_ALIASES:
        logger.warning(
            "HF_PROVIDER=%s does not host chat LLMs. Using %s instead.",
            requested_provider,
            _DEFAULT_PROVIDER,
        )
        requested_provider = _DEFAULT_PROVIDER

    routes.append((requested_provider, requested_model))

    for provider, model in (
        ("groq", "openai/gpt-oss-20b"),
        ("auto", "openai/gpt-oss-20b"),
        ("groq", "openai/gpt-oss-120b"),
    ):
        pair = (provider, model)
        if pair not in routes:
            routes.append(pair)
    return routes


if not TELEGRAM_BOT_TOKEN:
    raise ValueError("Missing TELEGRAM_BOT_TOKEN environment variable.")
if not HF_API_TOKEN:
    raise ValueError("Missing HF_API_TOKEN environment variable.")

ROUTES = _routes()

_provider_clients: dict[str, InferenceClient] = {}
for provider, _model in ROUTES:
    if provider not in _provider_clients:
        _provider_clients[provider] = InferenceClient(
            provider=provider,
            api_key=HF_API_TOKEN,
            timeout=_CLIENT_TIMEOUT,
        )

hf_attempts: list[tuple[str, str, InferenceClient]] = [
    (provider, model, _provider_clients[provider]) for provider, model in ROUTES
]

groq_client: Optional[InferenceClient] = None
if GROQ_API_KEY:
    groq_client = InferenceClient(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY,
        timeout=_CLIENT_TIMEOUT,
    )

# ---------------------------------------------------------------------------
# Hugging Face / Groq helpers
# ---------------------------------------------------------------------------


def _http_status(exc: BaseException) -> Optional[int]:
    """Best-effort HTTP status from huggingface_hub / httpx exceptions."""
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
    if "bad request" in text.lower() or "model_not_supported" in text.lower():
        return 400
    return None


def _is_token_permission_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(
        needle in text
        for needle in (
            "sufficient permissions",
            "does not have sufficient",
            "make calls to inference providers",
            "invalid username or password",
            "invalid token",
        )
    )


def _is_model_unavailable(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(
        needle in text
        for needle in (
            "model_not_supported",
            "not supported by any provider",
            "not supported by provider",
            "is not supported",
        )
    )


def _user_error_message(status: Optional[int]) -> str:
    if status == 401:
        return (
            "Hugging Face rejected the API token (401 Unauthorized). "
            "Check that HF_API_TOKEN on Render is a valid token from "
            "https://huggingface.co/settings/tokens"
        )
    if status == 402:
        return (
            "Hugging Face credits are exhausted (402 Payment Required). "
            "Free accounts only get a small monthly credit. "
            "Add credits at https://huggingface.co/settings/billing "
            "or set GROQ_API_KEY on Render to use Groq's free tier instead."
        )
    if status == 403:
        return (
            "Hugging Face returned 403 Forbidden. Fine-grained tokens need "
            "the permission \"Make calls to Inference Providers\" — create a "
            "new token at https://huggingface.co/settings/tokens with that "
            "box checked (or use a classic Read token), then update "
            "HF_API_TOKEN on Render."
        )
    if status in (400, 404):
        return (
            "None of the configured models are available on the Inference "
            "Providers enabled for this Hugging Face account. Enable more "
            "providers at https://huggingface.co/settings/inference-providers "
            "or set GROQ_API_KEY on Render."
        )
    if status == 429:
        return "The AI provider is rate-limiting us right now. Please try again in a moment."
    return (
        "Sorry, I couldn't reach the AI model just now "
        "(it may still be loading or there was a network issue). "
        "Please try again in a moment."
    )


def _complete(client: InferenceClient, model: str, user_text: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_text}],
        max_tokens=300,
    )
    content = response.choices[0].message.content
    return content or ""


def generate_reply(user_text: str) -> str:
    """Try Hugging Face (provider, model) routes, then optional Groq."""
    last_status: Optional[int] = None

    for provider, model, client in hf_attempts:
        try:
            reply = _complete(client, model, user_text)
            logger.info("HF reply ok (provider=%s, model=%s)", provider, model)
            return reply
        except Exception as exc:  # noqa: BLE001
            last_status = _http_status(exc) or last_status
            logger.error(
                "Hugging Face API call failed (provider=%s, model=%s, status=%s): %s",
                provider,
                model,
                last_status,
                exc,
                exc_info=True,
            )
            if last_status == 401 or _is_token_permission_error(exc):
                break
            # Wrong model / provider combo — try the next route.
            if last_status in (400, 404) or _is_model_unavailable(exc):
                continue
            if last_status == 403:
                break

    if groq_client is not None:
        try:
            reply = _complete(groq_client, GROQ_MODEL, user_text)
            logger.info("Groq reply ok (model=%s)", GROQ_MODEL)
            return reply
        except Exception as exc:  # noqa: BLE001
            last_status = _http_status(exc) or last_status
            logger.error("Groq API call failed (model=%s): %s", GROQ_MODEL, exc, exc_info=True)

    return _user_error_message(last_status)


# ---------------------------------------------------------------------------
# Telegram handlers
# ---------------------------------------------------------------------------


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /start command."""
    await update.message.reply_text(
        "Hi! I'm a test bot powered by open-source models via Hugging Face. "
        "Send me any message and I'll reply using the AI model."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles any regular text message from the user."""
    user_text = update.message.text
    logger.info("Received message: %s", user_text)
    reply_text = generate_reply(user_text)
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
    logger.info(
        "Telegram application started. routes=%s groq=%s",
        ROUTES,
        bool(groq_client),
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
    """Simple health check endpoint so you can confirm the service is up."""
    return {
        "status": "ok",
        "routes": [{"provider": p, "model": m} for p, m in ROUTES],
        "groq_fallback": bool(groq_client),
    }


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
