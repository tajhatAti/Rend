"""
Telegram bot powered by Hugging Face Inference Providers (Qwen2.5-7B-Instruct).
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

HF_MODEL = (os.environ.get("HF_MODEL") or "Qwen/Qwen2.5-7B-Instruct").strip()
GROQ_MODEL = (os.environ.get("GROQ_MODEL") or "llama-3.1-8b-instant").strip()

# Chat LLMs are served by partner providers (Featherless, Groq, Together, …),
# NOT by Hugging Face's own "hf-inference" serverless API.
#
# As of July 2025, hf-inference is CPU-oriented (embeddings, classification,
# BERT/GPT-2). Calling it for Qwen2.5-7B-Instruct hits
#   https://router.huggingface.co/hf-inference/models/Qwen/Qwen2.5-7B-Instruct/v1/chat/completions
# and returns HTTP 403 Forbidden.
#
# Qwen2.5-7B-Instruct is hosted by Featherless AI (and "auto" should pick it).
# Never fall back to hf-inference for this model.
_HF_INFERENCE_ALIASES = {"hf-inference", "hf_inference", "huggingface"}
# Featherless AI is the provider that actually hosts Qwen2.5-7B-Instruct.
# "auto" is next in case HF remaps the model later.
_CHAT_PROVIDERS = ("featherless-ai", "auto", "groq")
_CLIENT_TIMEOUT = 30


def _provider_chain() -> list[str]:
    """Providers to try, in order. Explicit HF_PROVIDER (if set) goes first."""
    requested = (os.environ.get("HF_PROVIDER") or "").strip()
    chain: list[str] = []

    if requested:
        if requested.lower() in _HF_INFERENCE_ALIASES:
            logger.warning(
                "HF_PROVIDER=%s does not serve chat model %s (that combination "
                "returns HTTP 403). Using auto / partner providers instead.",
                requested,
                HF_MODEL,
            )
        else:
            chain.append(requested)

    for provider in _CHAT_PROVIDERS:
        if provider not in chain:
            chain.append(provider)
    return chain


if not TELEGRAM_BOT_TOKEN:
    raise ValueError("Missing TELEGRAM_BOT_TOKEN environment variable.")
if not HF_API_TOKEN:
    raise ValueError("Missing HF_API_TOKEN environment variable.")

HF_PROVIDERS = _provider_chain()

# One client per provider. Created once at import time so we don't rebuild
# HTTP sessions on every Telegram message.
hf_clients: list[tuple[str, InferenceClient]] = [
    (
        provider,
        InferenceClient(provider=provider, api_key=HF_API_TOKEN, timeout=_CLIENT_TIMEOUT),
    )
    for provider in HF_PROVIDERS
]

groq_client: Optional[InferenceClient] = None
if GROQ_API_KEY:
    # Direct Groq OpenAI-compatible API — billed against Groq's free tier,
    # not Hugging Face credits.
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
            "Hugging Face returned 403 Forbidden. Two common causes:\n"
            "1) Fine-grained tokens need the permission "
            "\"Make calls to Inference Providers\" — create a new token at "
            "https://huggingface.co/settings/tokens with that box checked "
            "(or use a classic Read token), then update HF_API_TOKEN on Render.\n"
            "2) Don't set HF_PROVIDER=hf-inference — that provider does not "
            f"host {HF_MODEL}."
        )
    if status == 404:
        return (
            f"No Inference Provider currently hosts {HF_MODEL} (404). "
            "Set HF_MODEL on Render to a model that is served, e.g. "
            "Qwen/Qwen3-4B-Instruct-2507, or add GROQ_API_KEY."
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
    """Try Hugging Face partner providers, then optional Groq."""
    last_status: Optional[int] = None

    for provider, client in hf_clients:
        try:
            reply = _complete(client, HF_MODEL, user_text)
            logger.info("HF reply ok (provider=%s, model=%s)", provider, HF_MODEL)
            return reply
        except Exception as exc:  # noqa: BLE001
            last_status = _http_status(exc) or last_status
            logger.error(
                "Hugging Face API call failed (provider=%s, model=%s, status=%s): %s",
                provider,
                HF_MODEL,
                last_status,
                exc,
                exc_info=True,
            )
            # Auth failures are account-wide. A 403 aimed at hf-inference
            # (wrong provider) is not — keep trying partner providers.
            if last_status == 401 or _is_token_permission_error(exc):
                break
            if last_status == 403 and "hf-inference" not in str(exc).lower():
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
        "Hi! I'm a test bot powered by Qwen2.5 (via Hugging Face). "
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
        "Telegram application started. model=%s providers=%s groq=%s",
        HF_MODEL,
        HF_PROVIDERS,
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
        "model": HF_MODEL,
        "providers": HF_PROVIDERS,
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
