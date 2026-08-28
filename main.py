"""
Telegram lyrics bot.

Flow:
    Telegram (audio or "Title - Artist")
      -> Render webhook
      -> Hugging Face Space  madarauchihagmailcom/My  (Lyr Online)
      -> Render
      -> Telegram (lyrics + .lrc)

The Space never talks to Telegram. Render holds the bot token and calls
the Gradio API server-side.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
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
HF_SPACE_ID = (
    os.environ.get("HF_SPACE_ID")
    or os.environ.get("HF_SPACE_URL")
    or "madarauchihagmailcom/My"
).strip()
HF_API_TOKEN = (os.environ.get("HF_API_TOKEN") or "").strip() or None
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL", "")

# Telegram Bot API download cap.
_MAX_DOWNLOAD_BYTES = 19 * 1024 * 1024
_AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".aac", ".opus", ".oga"}

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("Missing TELEGRAM_BOT_TOKEN environment variable.")

_space_client = None


def _get_space_client():
    """Connect once. Sleeping Spaces can take a minute to wake."""
    global _space_client
    if _space_client is not None:
        return _space_client
    from gradio_client import Client

    logger.info("Connecting to Hugging Face Space %s …", HF_SPACE_ID)
    _space_client = Client(HF_SPACE_ID, hf_token=HF_API_TOKEN, verbose=False)
    return _space_client


def _parse_title_artist(text: str) -> tuple[str, str]:
    text = (text or "").strip()
    if not text:
        return "", ""
    for sep in (" - ", " – ", " — ", " by ", " | "):
        if sep.lower() in text.lower():
            # keep original split on first matching sep (case-sensitive first)
            idx = text.lower().find(sep.lower())
            left, right = text[:idx], text[idx + len(sep) :]
            return left.strip(), right.strip()
    return text, ""


def _unpack_lyr_result(result: Any) -> tuple[str, str, str, Optional[str]]:
    """
    Lyr Online returns:
      (status markdown, timestamped LRC text, plain lyrics, json, lrc file)
    """
    if not isinstance(result, (list, tuple)):
        return str(result), "", "", None
    status = result[0] if len(result) > 0 else ""
    timed = result[1] if len(result) > 1 else ""
    plain = result[2] if len(result) > 2 else ""
    lrc = result[4] if len(result) > 4 else None
    if isinstance(lrc, dict):
        lrc = lrc.get("path") or lrc.get("url")
    if not isinstance(lrc, str) or not lrc:
        lrc = None
    return str(status or ""), str(timed or ""), str(plain or ""), lrc


def transcribe_song(
    audio_path: str,
    title: str = "",
    artist: str = "",
    language_label: str = "Auto detect",
) -> tuple[str, str, str, Optional[str]]:
    from gradio_client import handle_file

    client = _get_space_client()
    result = client.predict(
        audio_path=handle_file(audio_path),
        title=title or "",
        artist=artist or "",
        language_label=language_label or "Auto detect",
        api_name="/transcribe_song",
    )
    return _unpack_lyr_result(result)


def lookup_lyrics(title: str, artist: str = "", duration_seconds: float = 0) -> tuple[str, str, str, Optional[str]]:
    client = _get_space_client()
    result = client.predict(
        title=title or "",
        artist=artist or "",
        duration_seconds=float(duration_seconds or 0),
        api_name="/lookup_lyrics",
    )
    return _unpack_lyr_result(result)


# ---------------------------------------------------------------------------
# Telegram helpers
# ---------------------------------------------------------------------------


async def _reply_long(message, text: str) -> None:
    text = (text or "").strip()
    if not text:
        return
    chunk = 3900
    for start in range(0, len(text), chunk):
        await message.reply_text(text[start : start + chunk])


async def _send_lyrics(
    message,
    status: str,
    timed: str,
    plain: str,
    lrc_path: Optional[str],
) -> None:
    body = timed.strip() or plain.strip() or status.strip()
    if not body:
        await message.reply_text("No lyrics came back from the Space.")
        return
    await _reply_long(message, body)
    if lrc_path and Path(lrc_path).exists():
        with Path(lrc_path).open("rb") as handle:
            await message.reply_document(document=handle, filename="lyrics.lrc")


def _audio_meta(update: Update) -> tuple[Optional[str], str, int, str, str]:
    """file_id, filename, size, title, artist from voice / audio / document."""
    msg = update.message
    if msg.voice:
        return (
            msg.voice.file_id,
            "voice.ogg",
            msg.voice.file_size or 0,
            "",
            "",
        )
    if msg.audio:
        name = msg.audio.file_name or "song.mp3"
        return (
            msg.audio.file_id,
            name,
            msg.audio.file_size or 0,
            msg.audio.title or "",
            msg.audio.performer or "",
        )
    if msg.document:
        name = msg.document.file_name or "song.bin"
        return (
            msg.document.file_id,
            name,
            msg.document.file_size or 0,
            "",
            "",
        )
    return None, "", 0, "", ""


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Lyr Online via Render.\n\n"
        "• Send a song (MP3 / M4A / WAV / voice note)\n"
        "• Optional caption:  Title - Artist\n"
        "• Or text only:  Title - Artist   (name lookup, no audio)\n\n"
        "I'll send synced lyrics back. First run can take a minute if the "
        "Hugging Face Space is waking up."
    )


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    file_id, filename, size, title, artist = _audio_meta(update)
    if not file_id:
        await update.message.reply_text("I couldn't read that audio file.")
        return

    caption_title, caption_artist = _parse_title_artist(update.message.caption or "")
    title = caption_title or title
    artist = caption_artist or artist

    if size and size > _MAX_DOWNLOAD_BYTES:
        await update.message.reply_text(
            "That file is larger than Telegram lets bots download (~20 MB). "
            "Send a shorter clip or a smaller encode."
        )
        return

    suffix = Path(filename).suffix.lower() or ".ogg"
    if suffix not in _AUDIO_EXTS:
        suffix = ".ogg"

    await update.message.reply_text(
        "Got the song. Sending it to the Hugging Face Space… "
        "this can take a few minutes."
    )
    await update.message.chat.send_action(action="typing")

    tmp_path = None
    try:
        tg_file = await context.bot.get_file(file_id)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
        await tg_file.download_to_drive(custom_path=tmp_path)

        status, timed, plain, lrc = await asyncio.to_thread(
            transcribe_song, tmp_path, title, artist, "Auto detect"
        )
        await _send_lyrics(update.message, status, timed, plain, lrc)
    except Exception as exc:  # noqa: BLE001
        logger.exception("transcribe_song failed")
        await update.message.reply_text(
            "The Hugging Face Space failed or is still waking up. "
            f"Try again in a minute. ({exc.__class__.__name__})"
        )
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    title, artist = _parse_title_artist(update.message.text or "")
    if not title:
        await update.message.reply_text(
            "Send a song file, or text like:  Song Title - Artist"
        )
        return

    await update.message.reply_text(
        f"Looking up “{title}”"
        + (f" by {artist}" if artist else "")
        + " on the Space…"
    )
    await update.message.chat.send_action(action="typing")
    try:
        status, timed, plain, lrc = await asyncio.to_thread(
            lookup_lyrics, title, artist, 0
        )
        await _send_lyrics(update.message, status, timed, plain, lrc)
    except Exception as exc:  # noqa: BLE001
        logger.exception("lookup_lyrics failed")
        await update.message.reply_text(
            "Lookup failed. Send the audio file instead, or try again. "
            f"({exc.__class__.__name__})"
        )


# ---------------------------------------------------------------------------
# python-telegram-bot + FastAPI webhook
# ---------------------------------------------------------------------------

# FileExtension takes one string in PTB 21, not a list.
_AUDIO_FILTER = (
    filters.VOICE
    | filters.AUDIO
    | filters.Document.AUDIO
    | filters.Document.FileExtension("mp3")
    | filters.Document.FileExtension("m4a")
    | filters.Document.FileExtension("wav")
    | filters.Document.FileExtension("flac")
    | filters.Document.FileExtension("ogg")
    | filters.Document.FileExtension("aac")
    | filters.Document.FileExtension("opus")
    | filters.Document.FileExtension("oga")
)

telegram_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
telegram_app.add_handler(CommandHandler("start", start_command))
telegram_app.add_handler(MessageHandler(_AUDIO_FILTER, handle_audio))
telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

app = FastAPI()


@app.on_event("startup")
async def on_startup() -> None:
    await telegram_app.initialize()
    await telegram_app.start()
    logger.info("Telegram application started. space=%s", HF_SPACE_ID)
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
        "space": HF_SPACE_ID,
        "endpoints": ["/transcribe_song", "/lookup_lyrics"],
    }


@app.post("/webhook/{token}")
async def telegram_webhook(token: str, request: Request):
    """200 immediately — Space transcription can take longer than Telegram's wait."""
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
