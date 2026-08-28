"""
Telegram image bot → YOUR Hugging Face Space (not random public Spaces).

Flow:
    Telegram
      -> Render (this webhook)
      -> your Space  /caption /ocr /detect /bg
      -> Render
      -> Telegram

Deploy the files in ./space to a Gradio Space, then set HF_SPACE_ID.
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

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
HF_SPACE_ID = (
    os.environ.get("HF_SPACE_ID")
    or os.environ.get("HF_SPACE_URL")
    or "madarauchihagmailcom/image-bot"
).strip()
HF_API_TOKEN = (os.environ.get("HF_API_TOKEN") or "").strip() or None
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL", "")
_MAX_DOWNLOAD_BYTES = 19 * 1024 * 1024
_PHOTO_MODES = {"caption", "ocr", "bg", "detect"}

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("Missing TELEGRAM_BOT_TOKEN environment variable.")

_space = None


def _client():
    global _space
    if _space is None:
        from gradio_client import Client

        logger.info("Connecting to your Space %s …", HF_SPACE_ID)
        _space = Client(HF_SPACE_ID, hf_token=HF_API_TOKEN, verbose=False)
    return _space


def _as_path(value: Any) -> Optional[str]:
    if value is None or value is False:
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _as_path(item)
            if found:
                return found
        return None
    if isinstance(value, dict):
        return _as_path(value.get("path") or value.get("url"))
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _as_text(value: Any) -> str:
    if value is None or value is False:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(p for p in (_as_text(v) for v in value) if p)
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        return str(value)
    return str(value).strip()


def call_caption(image_path: str) -> str:
    from gradio_client import handle_file

    return _as_text(_client().predict(handle_file(image_path), api_name="/caption"))


def call_ocr(image_path: str) -> str:
    from gradio_client import handle_file

    return _as_text(_client().predict(handle_file(image_path), api_name="/ocr"))


def call_detect(image_path: str) -> tuple[str, Optional[str]]:
    from gradio_client import handle_file

    result = _client().predict(handle_file(image_path), api_name="/detect")
    if isinstance(result, (list, tuple)) and len(result) >= 2:
        return _as_text(result[1]), _as_path(result[0])
    return _as_text(result), _as_path(result)


def call_bg(image_path: str) -> str:
    from gradio_client import handle_file

    path = _as_path(_client().predict(handle_file(image_path), api_name="/bg"))
    if not path:
        raise RuntimeError("Space /bg returned no image")
    return path


HELP = (
    "Image bot → your Hugging Face Space.\n\n"
    "Send a photo:\n"
    "  (no caption)  describe\n"
    "  ocr           read text\n"
    "  detect        objects\n"
    "  bg            remove background\n\n"
    "Commands: /caption /ocr /detect /bg\n\n"
    f"Space: {HF_SPACE_ID}"
)


async def _reply_long(message, text: str) -> None:
    text = (text or "").strip() or "(empty)"
    for start in range(0, len(text), 3900):
        await message.reply_text(text[start : start + 3900])


async def _send_image(message, path: str, caption: str = "", as_document: bool = False) -> None:
    p = Path(path)
    if p.exists():
        with p.open("rb") as handle:
            if as_document:
                await message.reply_document(document=handle, filename=p.name, caption=caption or None)
            else:
                await message.reply_photo(photo=handle, caption=caption or None)
        return
    if path.startswith("http"):
        await message.reply_photo(photo=path, caption=caption or None)
        return
    await message.reply_text(caption or "No image file came back.")


async def _download_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    msg = update.message
    file_id = None
    suffix = ".jpg"
    size = 0
    if msg.photo:
        photo = msg.photo[-1]
        file_id, size, suffix = photo.file_id, photo.file_size or 0, ".jpg"
    elif msg.document:
        file_id = msg.document.file_id
        size = msg.document.file_size or 0
        suffix = Path(msg.document.file_name or "image.jpg").suffix.lower() or ".jpg"
    if not file_id:
        raise RuntimeError("No photo on this message.")
    if size and size > _MAX_DOWNLOAD_BYTES:
        raise RuntimeError("Image is larger than Telegram lets bots download (~20 MB).")
    tg_file = await context.bot.get_file(file_id)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.close()
    await tg_file.download_to_drive(custom_path=tmp.name)
    return tmp.name


def _photo_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    caption = (update.message.caption or "").strip().lower()
    saved = (context.user_data or {}).get("mode") or "caption"
    if caption in {"ocr", "/ocr"} or caption.startswith("ocr"):
        return "ocr"
    if caption in {"bg", "background", "remove", "/bg"}:
        return "bg"
    if caption in {"detect", "objects", "/detect"} or caption.startswith("detect"):
        return "detect"
    if caption in {"caption", "describe", "/caption"}:
        return "caption"
    return saved


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP)


async def mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cmd = update.message.text.split()[0].lstrip("/").split("@")[0]
    if cmd not in _PHOTO_MODES:
        await update.message.reply_text(HELP)
        return
    context.user_data["mode"] = cmd
    await update.message.reply_text(f"Mode: {cmd}. Send a photo.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Send a photo. Text-to-image (FLUX) needs a GPU on your Space — "
        "this Space runs caption / ocr / detect / bg on CPU.\n\n" + HELP
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    mode = _photo_mode(update, context)
    await update.message.reply_text(f"Sending to your Space ({mode})… first run can take a minute.")
    await update.message.chat.send_action(action="upload_photo")
    tmp_path = None
    try:
        tmp_path = await _download_photo(update, context)
        if mode == "bg":
            out = await asyncio.to_thread(call_bg, tmp_path)
            await _send_image(update.message, out, caption="background removed", as_document=True)
            return
        if mode == "detect":
            text, image = await asyncio.to_thread(call_detect, tmp_path)
            if image:
                await _send_image(update.message, image, caption=(text or "detect")[:900])
            if text:
                await _reply_long(update.message, text)
            return
        fn = call_ocr if mode == "ocr" else call_caption
        text = await asyncio.to_thread(fn, tmp_path)
        await _reply_long(update.message, text)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Space call failed (%s)", mode)
        await update.message.reply_text(
            "Could not reach your Space. Build the Space from the repo `space/` "
            "folder, wait until it is Running, then check HF_SPACE_ID on Render.\n"
            f"({exc.__class__.__name__}: {exc})"
        )
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


telegram_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
telegram_app.add_handler(CommandHandler("start", start_command))
telegram_app.add_handler(CommandHandler("help", start_command))
for _cmd in ("caption", "ocr", "detect", "bg"):
    telegram_app.add_handler(CommandHandler(_cmd, mode_command))
telegram_app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_photo))
telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

app = FastAPI()


@app.on_event("startup")
async def on_startup() -> None:
    await telegram_app.initialize()
    await telegram_app.start()
    logger.info("Image bot started. space=%s", HF_SPACE_ID)
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
    return {"status": "ok", "space": HF_SPACE_ID, "apis": ["/caption", "/ocr", "/detect", "/bg"]}


@app.post("/webhook/{token}")
async def telegram_webhook(token: str, request: Request):
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
