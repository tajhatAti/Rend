"""
Telegram image bot.

Flow:
    Telegram (text or photo)
      -> Render webhook
      -> Hugging Face Space (FLUX / Florence-2 / RMBG / InstructPix2Pix)
      -> Render
      -> Telegram (photo or text)

Spaces never talk to Telegram. Render holds the bot token.
"""

from __future__ import annotations

import asyncio
import json
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
HF_API_TOKEN = (os.environ.get("HF_API_TOKEN") or "").strip() or None
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL", "")

SPACE_FLUX = (os.environ.get("SPACE_FLUX") or "black-forest-labs/FLUX.1-schnell").strip()
SPACE_VISION = (os.environ.get("SPACE_VISION") or "gokaygokay/Florence-2").strip()
SPACE_BG = (os.environ.get("SPACE_BG") or "not-lain/background-removal").strip()
SPACE_STYLE = (os.environ.get("SPACE_STYLE") or "timbrooks/instruct-pix2pix").strip()

_MAX_DOWNLOAD_BYTES = 19 * 1024 * 1024
_PHOTO_MODES = {"caption", "ocr", "bg", "detect", "style"}

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("Missing TELEGRAM_BOT_TOKEN environment variable.")

_clients: dict[str, Any] = {}


def _client(space_id: str):
    if space_id not in _clients:
        from gradio_client import Client

        logger.info("Connecting to Hugging Face Space %s …", space_id)
        _clients[space_id] = Client(space_id, hf_token=HF_API_TOKEN, verbose=False)
    return _clients[space_id]


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
        return "\n".join(part for part in (_as_text(v) for v in value) if part)
    if isinstance(value, dict):
        if "text" in value and isinstance(value["text"], str):
            return value["text"]
        try:
            return json.dumps(value, ensure_ascii=False, indent=2)
        except TypeError:
            return str(value)
    return str(value).strip()


def flux_generate(prompt: str) -> str:
    result = _client(SPACE_FLUX).predict(
        prompt=prompt,
        seed=0,
        randomize_seed=True,
        width=768,
        height=768,
        num_inference_steps=4,
        api_name="/infer",
    )
    path = _as_path(result)
    if not path:
        raise RuntimeError(f"FLUX returned no image: {result!r}")
    return path


def florence(image_path: str, task: str) -> tuple[str, Optional[str]]:
    from gradio_client import handle_file

    result = _client(SPACE_VISION).predict(
        handle_file(image_path),
        task,
        "",
        "microsoft/Florence-2-base",
        api_name="/process_image",
    )
    text = _as_text(result[0] if isinstance(result, (list, tuple)) and result else result)
    image = None
    if isinstance(result, (list, tuple)) and len(result) > 1:
        image = _as_path(result[1])
    return text, image


def remove_background(image_path: str) -> str:
    from gradio_client import handle_file

    result = _client(SPACE_BG).predict(
        f=handle_file(image_path),
        api_name="/png",
    )
    path = _as_path(result)
    if not path:
        raise RuntimeError(f"Background removal returned no file: {result!r}")
    return path


def style_edit(image_path: str, instruction: str) -> str:
    from gradio_client import handle_file

    result = _client(SPACE_STYLE).predict(
        input_image=handle_file(image_path),
        instruction=instruction,
        steps=20,
        randomize_seed="Randomize Seed",
        seed=1371,
        randomize_cfg="Fix CFG",
        text_cfg_scale=7.5,
        image_cfg_scale=1.5,
        api_name="/generate",
    )
    path = _as_path(result)
    if not path:
        raise RuntimeError(f"InstructPix2Pix returned no image: {result!r}")
    return path


# ---------------------------------------------------------------------------
# Telegram helpers
# ---------------------------------------------------------------------------

HELP = (
    "Image bot via Hugging Face Spaces.\n\n"
    "Text → FLUX image\n"
    "  a cat on a rooftop at sunset\n\n"
    "Photo + caption:\n"
    "  caption / describe   — describe the photo\n"
    "  ocr                  — read text in the photo\n"
    "  detect               — objects + boxes\n"
    "  bg                   — remove background (PNG)\n"
    "  make it anime        — style / edit the photo\n\n"
    "Commands: /flux /caption /ocr /detect /bg /style\n"
    "First call can take a minute while a Space wakes up."
)


async def _reply_long(message, text: str) -> None:
    text = (text or "").strip()
    if not text:
        return
    for start in range(0, len(text), 3900):
        await message.reply_text(text[start : start + 3900])


async def _send_image(message, path: str, caption: str = "", as_document: bool = False) -> None:
    p = Path(path)
    if p.exists():
        with p.open("rb") as handle:
            if as_document or p.suffix.lower() == ".png" and as_document:
                await message.reply_document(document=handle, filename=p.name, caption=caption or None)
            else:
                await message.reply_photo(photo=handle, caption=caption or None)
        return
    if path.startswith("http"):
        await message.reply_photo(photo=path, caption=caption or None)
        return
    await message.reply_text(caption or "Got a result but no image file.")


async def _download_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    msg = update.message
    file_id = None
    suffix = ".jpg"
    size = 0
    if msg.photo:
        photo = msg.photo[-1]
        file_id = photo.file_id
        size = photo.file_size or 0
        suffix = ".jpg"
    elif msg.document:
        file_id = msg.document.file_id
        size = msg.document.file_size or 0
        name = msg.document.file_name or "image.jpg"
        suffix = Path(name).suffix.lower() or ".jpg"
    if not file_id:
        raise RuntimeError("No photo on this message.")
    if size and size > _MAX_DOWNLOAD_BYTES:
        raise RuntimeError("Image is larger than Telegram lets bots download (~20 MB).")
    tg_file = await context.bot.get_file(file_id)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.close()
    await tg_file.download_to_drive(custom_path=tmp.name)
    return tmp.name


def _photo_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> tuple[str, str]:
    caption = (update.message.caption or "").strip()
    saved = (context.user_data or {}).get("mode") or "caption"
    lower = caption.lower()

    if lower in {"ocr", "/ocr"} or lower.startswith("ocr "):
        return "ocr", caption
    if lower in {"bg", "background", "remove", "/bg"}:
        return "bg", caption
    if lower in {"detect", "objects", "/detect"} or lower.startswith("detect"):
        return "detect", caption
    if lower in {"caption", "describe", "/caption"}:
        return "caption", caption
    if lower in {"style", "/style"}:
        return "style", "make it cinematic"
    if caption:
        return "style", caption
    return saved, caption


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP)


async def mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cmd = update.message.text.split()[0].lstrip("/").split("@")[0]
    rest = update.message.text.split(None, 1)
    extra = rest[1].strip() if len(rest) > 1 else ""
    if cmd == "flux":
        if extra:
            await _run_flux(update, extra)
            return
        context.user_data["mode"] = "flux"
        await update.message.reply_text("FLUX mode. Send a text prompt.")
        return
    if cmd in _PHOTO_MODES:
        context.user_data["mode"] = cmd
        hint = {
            "caption": "Send a photo to describe.",
            "ocr": "Send a photo with text to read.",
            "detect": "Send a photo to detect objects.",
            "bg": "Send a photo to remove the background.",
            "style": "Send a photo with an edit instruction as caption, e.g. make it anime.",
        }[cmd]
        await update.message.reply_text(f"Mode: {cmd}. {hint}")
        return
    await update.message.reply_text(HELP)


async def _run_flux(update: Update, prompt: str) -> None:
    await update.message.reply_text("Generating with FLUX… Space queue can take a minute.")
    await update.message.chat.send_action(action="upload_photo")
    try:
        path = await asyncio.to_thread(flux_generate, prompt)
        await _send_image(update.message, path, caption=prompt[:900])
    except Exception as exc:  # noqa: BLE001
        logger.exception("FLUX failed")
        await update.message.reply_text(
            f"FLUX Space failed or is busy. Try again in a minute. ({exc.__class__.__name__})"
        )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    prompt = (update.message.text or "").strip()
    if not prompt:
        return
    mode = (context.user_data or {}).get("mode") or "flux"
    if mode != "flux" and mode in _PHOTO_MODES:
        await update.message.reply_text(f"Mode is {mode}. Send a photo, or /flux for text-to-image.")
        return
    await _run_flux(update, prompt)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    mode, extra = _photo_mode(update, context)
    await update.message.reply_text(
        f"Running {mode} on the Hugging Face Space… this can take a minute."
    )
    await update.message.chat.send_action(action="upload_photo")
    tmp_path = None
    try:
        tmp_path = await _download_photo(update, context)
        if mode == "bg":
            out = await asyncio.to_thread(remove_background, tmp_path)
            await _send_image(update.message, out, caption="background removed", as_document=True)
            return
        if mode == "style":
            instruction = extra or "make it cinematic"
            out = await asyncio.to_thread(style_edit, tmp_path, instruction)
            await _send_image(update.message, out, caption=instruction[:900])
            return
        task = {
            "ocr": "OCR",
            "detect": "Object Detection",
            "caption": "Detailed Caption",
        }.get(mode, "Detailed Caption")
        text, image = await asyncio.to_thread(florence, tmp_path, task)
        if image:
            await _send_image(update.message, image, caption=(text or task)[:900])
        if text:
            await _reply_long(update.message, text)
        if not text and not image:
            await update.message.reply_text("The Space returned an empty result.")
    except Exception as exc:  # noqa: BLE001
        logger.exception("photo task %s failed", mode)
        await update.message.reply_text(
            f"Space failed or is waking up. Try again. ({exc.__class__.__name__}: {exc})"
        )
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# FastAPI + webhook
# ---------------------------------------------------------------------------

telegram_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
telegram_app.add_handler(CommandHandler("start", start_command))
telegram_app.add_handler(CommandHandler("help", help_command))
for _cmd in ("flux", "caption", "ocr", "detect", "bg", "style"):
    telegram_app.add_handler(CommandHandler(_cmd, mode_command))
telegram_app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_photo))
telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

app = FastAPI()


@app.on_event("startup")
async def on_startup() -> None:
    await telegram_app.initialize()
    await telegram_app.start()
    logger.info(
        "Image bot started. flux=%s vision=%s bg=%s style=%s",
        SPACE_FLUX,
        SPACE_VISION,
        SPACE_BG,
        SPACE_STYLE,
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
        "kind": "image",
        "spaces": {
            "flux": SPACE_FLUX,
            "vision": SPACE_VISION,
            "bg": SPACE_BG,
            "style": SPACE_STYLE,
        },
    }


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
