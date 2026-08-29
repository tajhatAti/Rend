"""
Telegram → Render → your Hugging Face Space → Render → Telegram.

FLUX.1-schnell, photo tools, and lyrics all run through YOUR Space.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Request
from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    Update,
)
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

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
_MAX_DOWNLOAD_BYTES = 19 * 1024 * 1024

_PHOTO_MODES = {"caption", "ocr", "bg", "detect", "sketch"}
_PROMPT_MODES = {"imagine", "lyrics"}
_ALL_MODES = _PHOTO_MODES | _PROMPT_MODES

_BTN_MODE = {
    "✨ Imagine": "imagine",
    "🎵 Lyrics": "lyrics",
    "🖼 Describe": "caption",
    "🔤 OCR": "ocr",
    "📦 Detect": "detect",
    "🪄 No BG": "bg",
    "✏️ Sketch": "sketch",
}

MENU_KEYBOARD = ReplyKeyboardMarkup(
    [
        [KeyboardButton("✨ Imagine"), KeyboardButton("🎵 Lyrics")],
        [KeyboardButton("🖼 Describe"), KeyboardButton("🔤 OCR")],
        [KeyboardButton("📦 Detect"), KeyboardButton("🪄 No BG")],
        [KeyboardButton("✏️ Sketch"), KeyboardButton("📋 Menu")],
    ],
    resize_keyboard=True,
    is_persistent=True,
)

INLINE_MENU = InlineKeyboardMarkup(
    [
        [
            InlineKeyboardButton("✨ Imagine", callback_data="mode:imagine"),
            InlineKeyboardButton("🎵 Lyrics", callback_data="mode:lyrics"),
        ],
        [
            InlineKeyboardButton("🖼 Describe", callback_data="mode:caption"),
            InlineKeyboardButton("🔤 OCR", callback_data="mode:ocr"),
        ],
        [
            InlineKeyboardButton("📦 Detect", callback_data="mode:detect"),
            InlineKeyboardButton("🪄 No BG", callback_data="mode:bg"),
        ],
        [InlineKeyboardButton("✏️ Sketch", callback_data="mode:sketch")],
    ]
)

START_HTML = (
    "<b>Image Bot</b>\n"
    "Telegram → Render → your Space → you.\n\n"
    "<b>Create</b>\n"
    "✨ Imagine — FLUX.1-schnell on your ZeroGPU\n"
    "🎵 Lyrics — song name (lrclib, via your Space)\n\n"
    "<b>Photo tools</b>\n"
    "🖼 Describe · 🔤 OCR · 📦 Detect · 🪄 No BG · ✏️ Sketch\n\n"
    "<i>Type a prompt for FLUX, a song name for lyrics,\n"
    "or send a photo for the selected tool.\n"
    "First FLUX run downloads the model — can take a few minutes.</i>"
)

MODE_HINT = {
    "imagine": "Send a prompt.\nExample: a tea stall in Rangpur rain, cinematic",
    "lyrics": "Send a song name.\nExample: Shape of You",
    "caption": "Send a photo — I will describe it.",
    "ocr": "Send a photo with text to read.",
    "detect": "Send a photo — I will box objects.",
    "bg": "Send a photo — I will cut the background.",
    "sketch": "Send a photo — I will draw a pencil sketch.",
}

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("Missing TELEGRAM_BOT_TOKEN environment variable.")

_space = None


def _client():
    global _space
    if _space is None:
        from gradio_client import Client

        logger.info("Connecting to your Space %s …", HF_SPACE_ID)
        try:
            _space = Client(
                HF_SPACE_ID,
                hf_token=HF_API_TOKEN,
                verbose=False,
                httpx_kwargs={"timeout": 180.0},
            )
        except TypeError:
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


def _predict(*args, api_name: str):
    return _client().predict(*args, api_name=api_name)


def call_caption(image_path: str) -> str:
    from gradio_client import handle_file

    return _as_text(_predict(handle_file(image_path), api_name="/caption"))


def call_ocr(image_path: str) -> str:
    from gradio_client import handle_file

    return _as_text(_predict(handle_file(image_path), api_name="/ocr"))


def call_detect(image_path: str) -> tuple[str, Optional[str]]:
    from gradio_client import handle_file

    result = _predict(handle_file(image_path), api_name="/detect")
    if isinstance(result, (list, tuple)) and len(result) >= 2:
        return _as_text(result[1]), _as_path(result[0])
    return _as_text(result), _as_path(result)


def call_bg(image_path: str) -> str:
    from gradio_client import handle_file

    path = _as_path(_predict(handle_file(image_path), api_name="/bg"))
    if not path:
        raise RuntimeError("Space /bg returned no image")
    return path


def call_sketch(image_path: str) -> str:
    from gradio_client import handle_file

    path = _as_path(_predict(handle_file(image_path), api_name="/sketch"))
    if not path:
        raise RuntimeError("Space /sketch returned no image")
    return path


def call_imagine(prompt: str) -> tuple[str, str]:
    result = _predict(prompt, api_name="/imagine")
    path = _as_path(result)
    text = _as_text(result)
    if not path:
        raise RuntimeError(text or "FLUX returned no image")
    return path, text


def call_lyrics(query: str) -> str:
    text = _as_text(_predict(query, api_name="/lyrics"))
    if not text:
        raise RuntimeError("No lyrics came back")
    return text


async def _reply_long(message, text: str) -> None:
    text = (text or "").strip() or "(empty)"
    for start in range(0, len(text), 3900):
        await message.reply_text(text[start : start + 3900], reply_markup=MENU_KEYBOARD)


async def _send_image(message, path: str, caption: str = "", as_document: bool = False) -> None:
    p = Path(path)
    if p.exists():
        with p.open("rb") as handle:
            if as_document:
                await message.reply_document(
                    document=handle,
                    filename=p.name,
                    caption=caption or None,
                    reply_markup=MENU_KEYBOARD,
                )
            else:
                await message.reply_photo(
                    photo=handle,
                    caption=caption or None,
                    reply_markup=MENU_KEYBOARD,
                )
        return
    if path.startswith("http"):
        await message.reply_photo(photo=path, caption=caption or None, reply_markup=MENU_KEYBOARD)
        return
    await message.reply_text(caption or "No image file came back.", reply_markup=MENU_KEYBOARD)


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
    if caption in {"sketch", "pencil", "/sketch"} or caption.startswith("sketch"):
        return "sketch"
    if caption in {"caption", "describe", "/caption"}:
        return "caption"
    return saved if saved in _PHOTO_MODES else "caption"


async def _send_menu(message) -> None:
    await message.reply_text(
        START_HTML,
        parse_mode="HTML",
        reply_markup=MENU_KEYBOARD,
    )
    await message.reply_text("Pick a tool:", reply_markup=INLINE_MENU)


async def _set_mode(message, context: ContextTypes.DEFAULT_TYPE, mode: str) -> None:
    context.user_data["mode"] = mode
    await message.reply_text(
        f"<b>{mode}</b>\n{MODE_HINT.get(mode, 'Send a photo or a prompt.')}",
        parse_mode="HTML",
        reply_markup=MENU_KEYBOARD,
    )


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _send_menu(update.message)


async def mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    raw = update.message.text or ""
    cmd = raw.split()[0].lstrip("/").split("@")[0]
    rest = raw.split(None, 1)[1].strip() if len(raw.split(None, 1)) > 1 else ""
    if cmd in {"menu", "help"}:
        await _send_menu(update.message)
        return
    if cmd not in _ALL_MODES:
        await _send_menu(update.message)
        return
    context.user_data["mode"] = cmd
    if rest and cmd in _PROMPT_MODES:
        await _run_prompt(update.message, cmd, rest)
        return
    await _set_mode(update.message, context, cmd)


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    data = query.data or ""
    if data.startswith("mode:"):
        mode = data.split(":", 1)[1]
        if mode in _ALL_MODES:
            context.user_data["mode"] = mode
            await query.message.reply_text(
                f"<b>{mode}</b>\n{MODE_HINT.get(mode, '')}",
                parse_mode="HTML",
                reply_markup=MENU_KEYBOARD,
            )


async def _run_prompt(message, mode: str, prompt: str) -> None:
    await message.reply_text(
        f"Working ({mode})… first FLUX run can take a few minutes.",
        reply_markup=MENU_KEYBOARD,
    )
    try:
        if mode == "lyrics":
            await message.chat.send_action(action="typing")
            text = await asyncio.to_thread(call_lyrics, prompt)
            await _reply_long(message, text)
            return
        await message.chat.send_action(action="upload_photo")
        path, note = await asyncio.to_thread(call_imagine, prompt)
        await _send_image(message, path, caption=note or "FLUX")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prompt call failed (%s)", mode)
        await message.reply_text(
            f"Could not finish ({mode}).\n({exc.__class__.__name__}: {exc})",
            reply_markup=MENU_KEYBOARD,
        )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (update.message.text or "").strip()
    if text == "📋 Menu":
        await _send_menu(update.message)
        return
    if text in _BTN_MODE:
        await _set_mode(update.message, context, _BTN_MODE[text])
        return
    mode = (context.user_data or {}).get("mode") or "imagine"
    if mode not in _PROMPT_MODES:
        mode = "imagine"
        context.user_data["mode"] = "imagine"
    await _run_prompt(update.message, mode, text)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    mode = _photo_mode(update, context)
    await update.message.reply_text(
        f"Sending to your Space ({mode})… first run can take a minute.",
        reply_markup=MENU_KEYBOARD,
    )
    tmp_path = None
    try:
        tmp_path = await _download_photo(update, context)
        if mode == "bg":
            await update.message.chat.send_action(action="upload_document")
            out = await asyncio.to_thread(call_bg, tmp_path)
            await _send_image(update.message, out, caption="background removed", as_document=True)
            return
        if mode == "sketch":
            await update.message.chat.send_action(action="upload_photo")
            out = await asyncio.to_thread(call_sketch, tmp_path)
            await _send_image(update.message, out, caption="sketch")
            return
        if mode == "detect":
            await update.message.chat.send_action(action="upload_photo")
            text, image = await asyncio.to_thread(call_detect, tmp_path)
            if image:
                await _send_image(update.message, image, caption=(text or "detect")[:900])
            if text:
                await _reply_long(update.message, text)
            return
        await update.message.chat.send_action(action="typing")
        fn = call_ocr if mode == "ocr" else call_caption
        text = await asyncio.to_thread(fn, tmp_path)
        await _reply_long(update.message, text)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Space call failed (%s)", mode)
        await update.message.reply_text(
            "Could not finish that request.\n"
            f"({exc.__class__.__name__}: {exc})",
            reply_markup=MENU_KEYBOARD,
        )
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


telegram_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
telegram_app.add_handler(CommandHandler("start", start_command))
telegram_app.add_handler(CommandHandler("help", start_command))
telegram_app.add_handler(CommandHandler("menu", start_command))
for _cmd in ("caption", "ocr", "detect", "bg", "sketch", "imagine", "lyrics"):
    telegram_app.add_handler(CommandHandler(_cmd, mode_command))
telegram_app.add_handler(CallbackQueryHandler(on_button))
telegram_app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_photo))
telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

app = FastAPI()


@app.on_event("startup")
async def on_startup() -> None:
    await telegram_app.initialize()
    await telegram_app.start()
    try:
        await telegram_app.bot.set_my_commands(
            [
                BotCommand("start", "Open the menu"),
                BotCommand("imagine", "FLUX text → image"),
                BotCommand("lyrics", "Song name → lyrics"),
                BotCommand("caption", "Describe a photo"),
                BotCommand("ocr", "Read text in a photo"),
                BotCommand("detect", "Find objects"),
                BotCommand("bg", "Remove background"),
                BotCommand("sketch", "Pencil sketch"),
                BotCommand("menu", "Show buttons"),
            ]
        )
    except Exception:  # noqa: BLE001
        logger.exception("Could not set bot commands")
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
    return {
        "status": "ok",
        "space": HF_SPACE_ID,
        "apis": [
            "/imagine",
            "/caption",
            "/ocr",
            "/detect",
            "/bg",
            "/sketch",
            "/lyrics",
        ],
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
