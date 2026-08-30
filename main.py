"""
Telegram → Render → your Hugging Face Space → Render → Telegram.

FLUX.1-schnell, photo tools, and lyrics all run through YOUR Space.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import requests

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

# Hugging Face Space URL this Render app pings so the Space does not sleep.
# Format: https://<username>-<space-name>.hf.space
# Override with HF_SPACE_PING_URL (comma-separated if more than one Space).
# Extra targets (pinged at the same time): PING_URLS=https://a/,https://b/,https://c/
HF_SPACE_PING_URL = (
    os.environ.get("HF_SPACE_PING_URL")
    or "https://madarauchihagmailcom-my.hf.space/"
).strip()
# Render → Hugging Face: every 10 hours.
KEEP_ALIVE_INTERVAL_SECONDS = 10 * 60 * 60
KEEP_ALIVE_TIMEOUT_SECONDS = 30
_MAX_DOWNLOAD_BYTES = 19 * 1024 * 1024

_PHOTO_MODES = {"caption", "ocr", "bg", "detect", "sketch"}
_PROMPT_MODES = {"chat", "imagine", "lyrics", "git"}
_ALL_MODES = _PHOTO_MODES | _PROMPT_MODES
_REFER_FILE = Path(os.environ.get("REFER_FILE") or "refer.json")
_BOT_USERNAME = ""
_POINTS_PER_REF = 10

_BTN_MODE = {
    "💬 Chat": "chat",
    "✨ Imagine": "imagine",
    "🎵 Lyrics": "lyrics",
    "📁 GitHub": "git",
    "💸 Refer": "refer",
    "🖼 Describe": "caption",
    "🔤 OCR": "ocr",
    "📦 Detect": "detect",
    "🪄 No BG": "bg",
    "✏️ Sketch": "sketch",
}

MENU_KEYBOARD = ReplyKeyboardMarkup(
    [
        [KeyboardButton("💬 Chat"), KeyboardButton("✨ Imagine")],
        [KeyboardButton("🎵 Lyrics"), KeyboardButton("📁 GitHub")],
        [KeyboardButton("💸 Refer"), KeyboardButton("🖼 Describe")],
        [KeyboardButton("🔤 OCR"), KeyboardButton("📦 Detect")],
        [KeyboardButton("🪄 No BG"), KeyboardButton("✏️ Sketch")],
        [KeyboardButton("📋 Menu")],
    ],
    resize_keyboard=True,
    is_persistent=True,
)

INLINE_MENU = InlineKeyboardMarkup(
    [
        [
            InlineKeyboardButton("💬 Chat", callback_data="mode:chat"),
            InlineKeyboardButton("✨ Imagine", callback_data="mode:imagine"),
        ],
        [
            InlineKeyboardButton("🎵 Lyrics", callback_data="mode:lyrics"),
            InlineKeyboardButton("📁 GitHub", callback_data="mode:git"),
        ],
        [
            InlineKeyboardButton("💸 Refer", callback_data="mode:refer"),
            InlineKeyboardButton("🖼 Describe", callback_data="mode:caption"),
        ],
        [
            InlineKeyboardButton("🔤 OCR", callback_data="mode:ocr"),
            InlineKeyboardButton("📦 Detect", callback_data="mode:detect"),
        ],
        [
            InlineKeyboardButton("🪄 No BG", callback_data="mode:bg"),
            InlineKeyboardButton("✏️ Sketch", callback_data="mode:sketch"),
        ],
    ]
)

START_HTML = (
    "<b>Image Bot</b>\n"
    "Telegram → Render → আপনার Space → আপনি।\n\n"
    "<b>এআই</b>\n"
    "💬 Chat — Llama 3.2 3B (দ্রুত, ZeroGPU)\n"
    "✨ Imagine — FLUX লেখা → ছবি\n\n"
    "<b>অন্যান্য</b>\n"
    "🎵 Lyrics · 📁 GitHub ZIP · 💸 Refer\n"
    "🖼 Describe · 🔤 OCR · 📦 Detect · 🪄 No BG · ✏️ Sketch\n\n"
    "<i>ডিফল্ট: চ্যাট। প্রথম চ্যাটে মডেল ডাউনলোড হতে পারে।</i>"
)

MODE_HINT = {
    "chat": "কথা বলুন। Llama 3.2 3B উত্তর দেবে।\n/imagine দিলে ছবি মোডে যাবে।",
    "imagine": "একটা প্রম্পট পাঠান।\nযেমন: a tea stall in Rangpur rain, cinematic",
    "lyrics": "গানের নাম পাঠান।\nযেমন: Shape of You",
    "git": "GitHub রিপো পাঠান।\nযেমন: tajhatAti/Lyr",
    "caption": "একটা ছবি পাঠান — বর্ণনা করব।",
    "ocr": "লেখার ছবি পাঠান — পড়ে দেব।",
    "detect": "ছবি পাঠান — বস্তু বক্স করব।",
    "bg": "ছবি পাঠান — ব্যাকগ্রাউন্ড কাটব।",
    "sketch": "ছবি পাঠান — স্কেচ করব।",
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


def call_gitzip(repo: str) -> tuple[str, str]:
    result = _predict(repo, api_name="/gitzip")
    path = _as_path(result)
    text = _as_text(result)
    if not path:
        raise RuntimeError(text or "GitHub ZIP came back empty")
    return path, text


def call_chat(prompt: str) -> str:
    text = _as_text(_predict(prompt, api_name="/chat"))
    if not text:
        raise RuntimeError("Chat returned empty")
    return text


def _refer_load() -> dict:
    if not _REFER_FILE.exists():
        return {}
    try:
        return json.loads(_REFER_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _refer_save(data: dict) -> None:
    _REFER_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _refer_user(uid: int) -> dict:
    data = _refer_load()
    key = str(uid)
    entry = data.get(key) or {"points": 0, "refs": [], "via": None}
    data[key] = entry
    _refer_save(data)
    return entry


def _refer_apply(new_uid: int, via_uid: int) -> str:
    if new_uid == via_uid:
        return "নিজেকে রেফার করা যায় না।"
    data = _refer_load()
    newbie = data.get(str(new_uid)) or {"points": 0, "refs": [], "via": None}
    if newbie.get("via"):
        return "আপনি আগেই রেফার হয়েছেন।"
    host = data.get(str(via_uid)) or {"points": 0, "refs": [], "via": None}
    refs = list(host.get("refs") or [])
    if str(new_uid) in refs:
        return "এই ইউজার আগেই কাউন্ট হয়েছে।"
    newbie["via"] = via_uid
    refs.append(str(new_uid))
    host["refs"] = refs
    host["points"] = int(host.get("points") or 0) + _POINTS_PER_REF
    data[str(new_uid)] = newbie
    data[str(via_uid)] = host
    _refer_save(data)
    return f"রেফার OK। ইউজার {via_uid} পেলেন {_POINTS_PER_REF} পয়েন্ট।"


def _refer_text(uid: int) -> str:
    entry = _refer_user(uid)
    uname = _BOT_USERNAME or "your_bot"
    link = f"https://t.me/{uname}?start=ref_{uid}"
    n = len(entry.get("refs") or [])
    pts = int(entry.get("points") or 0)
    return (
        f"<b>রেফার / আর্নিং</b>\n"
        f"পয়েন্ট: <b>{pts}</b>\n"
        f"রেফার সংখ্যা: <b>{n}</b> (প্রতি রেফারে {_POINTS_PER_REF} পয়েন্ট)\n\n"
        f"আপনার লিংক:\n<code>{link}</code>\n\n"
        f"এই লিংকে কেউ /start দিলে আপনি পয়েন্ট পাবেন।\n"
        f"<i>এটা ইন-বট পয়েন্ট — আসল টাকা/পেআউট না। "
        f"Render রিডিপ্লয় হলে হিসাব রিসেট হতে পারে।</i>"
    )


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
    args = context.args or []
    uid = update.effective_user.id if update.effective_user else 0
    if args and str(args[0]).startswith("ref_") and uid:
        try:
            via = int(str(args[0])[4:])
            note = _refer_apply(uid, via)
            await update.message.reply_text(note, reply_markup=MENU_KEYBOARD)
        except ValueError:
            pass
    await _send_menu(update.message)


async def refer_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id if update.effective_user else 0
    await update.message.reply_text(
        _refer_text(uid),
        parse_mode="HTML",
        reply_markup=MENU_KEYBOARD,
        disable_web_page_preview=True,
    )


async def mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    raw = update.message.text or ""
    cmd = raw.split()[0].lstrip("/").split("@")[0]
    rest = raw.split(None, 1)[1].strip() if len(raw.split(None, 1)) > 1 else ""
    if cmd in {"menu", "help"}:
        await _send_menu(update.message)
        return
    if cmd in {"refer", "earn"}:
        await refer_command(update, context)
        return
    if cmd not in _ALL_MODES:
        await _send_menu(update.message)
        return
    context.user_data["mode"] = cmd
    if rest and cmd in _PROMPT_MODES:
        await _run_prompt(update.message, cmd, rest, context)
        return
    await _set_mode(update.message, context, cmd)


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    data = query.data or ""
    if data.startswith("mode:"):
        mode = data.split(":", 1)[1]
        if mode == "refer":
            uid = update.effective_user.id if update.effective_user else 0
            await query.message.reply_text(
                _refer_text(uid),
                parse_mode="HTML",
                reply_markup=MENU_KEYBOARD,
                disable_web_page_preview=True,
            )
            return
        if mode in _ALL_MODES:
            context.user_data["mode"] = mode
            await query.message.reply_text(
                f"<b>{mode}</b>\n{MODE_HINT.get(mode, '')}",
                parse_mode="HTML",
                reply_markup=MENU_KEYBOARD,
            )


async def _run_prompt(
    message,
    mode: str,
    prompt: str,
    context: ContextTypes.DEFAULT_TYPE | None = None,
) -> None:
    if mode != "chat":
        await message.reply_text(f"Working ({mode})…", reply_markup=MENU_KEYBOARD)
    try:
        if mode == "chat":
            await message.chat.send_action(action="typing")
            packed = prompt
            hist: list = []
            if context is not None:
                hist = list((context.user_data or {}).get("chat_hist") or [])
                bits = [
                    f"{t.get('role')}: {str(t.get('content') or '')[:400]}"
                    for t in hist[-6:]
                ]
                bits.append(f"user: {prompt}")
                packed = "\n".join(bits)
            text = await asyncio.to_thread(call_chat, packed)
            if context is not None:
                hist.append({"role": "user", "content": prompt[:400]})
                hist.append({"role": "assistant", "content": text[:800]})
                context.user_data["chat_hist"] = hist[-12:]
            await _reply_long(message, text)
            return
        if mode == "lyrics":
            await message.chat.send_action(action="typing")
            text = await asyncio.to_thread(call_lyrics, prompt)
            await _reply_long(message, text)
            return
        if mode == "git":
            await message.chat.send_action(action="upload_document")
            path, note = await asyncio.to_thread(call_gitzip, prompt)
            await _send_image(message, path, caption=note or "repo.zip", as_document=True)
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
        mode = _BTN_MODE[text]
        if mode == "refer":
            await refer_command(update, context)
            return
        await _set_mode(update.message, context, mode)
        return
    mode = (context.user_data or {}).get("mode") or "chat"
    if mode not in _PROMPT_MODES:
        mode = "chat"
        context.user_data["mode"] = "chat"
    await _run_prompt(update.message, mode, text, context)


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


def _split_ping_urls(*chunks: str) -> list[str]:
    """Parse comma / semicolon / whitespace separated https URLs, de-duplicated."""
    seen: set[str] = set()
    out: list[str] = []
    for chunk in chunks:
        for raw in re.split(r"[\s,;]+", chunk or ""):
            item = raw.strip().strip("'\"")
            if not item:
                continue
            if "://" not in item:
                item = "https://" + item
            parsed = urlparse(item)
            host = (parsed.hostname or "").lower()
            if parsed.scheme != "https" or not host:
                print(f"[keep-alive] skipped (need https URL): {item}", flush=True)
                continue
            url = f"https://{host}{parsed.path or '/'}"
            if url not in seen:
                seen.add(url)
                out.append(url)
    return out


def _keep_alive_targets() -> list[str]:
    """Primary Space URL(s) plus optional extra PING_URLS, all hit together."""
    extra = os.environ.get("PING_URLS") or os.environ.get("KEEP_ALIVE_URLS") or ""
    urls = _split_ping_urls(HF_SPACE_PING_URL, extra)
    return urls or _split_ping_urls("https://madarauchihagmailcom-my.hf.space/")


def _ping_one(url: str) -> None:
    """GET one URL. Never raises — timeout / connection errors are logged."""
    try:
        response = requests.get(
            url,
            timeout=KEEP_ALIVE_TIMEOUT_SECONDS,
            headers={"User-Agent": "Rend-KeepAlive/1.0", "Accept": "text/html"},
        )
        msg = f"[keep-alive] success {url} status={response.status_code}"
        print(msg, flush=True)
        logger.info("Keep-alive ping ok: %s status=%s", url, response.status_code)
    except requests.exceptions.Timeout:
        msg = f"[keep-alive] timeout after {KEEP_ALIVE_TIMEOUT_SECONDS}s: {url}"
        print(msg, flush=True)
        logger.warning("Keep-alive ping timeout: %s", url)
    except requests.exceptions.ConnectionError as exc:
        msg = f"[keep-alive] connection error: {url} ({exc})"
        print(msg, flush=True)
        logger.warning("Keep-alive ping connection error: %s (%s)", url, exc)
    except Exception as exc:  # noqa: BLE001 — never let this thread kill the bot
        msg = f"[keep-alive] failed: {url} ({exc})"
        print(msg, flush=True)
        logger.warning("Keep-alive ping failed: %s (%s)", url, exc)


def _keep_space_awake() -> None:
    """Background loop: GET Hugging Face (and extra URLs) every 10 hours, in parallel.

    Daemon thread so FastAPI / Telegram keep serving. Failures never crash the app.
    """
    urls = _keep_alive_targets()
    logger.info(
        "Keep-alive thread started. ping=%s every %ss",
        urls,
        KEEP_ALIVE_INTERVAL_SECONDS,
    )
    print(
        f"[keep-alive] started → {urls} every {KEEP_ALIVE_INTERVAL_SECONDS}s (parallel)",
        flush=True,
    )
    workers = min(8, max(1, len(urls)))
    while True:
        # Ping all URLs at the same time, then sleep 10 hours.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_ping_one, urls))
        time.sleep(KEEP_ALIVE_INTERVAL_SECONDS)


def _start_keep_alive_thread() -> None:
    """Start the Space ping as a daemon thread (does not block shutdown)."""
    thread = threading.Thread(
        target=_keep_space_awake,
        name="hf-space-keep-alive",
        daemon=True,
    )
    thread.start()


telegram_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
telegram_app.add_handler(CommandHandler("start", start_command))
telegram_app.add_handler(CommandHandler("help", start_command))
telegram_app.add_handler(CommandHandler("menu", start_command))
telegram_app.add_handler(CommandHandler("refer", refer_command))
telegram_app.add_handler(CommandHandler("earn", refer_command))
for _cmd in ("caption", "ocr", "detect", "bg", "sketch", "imagine", "lyrics", "git", "chat"):
    telegram_app.add_handler(CommandHandler(_cmd, mode_command))
telegram_app.add_handler(CallbackQueryHandler(on_button))
telegram_app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_photo))
telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

app = FastAPI()


@app.on_event("startup")
async def on_startup() -> None:
    await telegram_app.initialize()
    await telegram_app.start()
    global _BOT_USERNAME
    try:
        me = await telegram_app.bot.get_me()
        _BOT_USERNAME = me.username or ""
        await telegram_app.bot.set_my_commands(
            [
                BotCommand("start", "মেনু"),
                BotCommand("chat", "Llama 3.2 চ্যাট"),
                BotCommand("imagine", "FLUX লেখা → ছবি"),
                BotCommand("lyrics", "গানের লিরিক্স"),
                BotCommand("git", "GitHub রিপো ZIP"),
                BotCommand("refer", "রেফার লিংক / পয়েন্ট"),
                BotCommand("caption", "ছবি বর্ণনা"),
                BotCommand("ocr", "ছবির লেখা"),
                BotCommand("detect", "বস্তু খোঁজা"),
                BotCommand("bg", "ব্যাকগ্রাউন্ড কাটা"),
                BotCommand("sketch", "স্কেচ"),
                BotCommand("menu", "বাটন"),
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
    # FastAPI startup: wake the Space in the background (does not block requests).
    _start_keep_alive_thread()


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
