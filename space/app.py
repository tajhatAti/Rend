"""
Image + lyrics API on YOUR Hugging Face Space.

On this ZeroGPU: /caption /ocr /detect /bg /sketch /imagine (FLUX.1-schnell)
Lyrics lookup:   /lyrics  (lrclib.net, no GPU)
"""

from __future__ import annotations

import io
import json
import os
import random
import re
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache

import requests

import gradio as gr
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

try:
    import spaces
except ImportError:  # local / plain CPU Space
    class spaces:  # type: ignore[no-redef]
        @staticmethod
        def GPU(fn=None, **_kwargs):
            if fn is None:
                return lambda f: f
            return fn


def _rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


@lru_cache(maxsize=1)
def _captioner():
    from transformers import pipeline

    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")


@lru_cache(maxsize=1)
def _ocr_models():
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    return processor, model


@lru_cache(maxsize=1)
def _detector():
    from transformers import pipeline

    return pipeline("object-detection", model="facebook/detr-resnet-50")


@lru_cache(maxsize=1)
def _flux():
    import torch
    from diffusers import FluxPipeline

    return FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
    )


@spaces.GPU
def caption(image: Image.Image) -> str:
    if image is None:
        return "No image."
    result = _captioner()(_rgb(image))
    if isinstance(result, list) and result:
        return result[0].get("generated_text") or str(result[0])
    return str(result)


@spaces.GPU
def ocr(image: Image.Image) -> str:
    if image is None:
        return "No image."
    processor, model = _ocr_models()
    pixel_values = processor(images=_rgb(image), return_tensors="pt").pixel_values
    ids = model.generate(pixel_values)
    return processor.batch_decode(ids, skip_special_tokens=True)[0]


@spaces.GPU
def detect(image: Image.Image):
    if image is None:
        return None, "No image."
    rgb = _rgb(image)
    results = _detector()(rgb)
    annotated = rgb.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    lines = []
    for item in results:
        score = float(item.get("score") or 0)
        if score < 0.5:
            continue
        label = str(item.get("label") or "object")
        box = item.get("box") or {}
        xmin, ymin = int(box.get("xmin", 0)), int(box.get("ymin", 0))
        xmax, ymax = int(box.get("xmax", 0)), int(box.get("ymax", 0))
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
        tag = f"{label} {score:.2f}"
        draw.text((xmin + 4, ymin + 4), tag, fill="red", font=font)
        lines.append(tag)
    return annotated, "\n".join(lines) or "No objects above 0.5 confidence."


def remove_bg(image: Image.Image) -> Image.Image:
    if image is None:
        return None
    from rembg import remove

    buf = io.BytesIO()
    _rgb(image).save(buf, format="PNG")
    cut = remove(buf.getvalue())
    return Image.open(io.BytesIO(cut)).convert("RGBA")


def sketch(image: Image.Image):
    if image is None:
        return None, "No image."
    rgb = _rgb(image)
    rgb.thumbnail((1024, 1024))
    gray = ImageOps.grayscale(rgb)
    inverted = ImageOps.invert(gray)
    blur = inverted.filter(ImageFilter.GaussianBlur(radius=18))
    gpx = gray.load()
    bpx = blur.load()
    out = Image.new("L", gray.size)
    opx = out.load()
    w, h = gray.size
    for y in range(h):
        for x in range(w):
            b = bpx[x, y]
            if b >= 255:
                opx[x, y] = 255
            else:
                opx[x, y] = min(255, int(gpx[x, y] * 255 / (255 - b)))
    return out.convert("RGB"), "pencil sketch"


@spaces.GPU(duration=120)
def imagine(prompt: str):
    text = (prompt or "").strip()
    if not text:
        return None, "Type a prompt."
    import torch

    pipe = _flux()
    if torch.cuda.is_available():
        pipe.to("cuda")
    image = pipe(
        text[:500],
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
        height=768,
        width=768,
        generator=torch.Generator("cpu").manual_seed(random.randint(1, 2_000_000_000)),
    ).images[0]
    return image, "FLUX.1-schnell"


def lyrics(query: str) -> str:
    text = (query or "").strip()
    if not text:
        return "Type a song name."
    url = "https://lrclib.net/api/search?q=" + urllib.parse.quote(text[:200])
    req = urllib.request.Request(url, headers={"User-Agent": "ImageBot/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        return f"Lyrics lookup failed (HTTP {exc.code})."
    except Exception as exc:  # noqa: BLE001
        return f"Lyrics lookup failed: {exc}"
    if not isinstance(data, list) or not data:
        return f"No lyrics found for “{text}”."
    for item in data:
        body = (item or {}).get("plainLyrics") or (item or {}).get("syncedLyrics")
        if not body:
            continue
        track = item.get("trackName") or text
        artist = item.get("artistName") or "Unknown"
        out = f"🎵 {track} — {artist}\n\n{body}"
        return out if len(out) < 8000 else out[:7900] + "\n\n…"
    return f"No lyrics found for “{text}”."


_GIT_MAX = 40 * 1024 * 1024


def _parse_repo(text: str) -> tuple[str, str]:
    raw = (text or "").strip()
    raw = raw.replace("https://github.com/", "").replace("http://github.com/", "")
    raw = raw.replace("www.github.com/", "").removesuffix(".git").strip("/")
    parts = [p for p in raw.split("/") if p]
    if len(parts) < 2 or not re.match(r"^[\w.-]+$", parts[0]) or not re.match(r"^[\w.-]+$", parts[1]):
        raise ValueError("Send owner/repo or a GitHub URL. Example: tajhatAti/Lyr")
    return parts[0], parts[1]


def git_zip(repo: str):
    try:
        owner, name = _parse_repo(repo)
    except ValueError as exc:
        return None, str(exc)
    token = (os.environ.get("GITHUB_TOKEN") or "").strip()
    headers = {"User-Agent": "ImageBot/1.0", "Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    last = "no branch"
    for branch in ("main", "master"):
        url = f"https://codeload.github.com/{owner}/{name}/zip/refs/heads/{branch}"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read(_GIT_MAX + 1)
        except urllib.error.HTTPError as exc:
            last = f"{branch}: HTTP {exc.code}"
            continue
        except Exception as exc:  # noqa: BLE001
            last = f"{branch}: {exc}"
            continue
        if len(data) > _GIT_MAX:
            return None, f"{owner}/{name} is larger than ~40 MB (Telegram bot limit)."
        if len(data) < 100:
            last = f"{branch}: empty"
            continue
        fd, path = tempfile.mkstemp(suffix=f"-{name}.zip")
        os.close(fd)
        with open(path, "wb") as handle:
            handle.write(data)
        mb = len(data) / (1024 * 1024)
        return path, f"{owner}/{name} ({branch}, {mb:.1f} MB)"
    return None, f"Could not download {owner}/{name}. {last}"


_CHAT_ID = "meta-llama/Llama-3.2-3B-Instruct"
_chat_pipe = None


def _load_chat():
    global _chat_pipe
    if _chat_pipe is not None:
        return _chat_pipe
    import torch
    from transformers import pipeline

    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip() or None
    kwargs = {
        "task": "text-generation",
        "model": _CHAT_ID,
        "torch_dtype": torch.bfloat16,
        "device_map": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if token:
        kwargs["token"] = token
    _chat_pipe = pipeline(**kwargs)
    return _chat_pipe


@spaces.GPU(duration=30)
def chat(prompt: str) -> str:
    text = (prompt or "").strip()
    if not text:
        return "Type a message."
    text = text[:4000]
    pipe = _load_chat()
    messages = [
        {"role": "system", "content": "You are a fast, helpful assistant. Reply in the user's language. Keep answers short."},
        {"role": "user", "content": text},
    ]
    out = pipe(
        messages,
        max_new_tokens=160,
        do_sample=False,
        return_full_text=False,
    )
    if isinstance(out, list) and out:
        item = out[0]
        if isinstance(item, dict):
            return str(item.get("generated_text") or item).strip()
        return str(item).strip()
    return str(out).strip()


with gr.Blocks(title="Image Bot Space") as demo:
    gr.Markdown(
        "# Image Bot Space\n"
        "Telegram backend. **Chat:** Llama-3.2-3B-Instruct on ZeroGPU (`/chat`). "
        "**Imagine:** FLUX.1-schnell. Photo tools + lyrics + GitHub ZIP."
    )
    image = gr.Image(type="pil", label="Image")

    with gr.Tab("Imagine"):
        pr = gr.Textbox(label="Prompt", placeholder="a tea stall in Rangpur rain, cinematic")
        im_out = gr.Image(type="pil", label="FLUX")
        im_txt = gr.Textbox(label="Status")
        im_btn = gr.Button("Generate with FLUX")
        im_btn.click(imagine, inputs=pr, outputs=[im_out, im_txt], api_name="imagine")

    with gr.Tab("Caption"):
        cap_out = gr.Textbox(label="Caption")
        cap_btn = gr.Button("Caption")
        cap_btn.click(caption, inputs=image, outputs=cap_out, api_name="caption")

    with gr.Tab("OCR"):
        ocr_out = gr.Textbox(label="Text")
        ocr_btn = gr.Button("Read text")
        ocr_btn.click(ocr, inputs=image, outputs=ocr_out, api_name="ocr")

    with gr.Tab("Detect"):
        det_img = gr.Image(type="pil", label="Boxes")
        det_txt = gr.Textbox(label="Labels")
        det_btn = gr.Button("Detect")
        det_btn.click(detect, inputs=image, outputs=[det_img, det_txt], api_name="detect")

    with gr.Tab("Background"):
        bg_out = gr.Image(type="pil", label="No background", image_mode="RGBA")
        bg_btn = gr.Button("Remove background")
        bg_btn.click(remove_bg, inputs=image, outputs=bg_out, api_name="bg")

    with gr.Tab("Sketch"):
        sk_out = gr.Image(type="pil", label="Sketch")
        sk_txt = gr.Textbox(label="Status")
        sk_btn = gr.Button("Sketch")
        sk_btn.click(sketch, inputs=image, outputs=[sk_out, sk_txt], api_name="sketch")

    with gr.Tab("Lyrics"):
        ly_in = gr.Textbox(label="Song", placeholder="Shape of You")
        ly_out = gr.Textbox(label="Lyrics", lines=16)
        ly_btn = gr.Button("Find lyrics")
        ly_btn.click(lyrics, inputs=ly_in, outputs=ly_out, api_name="lyrics")

    with gr.Tab("GitHub"):
        gh_in = gr.Textbox(label="Repo", placeholder="tajhatAti/Lyr")
        gh_file = gr.File(label="ZIP")
        gh_txt = gr.Textbox(label="Status")
        gh_btn = gr.Button("Download repo ZIP")
        gh_btn.click(git_zip, inputs=gh_in, outputs=[gh_file, gh_txt], api_name="gitzip")

    with gr.Tab("Chat"):
        ch_in = gr.Textbox(label="Message", placeholder="কেমন আছো?")
        ch_out = gr.Textbox(label="Llama 3.2", lines=8)
        ch_btn = gr.Button("Chat")
        ch_btn.click(chat, inputs=ch_in, outputs=ch_out, api_name="chat")


# Render URL this Space pings so the free-tier web service does not sleep.
# Format: https://<app-name>.onrender.com/
# Override with RENDER_PING_URL if the Render hostname changes.
RENDER_PING_URL_DEFAULT = "https://rend-y1aw.onrender.com/"
KEEP_ALIVE_INTERVAL_SECONDS = 10 * 60 * 60  # 10 hours
KEEP_ALIVE_TIMEOUT_SECONDS = 30


def _render_ping_url() -> str:
    """Resolve the Render URL. Only https://*.onrender.com is allowed."""
    raw = (
        os.environ.get("RENDER_PING_URL")
        or os.environ.get("RENDER_URL")
        or os.environ.get("RENDER_EXTERNAL_URL")
        or RENDER_PING_URL_DEFAULT
    ).strip()
    if not raw:
        return ""
    if "://" not in raw:
        raw = "https://" + raw
    parsed = urllib.parse.urlparse(raw)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not host.endswith(".onrender.com"):
        print(f"[keep-alive] skipped: not https://*.onrender.com ({raw})", flush=True)
        return ""
    return f"https://{host}/"


def _keep_render_awake() -> None:
    """Background loop: GET Render every 10 hours.

    Daemon thread — Gradio keeps serving. Failures are logged, never crash the Space.
    """
    url = _render_ping_url()
    if not url:
        return
    print(f"[keep-alive] started → {url} every {KEEP_ALIVE_INTERVAL_SECONDS}s", flush=True)
    while True:
        # Ping first, then sleep, so a Space restart wakes Render immediately.
        try:
            response = requests.get(
                url,
                timeout=KEEP_ALIVE_TIMEOUT_SECONDS,
                headers={"User-Agent": "HF-Space-KeepAlive/1.0", "Accept": "application/json"},
            )
            print(f"[keep-alive] success {url} status={response.status_code}", flush=True)
        except requests.exceptions.Timeout:
            print(f"[keep-alive] timeout after {KEEP_ALIVE_TIMEOUT_SECONDS}s: {url}", flush=True)
        except requests.exceptions.ConnectionError as exc:
            print(f"[keep-alive] connection error: {url} ({exc})", flush=True)
        except Exception as exc:  # noqa: BLE001 — never let this thread kill Gradio
            print(f"[keep-alive] failed: {url} ({exc})", flush=True)
        time.sleep(KEEP_ALIVE_INTERVAL_SECONDS)


demo.queue()
# Start keep-alive as soon as the Space process loads (daemon = dies with Gradio).
threading.Thread(target=_keep_render_awake, daemon=True, name="render-ping").start()
if __name__ == "__main__":
    demo.launch()
