"""
Image API on YOUR Hugging Face Space.

Local ZeroGPU:  /caption /ocr /detect /bg /sketch
Online APIs:    /imagine /video /i2v   (Pollinations — not run on this GPU)
"""

from __future__ import annotations

import io
import os
import random
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache

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


def _pollinations_key() -> str:
    return (
        os.environ.get("POLLINATIONS_KEY")
        or os.environ.get("POLLINATIONS_API_KEY")
        or ""
    ).strip()


def _http_get(url: str, timeout: int = 180) -> bytes:
    key = _pollinations_key()
    if key and "key=" not in url:
        join = "&" if "?" in url else "?"
        url = f"{url}{join}key={urllib.parse.quote(key)}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 ImageBot",
            "Accept": "image/*,video/*,*/*",
            "Referer": "https://pollinations.ai/",
        },
    )
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            ctype = (resp.headers.get("Content-Type") or "").lower()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:400]
        raise RuntimeError(f"HTTP {exc.code}: {body or exc.reason}") from exc
    if data[:1] in (b"{", b"<") or "application/json" in ctype or "text/html" in ctype:
        text = data.decode("utf-8", "replace")[:400]
        raise RuntimeError(text or f"Unexpected {ctype}")
    if not data or len(data) < 32:
        raise RuntimeError("Empty response from API")
    return data


def _is_image(data: bytes) -> bool:
    return data[:8] == b"\x89PNG\r\n\x1a\n" or data[:3] == b"\xff\xd8\xff" or data[:4] == b"RIFF"


def _is_mp4(data: bytes) -> bool:
    return b"ftyp" in data[:64]


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
    # Color dodge: min(255, gray * 255 / (255 - blur))
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


def imagine(prompt: str):
    text = (prompt or "").strip()
    if not text:
        return None, "Type a prompt."
    text = text[:800]
    q = urllib.parse.quote(text)
    last = "no attempt"
    for model in ("flux", "sana", "dreamshaper", "zimage"):
        url = (
            f"https://image.pollinations.ai/prompt/{q}"
            f"?width=1024&height=1024&nologo=true&model={model}&safe=true"
        )
        try:
            data = _http_get(url, timeout=120)
            if not _is_image(data):
                last = f"{model}: not an image"
                continue
            img = Image.open(io.BytesIO(data))
            img.load()
            return img.convert("RGB"), f"Pollinations {model}"
        except Exception as exc:  # noqa: BLE001
            last = f"{model}: {exc}"
    return None, f"Text-to-image failed. {last}"


def _write_mp4(data: bytes) -> str:
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    with open(path, "wb") as handle:
        handle.write(data)
    return path


def make_video(prompt: str):
    text = (prompt or "").strip()
    if not text:
        return None, "Type a prompt."
    text = text[:500]
    q = urllib.parse.quote(text)
    last = "no attempt"
    queries = [
        f"https://gen.pollinations.ai/video/{q}?duration=4&resolution=480p&aspectRatio=16:9",
        f"https://gen.pollinations.ai/video/{q}?model=seedance-2.0-fast&duration=4&resolution=480p",
    ]
    for url in queries:
        try:
            data = _http_get(url, timeout=180)
            if not _is_mp4(data):
                last = "response was not mp4"
                continue
            return _write_mp4(data), "Pollinations video"
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
    hint = ""
    if not _pollinations_key():
        hint = (
            " Video models are not free-anonymous. "
            "Add a free key from enter.pollinations.ai as Space secret POLLINATIONS_KEY."
        )
    return None, f"Prompt-to-video failed. {last}.{hint}"


def image_to_video(image: Image.Image, prompt: str):
    if image is None:
        return None, "No image."
    text = (prompt or "").strip() or "slow cinematic camera move, natural motion"
    text = text[:500]
    q = urllib.parse.quote(text)
    # Pollinations i2v wants a public image URL. We only have pixels here, so
    # skip third-party hosts and ask for a key + media upload when available.
    key = _pollinations_key()
    if not key:
        return None, (
            "Image-to-video needs a public frame URL. "
            "Add a free POLLINATIONS_KEY on the Space (enter.pollinations.ai), "
            "or use /video with a text prompt instead."
        )
    buf = io.BytesIO()
    frame = _rgb(image)
    frame.thumbnail((768, 768))
    frame.save(buf, format="JPEG", quality=85)
    raw = buf.getvalue()
    boundary = "----ImageBotBoundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="frame.jpg"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode() + raw + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        "https://media.pollinations.ai/upload",
        data=body,
        method="POST",
        headers={
            "User-Agent": "ImageBot/1.0",
            "Authorization": f"Bearer {key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            uploaded = resp.read().decode("utf-8", "replace").strip()
    except urllib.error.HTTPError as exc:
        body_txt = exc.read().decode("utf-8", "replace")[:300]
        return None, f"Could not upload frame (HTTP {exc.code}): {body_txt or exc.reason}"
    except Exception as exc:  # noqa: BLE001
        return None, f"Could not upload frame: {exc}"

    frame_url = uploaded
    if uploaded.startswith("{"):
        if "http" in uploaded:
            start = uploaded.find("http")
            end = start
            while end < len(uploaded) and uploaded[end] not in "\"' <>":
                end += 1
            frame_url = uploaded[start:end]
        else:
            return None, f"Upload did not return a URL: {uploaded[:200]}"
    if not frame_url.startswith("http"):
        frame_url = f"https://media.pollinations.ai/{uploaded.strip('/')}"

    url = (
        f"https://gen.pollinations.ai/video/{q}"
        f"?duration=4&resolution=480p&aspectRatio=16:9"
        f"&image={urllib.parse.quote(frame_url, safe='')}"
    )
    try:
        data = _http_get(url, timeout=180)
        if not _is_mp4(data):
            return None, "Video API did not return mp4."
        return _write_mp4(data), "Pollinations image→video"
    except Exception as exc:  # noqa: BLE001
        return None, f"Image-to-video failed: {exc}"


with gr.Blocks(title="Image Bot Space") as demo:
    gr.Markdown(
        "# Image Bot Space\n"
        "Telegram backend. **On this GPU:** caption, OCR, detect, background, sketch. "
        "**Online APIs (not this GPU):** `/imagine` `/video` `/i2v` via Pollinations."
    )
    image = gr.Image(type="pil", label="Image")

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

    with gr.Tab("Imagine"):
        pr = gr.Textbox(label="Prompt", placeholder="a tea stall in Rangpur rain, cinematic")
        im_out = gr.Image(type="pil", label="Image")
        im_txt = gr.Textbox(label="Status")
        im_btn = gr.Button("Generate image")
        im_btn.click(imagine, inputs=pr, outputs=[im_out, im_txt], api_name="imagine")

    with gr.Tab("Video"):
        vpr = gr.Textbox(label="Prompt", placeholder="drone shot over a river at sunset")
        v_out = gr.Video(label="Video")
        v_txt = gr.Textbox(label="Status")
        v_btn = gr.Button("Generate video")
        v_btn.click(make_video, inputs=vpr, outputs=[v_out, v_txt], api_name="video")

    with gr.Tab("Photo to video"):
        iv_pr = gr.Textbox(label="Motion prompt", placeholder="slow zoom in, cinematic")
        iv_out = gr.Video(label="Video")
        iv_txt = gr.Textbox(label="Status")
        iv_btn = gr.Button("Animate photo")
        iv_btn.click(image_to_video, inputs=[image, iv_pr], outputs=[iv_out, iv_txt], api_name="i2v")


demo.queue()
if __name__ == "__main__":
    demo.launch()
