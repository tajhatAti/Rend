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


with gr.Blocks(title="Image Bot Space") as demo:
    gr.Markdown(
        "# Image Bot Space\n"
        "Telegram backend. **FLUX.1-schnell** on this ZeroGPU for `/imagine`. "
        "Photo tools: `/caption` `/ocr` `/detect` `/bg` `/sketch`. "
        "Lyrics: `/lyrics` via lrclib."
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


demo.queue()
if __name__ == "__main__":
    demo.launch()
