"""
Image API that runs on YOUR Hugging Face Space.

Named endpoints the Telegram bot calls:
  /caption  /ocr  /detect  /bg

Models load on first use so the Space can boot on free CPU.
"""

from __future__ import annotations

import io
from functools import lru_cache

import gradio as gr
from PIL import Image, ImageDraw, ImageFont

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


with gr.Blocks(title="Image Bot Space") as demo:
    gr.Markdown(
        "# Image Bot Space\n"
        "This Space is the backend for the Telegram bot on Render. "
        "Use the tabs here to test, or call `/caption` `/ocr` `/detect` `/bg` via API."
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


if __name__ == "__main__":
    demo.queue().launch()
