---
title: Image Bot
emoji: 🖼️
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

Backend for the Telegram bot.

On this ZeroGPU Space:
- `/chat` — Llama-3.2-3B-Instruct (fast GPU chat)
- `/imagine` — FLUX.1-schnell (text → image)
- `/caption` `/ocr` `/detect` `/bg` `/sketch`
- `/lyrics` — song name → lrclib.net
- `/gitzip` — GitHub `owner/repo` → ZIP (max ~40 MB)

Gated Llama needs Space secret `HF_TOKEN` (accept the model license on Hugging Face).

Keep Render awake: Space **Variable** (not Secret) `RENDER_PING_URL` = `https://YOUR-APP.onrender.com/`
The Space GETs that URL every 5 minutes. If the Space itself sleeps, pings stop.
