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

Mutual keep-alive (every 10 hours, `requests`, timeout 30s, daemon thread):
- Render → Space: `https://madarauchihagmailcom-my.hf.space/`
- Space → Render: `https://rend-y1aw.onrender.com/` (override with Space Variable `RENDER_PING_URL`)

If either side is already asleep, that side cannot ping. A Telegram message still wakes Render.
