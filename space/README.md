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

Chat: Llama-3.2-3B-Instruct if Space secret `HF_TOKEN` + license are set; otherwise ungated `Qwen/Qwen2.5-1.5B-Instruct`. First chat can take a minute (model download).

Mutual keep-alive (`requests`, timeout 30s, daemon thread, parallel GET):
- Render → Hugging Face: every **10 hours** — default `https://madarauchihagmailcom-my.hf.space/`
- Hugging Face → Render: every **2 minutes** — default `https://rend-y1aw.onrender.com/`

More than 2 URLs at the same time — comma-separated env `PING_URLS` on either side, e.g.
`PING_URLS=https://rend-y1aw.onrender.com/,https://other.onrender.com/,https://another-space.hf.space/`

If the Space itself sleeps, its 2-minute pings stop. A Telegram message still wakes Render.
