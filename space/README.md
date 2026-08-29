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

Backend for the Telegram image bot.

On this Space (ZeroGPU): `/caption` `/ocr` `/detect` `/bg` `/sketch`

Online APIs via Pollinations: `/imagine` `/video` `/i2v`

## POLLINATIONS_KEY — Secret, not Variable

On Hugging Face → your Space `madarauchihagmailcom/My` → **Settings** → **Variables and secrets**:

1. Click **New secret** (not New variable).
2. Name exactly: `POLLINATIONS_KEY`
3. Value: the `sk_...` key from https://enter.pollinations.ai/keys
4. Save, then **Factory reboot** the Space.

Secrets are encrypted and only visible to the app as an environment variable.
Variables are public — never put the key there.

Do **not** put this key on Render. Render only talks to this Space.
The Space is what calls Pollinations.
