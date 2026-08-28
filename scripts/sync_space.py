#!/usr/bin/env python3
"""Upload ./space to a Hugging Face Space. Used by GitHub Actions and locally."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parents[1]
SPACE_DIR = ROOT / "space"
REPO_ID = (os.environ.get("HF_SPACE_ID") or "madarauchihagmailcom/My").strip()
TOKEN = (os.environ.get("HF_TOKEN") or os.environ.get("HF_API_TOKEN") or "").strip()
MESSAGE = os.environ.get("SYNC_MESSAGE") or "sync Space from GitHub"


def main() -> int:
    if not TOKEN:
        print("Set HF_TOKEN (Hugging Face write token) before syncing.", file=sys.stderr)
        return 1
    if not SPACE_DIR.is_dir():
        print(f"Missing folder: {SPACE_DIR}", file=sys.stderr)
        return 1

    api = HfApi(token=TOKEN)
    create_repo(
        REPO_ID,
        repo_type="space",
        exist_ok=True,
        token=TOKEN,
        space_sdk="gradio",
        private=False,
    )
    api.upload_folder(
        folder_path=str(SPACE_DIR),
        repo_id=REPO_ID,
        repo_type="space",
        commit_message=MESSAGE,
        allow_patterns=["*.py", "*.txt", "*.md", "*.json", "*.yaml", "*.yml"],
        ignore_patterns=["**/__pycache__/**", "**/*.pyc"],
        delete_patterns=["**/*.py", "**/*.txt", "**/*.md"],
    )
    print(f"Synced {SPACE_DIR} -> https://huggingface.co/spaces/{REPO_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
