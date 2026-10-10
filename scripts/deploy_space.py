#!/usr/bin/env python3
"""Publish the app as a Docker Space on Hugging Face and set its secrets.

Usage: uv run --env-file .env python scripts/deploy_space.py [--space fiap-ssp-2025/guardia-ai] [--private]
Reads HF_TOKEN, OPENROUTER_API_KEY and the LLM_*/JUDGE_*/FAST_* values from the environment.
The Space README (front-matter) comes from deploy/space/README.md; data/raw, data/generated, .env never go up.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
IGNORE = ["data/raw/*", "data/generated/*", "data/processed/*.parquet", ".venv/*", ".git/*", ".env", "__pycache__/*", "*.pyc",
          "notebooks/*.ipynb", "logs/*", ".pytest_cache/*", ".ruff_cache/*", "README.md", "*.pdf", ".claude/*", ".github/*"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default=f"{os.getenv('HF_ORG', 'fiap-ssp-2025')}/guardia-ai")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--no-wait", action="store_true")
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN") or sys.exit("HF_TOKEN ausente")
    api = HfApi(token=token)
    api.create_repo(args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    secrets = {
        "LLM_BASE_URL": os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"), "LLM_MODEL": os.environ["LLM_MODEL"],
        "LLM_API_KEY": os.environ["LLM_API_KEY"],
        "JUDGE_BASE_URL": os.getenv("JUDGE_BASE_URL", "https://openrouter.ai/api/v1"), "JUDGE_MODEL": os.environ["JUDGE_MODEL"],
        "JUDGE_API_KEY": os.environ["JUDGE_API_KEY"],
        "FAST_BASE_URL": os.getenv("FAST_BASE_URL", "https://openrouter.ai/api/v1"), "FAST_MODEL": os.getenv("FAST_MODEL", os.environ["JUDGE_MODEL"]),
        "FAST_API_KEY": os.getenv("FAST_API_KEY", os.environ["JUDGE_API_KEY"]),
    }
    for k in ("LANGSMITH_API_KEY", "LANGSMITH_ENDPOINT"):
        if os.getenv(k):
            secrets[k] = os.environ[k]
    for k, v in secrets.items():
        api.add_space_secret(args.space, k, v)
    for k, v in {"PORT": "7860", "WHISPER_MODEL": os.getenv("SPACE_WHISPER_MODEL", "base"), "LANGCHAIN_TRACING_V2": os.getenv("LANGCHAIN_TRACING_V2", "false"),
                 "LANGCHAIN_PROJECT": "guardia-ai-space", "APP_PUBLIC_URL": f"https://huggingface.co/spaces/{args.space}",
                 "HF_HOME": "/app/data/generated/hf", "TOKENIZERS_PARALLELISM": "false"}.items():
        api.add_space_variable(args.space, k, v)
    print("segredos e variáveis definidos")
    api.upload_folder(repo_id=args.space, repo_type="space", folder_path=str(ROOT), path_in_repo=".", ignore_patterns=IGNORE,
                      commit_message="Guardiã AI: app, modelo, protocolos e Dockerfile")
    api.upload_file(repo_id=args.space, repo_type="space", path_or_fileobj=str(ROOT / "deploy" / "space" / "README.md"), path_in_repo="README.md",
                    commit_message="README do Space (front-matter docker, porta 7860)")
    url = f"https://huggingface.co/spaces/{args.space}"
    print("enviado:", url)
    if args.no_wait:
        return
    t0 = time.time()
    while time.time() - t0 < 2400:
        rt = api.get_space_runtime(args.space)
        print(f"[{int(time.time() - t0):4d}s] stage={rt.stage} hardware={rt.hardware}", flush=True)
        if rt.stage in ("RUNNING", "RUNNING_APP_STARTING"):
            break
        if rt.stage in ("BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR"):
            sys.exit(f"falha no Space: {rt.stage}")
        time.sleep(30)
    print("app:", f"https://{args.space.replace('/', '-')}.hf.space")


if __name__ == "__main__":
    main()
