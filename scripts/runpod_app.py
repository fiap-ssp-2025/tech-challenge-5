#!/usr/bin/env python3
"""Run the Guardiã AI app on a RunPod CPU pod (public URL via RunPod's HTTP proxy).

Two ways to get the code onto the pod:
  --from-git (default): a public base image with uv clones the public GitHub repo on a persistent volume,
      installs the locked dependencies, builds the RAG index and starts uvicorn. No registry, no secrets besides
      the LLM keys; restarts are fast because the volume keeps the venv and the model caches.
  --image: pulls ghcr.io/fiap-ssp-2025/guardia-ai:latest (private → registry credential saved in RunPod from the
      GitHub token, which needs the write:packages/read:packages scopes).
The LLM is remote (OpenRouter by default), so the pod only needs CPU + RAM for the web app, the referral model,
the RAG index and the whisper fallback.

Usage:
  uv run --env-file .env python scripts/runpod_app.py create [--flavor cpu3c] [--vcpu 2] [--image]
  uv run --env-file .env python scripts/runpod_app.py status | wait | stop | start | delete | balance
Reads RUNPOD_API_KEY, GHCR_USER/GHCR_TOKEN (or `gh auth token`) and the LLM_*/JUDGE_*/FAST_*/OPENROUTER_API_KEY
values to inject as environment variables. The pod id is cached in data/generated/runpod_app.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
API = "https://rest.runpod.io/v1"
GQL = "https://api.runpod.io/graphql"
STATE_FILE = ROOT / "data" / "generated" / "runpod_app.json"
IMAGE = os.getenv("APP_IMAGE", "ghcr.io/fiap-ssp-2025/guardia-ai:latest")
UA = {"User-Agent": "guardia-ai/1.0"}


def _headers() -> dict[str, str]:
    key = os.environ.get("RUNPOD_API_KEY") or sys.exit("RUNPOD_API_KEY ausente")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json", **UA}


def _gql(query: str, variables: dict | None = None) -> dict:
    r = requests.post(f"{GQL}?api_key={os.environ['RUNPOD_API_KEY']}", json={"query": query, "variables": variables or {}}, headers=UA, timeout=60)
    r.raise_for_status()
    data = r.json()
    if data.get("errors"):
        sys.exit(f"GraphQL: {data['errors']}")
    return data["data"]


def _registry_auth_id() -> str:
    """Save (once) a container-registry credential for GHCR and return its id."""
    user = os.getenv("GHCR_USER") or subprocess.run(["gh", "api", "user", "--jq", ".login"], check=False, capture_output=True, text=True).stdout.strip()
    token = os.getenv("GHCR_TOKEN") or subprocess.run(["gh", "auth", "token"], check=False, capture_output=True, text=True).stdout.strip()
    if not (user and token):
        sys.exit("Credencial do GHCR ausente: defina GHCR_USER/GHCR_TOKEN ou faça `gh auth login`.")
    existing = _gql("{ myself { containerRegistryCreds { id name } } }")["myself"].get("containerRegistryCreds") or []
    for c in existing:
        if c["name"] == "ghcr-guardia":
            return c["id"]
    out = _gql('mutation($i: SaveRegistryAuthInput) { saveRegistryAuth(input: $i) { id name } }',
               {"i": {"name": "ghcr-guardia", "username": user, "password": token}})
    return out["saveRegistryAuth"]["id"]


def _app_env() -> dict[str, str]:
    env = {k: os.environ[k] for k in ("LLM_BASE_URL", "LLM_MODEL", "JUDGE_BASE_URL", "JUDGE_MODEL", "FAST_BASE_URL", "FAST_MODEL",
                                      "TRANSCRIBE_BASE_URL", "TRANSCRIBE_MODEL", "OPENROUTER_API_KEY", "LANGSMITH_API_KEY", "LANGSMITH_ENDPOINT") if os.environ.get(k)}
    key = os.environ.get("OPENROUTER_API_KEY", "")
    for k in ("LLM_API_KEY", "JUDGE_API_KEY", "FAST_API_KEY"):
        v = os.environ.get(k, "")
        env[k] = key if (not v or v.startswith("${")) else v
    env.update({"PORT": "8000", "WHISPER_MODEL": os.getenv("APP_WHISPER_MODEL", "base"), "LANGCHAIN_TRACING_V2": os.getenv("LANGCHAIN_TRACING_V2", "false"),
                "LANGCHAIN_PROJECT": "guardia-ai-runpod", "TOKENIZERS_PARALLELISM": "false"})
    return env


REPO = os.getenv("APP_REPO", "https://github.com/fiap-ssp-2025/tech-challenge-5")
BASE_IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm-slim"
BOOTSTRAP = f"""set -e; export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq --no-install-recommends git ffmpeg curl >/dev/null
mkdir -p /workspace && cd /workspace
if [ -d app/.git ]; then cd app && git fetch -q --depth 1 origin main && git reset -q --hard origin/main; else git clone -q --depth 1 {REPO} app && cd app; fi
export UV_LINK_MODE=copy UV_CACHE_DIR=/workspace/uv-cache HF_HOME=/workspace/hf TOKENIZERS_PARALLELISM=false
uv sync --frozen --no-dev --group audio
uv run --no-sync python -c "from guardia.rag.retriever import vectorstore; print('chunks', vectorstore()._collection.count())"
exec uv run --no-sync uvicorn guardia.web.app:app --host 0.0.0.0 --port 8000"""


def create(args: argparse.Namespace) -> None:
    body = {
        "name": "guardia-ai-app",
        "computeType": "CPU",
        "cpuFlavorIds": [args.flavor],
        "vcpuCount": args.vcpu,
        "cloudType": "SECURE",
        "containerDiskInGb": 20,
        "ports": ["8000/http"],
        "env": _app_env(),
    }
    if args.image:
        body.update({"imageName": IMAGE, "containerRegistryAuthId": _registry_auth_id(), "volumeInGb": 0})
    else:
        body.update({"imageName": BASE_IMAGE, "dockerEntrypoint": ["bash", "-lc"], "dockerStartCmd": [BOOTSTRAP],
                     "volumeInGb": 20, "volumeMountPath": "/workspace"})
    r = requests.post(f"{API}/pods", headers=_headers(), json=body, timeout=90)
    if r.status_code >= 300:
        sys.exit(f"Falha ao criar pod: {r.status_code} {r.text[:600]}")
    pod = r.json()
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps({"id": pod["id"], "created": time.time()}, indent=2))
    print(json.dumps({k: pod.get(k) for k in ("id", "name", "desiredStatus", "costPerHr")}, indent=2))
    print("URL pública:", _url(pod["id"]))


def _url(pod_id: str) -> str:
    return f"https://{pod_id}-8000.proxy.runpod.net"


def _pod_id(args: argparse.Namespace) -> str:
    if getattr(args, "pod_id", None):
        return args.pod_id
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())["id"]
    sys.exit("Nenhum pod conhecido: passe --pod-id ou rode `create`.")


def _ready(pod_id: str, timeout: float = 8.0) -> tuple[bool, str]:
    try:
        r = requests.get(f"{_url(pod_id)}/api/health", headers=UA, timeout=timeout)
        return r.status_code == 200, f"HTTP {r.status_code}"
    except requests.RequestException as e:
        return False, type(e).__name__


def status(args: argparse.Namespace) -> None:
    pod_id = _pod_id(args)
    r = requests.get(f"{API}/pods/{pod_id}", headers=_headers(), timeout=30)
    pod = r.json() if r.status_code == 200 else {}
    print(json.dumps({k: pod.get(k) for k in ("id", "desiredStatus", "costPerHr", "lastStatusChange")}, indent=2, default=str))
    ok, info = _ready(pod_id)
    print(f"  app {_url(pod_id)} → {'READY' if ok else 'not ready'} ({info})")


def wait(args: argparse.Namespace) -> None:
    pod_id = _pod_id(args)
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        ok, info = _ready(pod_id)
        print(f"[{time.strftime('%H:%M:%S')}] {'READY' if ok else info}", flush=True)
        if ok:
            print("URL pública:", _url(pod_id))
            return
        time.sleep(args.interval)
    sys.exit("Timeout esperando o app.")


def simple(action: str):
    def fn(args: argparse.Namespace) -> None:
        pod_id = _pod_id(args)
        if action == "delete":
            r = requests.delete(f"{API}/pods/{pod_id}", headers=_headers(), timeout=30)
            if r.status_code < 300 and STATE_FILE.exists():
                STATE_FILE.unlink()
        else:
            r = requests.post(f"{API}/pods/{pod_id}/{action}", headers=_headers(), timeout=90)
        print(action, r.status_code, r.text[:200])
    return fn


def balance(_: argparse.Namespace) -> None:
    print(_gql("{ myself { clientBalance currentSpendPerHr } }"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create"); c.add_argument("--flavor", default="cpu3c"); c.add_argument("--vcpu", type=int, default=2)
    c.add_argument("--image", action="store_true", help="usar a imagem do GHCR em vez de clonar o repositório"); c.set_defaults(fn=create)
    for name in ("status", "stop", "start", "delete"):
        s = sub.add_parser(name); s.add_argument("--pod-id"); s.set_defaults(fn=status if name == "status" else simple(name))
    w = sub.add_parser("wait"); w.add_argument("--pod-id"); w.add_argument("--timeout", type=int, default=1200); w.add_argument("--interval", type=int, default=15); w.set_defaults(fn=wait)
    b = sub.add_parser("balance"); b.set_defaults(fn=balance)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
