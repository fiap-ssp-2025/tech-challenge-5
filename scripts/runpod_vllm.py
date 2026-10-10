#!/usr/bin/env python3
"""Create / inspect / stop the vLLM pod on RunPod that serves the Guardiã AI models.

One pod, one GPU (A40 48 GB by default), two vLLM servers:
  :8000 → generator  (LLM_MODEL,   default QuantTrio/Qwen3.6-27B-AWQ)
  :8001 → judge      (JUDGE_MODEL, default Qwen/Qwen2.5-7B-Instruct-AWQ)

Both expose the OpenAI-compatible API behind RunPod's HTTP proxy:
  https://<pod-id>-8000.proxy.runpod.net/v1   and   https://<pod-id>-8001.proxy.runpod.net/v1

Usage:
  uv run python scripts/runpod_vllm.py create            # creates and prints the URLs to put in .env
  uv run python scripts/runpod_vllm.py status            # desiredStatus + readiness probe of both servers
  uv run python scripts/runpod_vllm.py wait              # blocks until both /v1/models answer (model download ~10 min)
  uv run python scripts/runpod_vllm.py stop | start | delete
  uv run python scripts/runpod_vllm.py balance

Reads RUNPOD_API_KEY, HF_TOKEN, LLM_API_KEY, LLM_MODEL, JUDGE_MODEL, RUNPOD_GPU_TYPE from the environment / .env.
The pod id is cached in data/generated/runpod_pod.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

API = "https://rest.runpod.io/v1"
GQL = "https://api.runpod.io/graphql"
STATE_FILE = ROOT / "data" / "generated" / "runpod_pod.json"

GEN_MODEL = os.getenv("LLM_MODEL", "QuantTrio/Qwen3.6-27B-AWQ")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
GPU_TYPES = [g.strip() for g in os.getenv(
    "RUNPOD_GPU_TYPES", "NVIDIA A40,NVIDIA RTX A6000,NVIDIA L40S,NVIDIA RTX 6000 Ada Generation,NVIDIA L40"
).split(",") if g.strip()]
API_KEY_FOR_VLLM = os.getenv("LLM_API_KEY", "guardia-dev-key")
UA = {"User-Agent": "guardia-ai/1.0"}  # RunPod's proxy rejects the default urllib agent


def _headers() -> dict[str, str]:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        sys.exit("RUNPOD_API_KEY ausente (.env)")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json", **UA}


def _start_cmd() -> list[str]:
    """Two vLLM servers sharing the GPU (48 GB): judge 18% (7B AWQ ≈ 5 GB + KV), generator 76% (27B AWQ ≈ 20.6 GB + KV).
    First attempt OOMed with 16k ctx / 0.62: vLLM asked 12 GB extra for KV + CUDA graphs. 8k ctx, 8 seqs and eager mode fit."""
    gen = (
        f"python3 -m vllm.entrypoints.openai.api_server --model {GEN_MODEL} --port 8000 --host 0.0.0.0 "
        f"--api-key {API_KEY_FOR_VLLM} --max-model-len 8192 --max-num-seqs 8 --gpu-memory-utilization 0.76 "
        f"--served-model-name {GEN_MODEL} generator --download-dir /workspace/hf"
    )
    judge = (
        f"python3 -m vllm.entrypoints.openai.api_server --model {JUDGE_MODEL} --port 8001 --host 0.0.0.0 "
        f"--api-key {API_KEY_FOR_VLLM} --max-model-len 6144 --max-num-seqs 8 --gpu-memory-utilization 0.18 "
        f"--served-model-name {JUDGE_MODEL} judge --download-dir /workspace/hf"
    )
    # The judge starts first (small, fast); the generator follows once the judge is listening so the two
    # processes do not race for the same free memory during profiling.
    script = (
        f"({judge}) & J=$!; "
        "for i in $(seq 1 120); do curl -sf -o /dev/null http://127.0.0.1:8001/health && break; sleep 5; done; "
        f"({gen}) & G=$!; wait $J $G"
    )
    return [script]


def create(args: argparse.Namespace) -> None:
    cloud = args.cloud
    body = {
        "name": "guardia-ai-vllm",
        "imageName": "vllm/vllm-openai:latest",
        "gpuTypeIds": GPU_TYPES,
        "gpuTypePriority": "custom",
        "gpuCount": 1,
        "cloudType": cloud,
        "containerDiskInGb": 40,
        "volumeInGb": 60,
        "volumeMountPath": "/workspace",
        "ports": ["8000/http", "8001/http"],
        "env": {"HF_TOKEN": os.environ.get("HF_TOKEN", ""), "HF_HUB_ENABLE_HF_TRANSFER": "0"},
        # the vllm-openai image's ENTRYPOINT is the api_server itself; replace it with a shell so two servers can run
        "dockerEntrypoint": ["bash", "-lc"],
        "dockerStartCmd": _start_cmd(),
        "interruptible": False,
        "minRAMPerGPU": 16,
        "minVCPUPerGPU": 4,
    }
    pod = None
    for cloud in ([args.cloud] if args.cloud != "ANY" else ["COMMUNITY", "SECURE"]):
        body["cloudType"] = cloud
        r = requests.post(f"{API}/pods", headers=_headers(), json=body, timeout=60)
        if r.status_code < 300:
            pod = r.json()
            print(f"pod criado em {cloud}")
            break
        print(f"[{cloud}] {r.status_code} {r.text[:300]}")
    if pod is None:
        sys.exit("Falha ao criar pod em todas as opções.")
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps({"id": pod["id"], "created": time.time(), "gpu": pod.get("gpu") or GPU_TYPES}, indent=2, default=str))
    print(json.dumps({k: pod.get(k) for k in ("id", "name", "desiredStatus", "machineId", "costPerHr")}, indent=2))
    print_env(pod["id"])


def print_env(pod_id: str) -> None:
    print("\nColoque no .env:")
    print(f"LLM_BASE_URL=https://{pod_id}-8000.proxy.runpod.net/v1")
    print(f"LLM_MODEL={GEN_MODEL}")
    print(f"LLM_API_KEY={API_KEY_FOR_VLLM}")
    print(f"JUDGE_BASE_URL=https://{pod_id}-8001.proxy.runpod.net/v1")
    print(f"JUDGE_MODEL={JUDGE_MODEL}")
    print(f"JUDGE_API_KEY={API_KEY_FOR_VLLM}")


def _pod_id(args: argparse.Namespace) -> str:
    if getattr(args, "pod_id", None):
        return args.pod_id
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())["id"]
    sys.exit("Nenhum pod conhecido: passe --pod-id ou rode `create`.")


def get(pod_id: str) -> dict:
    r = requests.get(f"{API}/pods/{pod_id}", headers=_headers(), timeout=30)
    r.raise_for_status()
    return r.json()


def probe(pod_id: str, port: int, timeout: float = 8.0) -> tuple[bool, str]:
    url = f"https://{pod_id}-{port}.proxy.runpod.net/v1/models"
    try:
        r = requests.get(url, headers={"Authorization": f"Bearer {API_KEY_FOR_VLLM}", **UA}, timeout=timeout)
        if r.status_code == 200:
            ids = [m.get("id") for m in r.json().get("data", [])]
            return True, ",".join(ids)
        return False, f"HTTP {r.status_code}"
    except requests.RequestException as e:
        return False, type(e).__name__


def status(args: argparse.Namespace) -> None:
    pod_id = _pod_id(args)
    pod = get(pod_id)
    print(json.dumps({k: pod.get(k) for k in ("id", "desiredStatus", "costPerHr", "gpu", "lastStatusChange")}, indent=2, default=str))
    for port, label in ((8001, "judge"), (8000, "generator")):
        ok, info = probe(pod_id, port)
        print(f"  {label:<9} :{port} → {'READY' if ok else 'not ready'} ({info})")


def wait(args: argparse.Namespace) -> None:
    pod_id = _pod_id(args)
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        ok_j, _ = probe(pod_id, 8001)
        ok_g, _ = probe(pod_id, 8000)
        print(f"[{time.strftime('%H:%M:%S')}] judge={'ok' if ok_j else '..'} generator={'ok' if ok_g else '..'}", flush=True)
        if ok_j and ok_g:
            print_env(pod_id)
            return
        time.sleep(args.interval)
    sys.exit("Timeout esperando os servidores vLLM.")


def stop(args: argparse.Namespace) -> None:
    pod_id = _pod_id(args)
    r = requests.post(f"{API}/pods/{pod_id}/stop", headers=_headers(), timeout=30)
    print(r.status_code, r.text[:300])


def start(args: argparse.Namespace) -> None:
    pod_id = _pod_id(args)
    r = requests.post(f"{API}/pods/{pod_id}/start", headers=_headers(), timeout=60)
    print(r.status_code, r.text[:300])


def delete(args: argparse.Namespace) -> None:
    pod_id = _pod_id(args)
    r = requests.delete(f"{API}/pods/{pod_id}", headers=_headers(), timeout=30)
    print(r.status_code, r.text[:300])
    if r.status_code < 300 and STATE_FILE.exists():
        STATE_FILE.unlink()


def update(args: argparse.Namespace) -> None:
    """Rewrite the start command on the existing pod and restart it (the /workspace volume keeps the downloads)."""
    pod_id = _pod_id(args)
    body = {"dockerEntrypoint": ["bash", "-lc"], "dockerStartCmd": _start_cmd(), "ports": ["8000/http", "8001/http"]}
    r = requests.patch(f"{API}/pods/{pod_id}", headers=_headers(), json=body, timeout=60)
    print("patch:", r.status_code, r.text[:300])
    if r.status_code >= 300:
        return
    r = requests.post(f"{API}/pods/{pod_id}/stop", headers=_headers(), timeout=60); print("stop:", r.status_code)
    time.sleep(8)
    r = requests.post(f"{API}/pods/{pod_id}/start", headers=_headers(), timeout=120); print("start:", r.status_code, r.text[:200])


def balance(_: argparse.Namespace) -> None:
    key = os.environ["RUNPOD_API_KEY"]
    q = {"query": "{ myself { clientBalance currentSpendPerHr } }"}
    r = requests.post(f"{GQL}?api_key={key}", json=q, headers=UA, timeout=30)
    print(r.json())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create"); c.add_argument("--cloud", default="ANY", choices=["ANY", "COMMUNITY", "SECURE"]); c.set_defaults(fn=create)
    for name, fn in (("status", status), ("stop", stop), ("start", start), ("delete", delete), ("update", update)):
        s = sub.add_parser(name); s.add_argument("--pod-id"); s.set_defaults(fn=fn)
    w = sub.add_parser("wait"); w.add_argument("--pod-id"); w.add_argument("--timeout", type=int, default=1800); w.add_argument("--interval", type=int, default=20); w.set_defaults(fn=wait)
    b = sub.add_parser("balance"); b.set_defaults(fn=balance)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
