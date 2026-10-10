"""OpenAI-compatible LLM client with four roles.

generator  → LLM_BASE_URL / LLM_MODEL (OpenRouter or vLLM on RunPod); optional fallback to OpenAI (OPENAI_API_KEY)
judge      → JUDGE_BASE_URL / JUDGE_MODEL (a different model/server, so the judge never audits itself)
fast       → FAST_* (live extraction while the user types; defaults to the judge server)
transcribe → TRANSCRIBE_* (audio-capable model for dictation; defaults to google/gemini-2.5-flash on OpenRouter)

Structured output: when a JSON schema is given we ask the server for ``response_format=json_schema``
(vLLM applies constrained decoding; OpenAI applies structured outputs). If the server rejects it we
retry with ``json_object`` and validate the parsed result ourselves.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

try:  # LangSmith tracing is a no-op unless LANGCHAIN_TRACING_V2=true
    from langsmith import traceable
except ImportError:  # pragma: no cover
    def traceable(*_a, **_k):
        def deco(fn):
            return fn
        return deco

UA = {"User-Agent": "guardia-ai/1.0",
      # OpenRouter uses these for attribution; other OpenAI-compatible servers ignore them
      "HTTP-Referer": os.getenv("APP_PUBLIC_URL", "https://github.com/fiap-ssp-2025/tech-challenge-5"), "X-Title": "Guardia AI"}


@dataclass
class Endpoint:
    role: str
    base_url: str | None
    model: str | None
    api_key: str | None

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.model)


def _key(*names: str, base_url: str | None) -> str:
    """First non-empty key among `names`; `${VAR}` placeholders are resolved; OpenRouter/HF endpoints fall back to their own keys."""
    for n in names:
        v = (os.getenv(n) or "").strip()
        if v.startswith("${") and v.endswith("}"):
            v = (os.getenv(v[2:-1]) or "").strip()
        if v:
            return v
    url = base_url or ""
    if "openrouter.ai" in url and os.getenv("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    if "huggingface.co" in url and os.getenv("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    return "none"


def endpoint(role: str) -> Endpoint:
    if role == "transcribe":  # audio-capable model for the dictation fallback; defaults to the fast/judge server (OpenRouter)
        url = os.getenv("TRANSCRIBE_BASE_URL") or os.getenv("FAST_BASE_URL") or os.getenv("JUDGE_BASE_URL")
        model = os.getenv("TRANSCRIBE_MODEL") or ("google/gemini-2.5-flash" if "openrouter.ai" in (url or "") else None)
        return Endpoint("transcribe", url, model, _key("TRANSCRIBE_API_KEY", "FAST_API_KEY", "JUDGE_API_KEY", base_url=url))
    if role == "fast":  # small model for live, low-latency extraction while the user types; defaults to the judge server
        url = os.getenv("FAST_BASE_URL") or os.getenv("JUDGE_BASE_URL")
        return Endpoint("fast", url, os.getenv("FAST_MODEL") or os.getenv("JUDGE_MODEL"), _key("FAST_API_KEY", "JUDGE_API_KEY", base_url=url))
    if role == "judge":
        url = os.getenv("JUDGE_BASE_URL")
        return Endpoint("judge", url, os.getenv("JUDGE_MODEL"), _key("JUDGE_API_KEY", base_url=url))
    url = os.getenv("LLM_BASE_URL")
    return Endpoint("generator", url, os.getenv("LLM_MODEL"), _key("LLM_API_KEY", base_url=url))


def fallback_endpoint() -> Endpoint | None:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    return Endpoint("fallback", "https://api.openai.com/v1", os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4o-mini"), key)


@lru_cache(maxsize=8)
def _client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key, default_headers=UA, timeout=180, max_retries=1)


@dataclass
class LLMResult:
    content: str
    backend: str          # generator | fallback | judge | error
    model: str
    latency_ms: int
    tokens: int = 0
    parsed: dict[str, Any] | None = None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _clean_error(msg: str, limit: int = 160) -> str:
    """Proxies (RunPod while booting, for example) answer with whole HTML pages; keep the gist only."""
    msg = re.sub(r"<[^>]+>", " ", msg)
    msg = re.sub(r"\s+", " ", msg).strip()
    return msg[:limit]


def _extract_json(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[4:] if text.lower().startswith("json") else text
    try:
        return json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except ValueError:
                return None
    return None


def _call(ep: Endpoint, messages: list[dict], *, schema: dict | None, schema_name: str, temperature: float,
          max_tokens: int, disable_thinking: bool) -> LLMResult:
    client = _client(ep.base_url, ep.api_key)
    extra: dict[str, Any] = {}
    if disable_thinking:
        extra["chat_template_kwargs"] = {"enable_thinking": False}  # vLLM / Qwen3 chat template
    if "openrouter.ai" in (ep.base_url or ""):
        extra["provider"] = {"sort": "latency"}                      # route to the fastest provider available
        if disable_thinking:
            extra["reasoning"] = {"enabled": False}                  # OpenRouter's switch for thinking models
    kwargs: dict[str, Any] = {"model": ep.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if extra:
        kwargs["extra_body"] = extra
    t0 = time.monotonic()
    attempts: list[dict[str, Any]] = []
    if schema:
        attempts.append({"response_format": {"type": "json_schema", "json_schema": {"name": schema_name, "schema": schema, "strict": True}}})
        attempts.append({"response_format": {"type": "json_object"}})
    else:
        attempts.append({})
    last_err = ""
    for fmt in attempts:
        try:
            resp = client.chat.completions.create(**kwargs, **fmt)
            content = (resp.choices[0].message.content or "").strip()
            parsed = _extract_json(content) if schema else None
            if schema and parsed is None:
                last_err = "resposta não era JSON válido"
                continue
            return LLMResult(
                content=content, backend=ep.role, model=ep.model or "", latency_ms=int((time.monotonic() - t0) * 1000),
                tokens=(resp.usage.total_tokens if resp.usage else 0), parsed=parsed,
                meta={"response_format": fmt.get("response_format", {}).get("type", "text")},
            )
        except Exception as e:  # noqa: BLE001 — any transport/API error → try the next format / backend
            last_err = f"{type(e).__name__}: {_clean_error(str(e))}"
            continue
    return LLMResult(content="", backend="error", model=ep.model or "", latency_ms=int((time.monotonic() - t0) * 1000), error=last_err)


@traceable(name="llm.chat", run_type="llm")
def chat(messages: list[dict], *, role: str = "generator", schema: dict | None = None, schema_name: str = "resposta",
         temperature: float = 0.2, max_tokens: int = 1200, allow_fallback: bool = True, disable_thinking: bool | None = None) -> LLMResult:
    """Send a chat to the role's endpoint. Generator falls back to OpenAI when configured.
    ``disable_thinking`` defaults to True for the generator and the transcriber (thinking only adds latency there)."""
    ep = endpoint(role)
    errors: list[str] = []
    if disable_thinking is None:
        disable_thinking = role in ("generator", "transcribe")
    if ep.configured:
        res = _call(ep, messages, schema=schema, schema_name=schema_name, temperature=temperature, max_tokens=max_tokens,
                    disable_thinking=disable_thinking)
        if res.backend != "error":
            return res
        errors.append(f"{ep.role}: {res.error}")
    else:
        errors.append(f"{ep.role}: não configurado (LLM_BASE_URL/LLM_MODEL)")
    if role == "generator" and allow_fallback and (fb := fallback_endpoint()):
        res = _call(fb, messages, schema=schema, schema_name=schema_name, temperature=temperature, max_tokens=max_tokens, disable_thinking=False)
        if res.backend != "error":
            return res
        errors.append(f"fallback: {res.error}")
    return LLMResult(content="", backend="error", model=ep.model or "", latency_ms=0, error=" | ".join(errors))


def probe(role: str = "generator", timeout: float = 6.0) -> dict[str, Any]:
    ep = endpoint(role)
    if not ep.configured:
        return {"role": role, "ok": False, "backend": "none", "model": None, "endpoint": None, "note": "não configurado"}
    try:
        client = OpenAI(base_url=ep.base_url, api_key=ep.api_key, default_headers=UA, timeout=timeout, max_retries=0)
        ids = [m.id for m in client.models.list().data]
        return {"role": role, "ok": True, "backend": role, "model": ep.model, "endpoint": ep.base_url, "served": ids}
    except Exception as e:  # noqa: BLE001
        note = f"{type(e).__name__}"
        fb = fallback_endpoint() if role == "generator" else None
        return {"role": role, "ok": bool(fb), "backend": "fallback" if fb else "none", "model": fb.model if fb else None,
                "endpoint": ep.base_url, "note": f"endpoint indisponível ({note})" + (" → fallback OpenAI" if fb else "")}
