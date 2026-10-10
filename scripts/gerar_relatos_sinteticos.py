#!/usr/bin/env python3
"""Generate synthetic reports from real SINAN rows (labels come for free from the fields).

For each sampled notification the LLM writes a short first-person-free report, as a professional
would summarise it, consistent with the structured fields. Output: data/synthetic/relatos.jsonl with
{"id", "case", "target", "relato", "model"}. Used for demos, the F2 evaluation and the judge batch.

Usage: uv run python scripts/gerar_relatos_sinteticos.py --n 60 [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from guardia.data.features import AG, CATEGORICAL, FEATURE_NAMES, PROCESSED_PATH, REL_GROUPS, TARGET, VIOL
from guardia.llm.client import chat
from guardia.web.forms import label

OUT = ROOT / "data" / "synthetic" / "relatos.jsonl"

SYSTEM = (
    "Você escreve, em português do Brasil, o relato resumido de um atendimento a uma mulher em situação de violência, "
    "como uma assistente social registraria. Use APENAS os fatos da ficha fornecida; não invente nomes, endereços, "
    "datas exatas ou documentos. Refira-se à mulher como 'a atendida' ou 'ela' e ao autor pela relação informada. "
    "Entre 60 e 140 palavras, tom profissional e humano. Não mencione encaminhamentos já feitos. Responda só com o relato."
)

SCHEMA = {"type": "object", "properties": {"relato": {"type": "string"}}, "required": ["relato"], "additionalProperties": False}


def describe(case: dict) -> str:
    lines = []
    for k, v in case.items():
        if v in (None, "", 0, "IGNORADO"):
            continue
        if k in VIOL:
            lines.append(f"- tipo de violência: {VIOL[k]}")
        elif k in AG:
            lines.append(f"- meio de agressão: {AG[k]}")
        elif k in REL_GROUPS:
            lines.append(f"- autor: {REL_GROUPS[k][0]}")
        elif k in CATEGORICAL:
            lines.append(f"- {CATEGORICAL[k][0]}: {label(str(v))}")
        elif k == "IDADE_ANOS":
            lines.append(f"- idade: {int(v)} anos")
        elif k == "SG_UF":
            lines.append(f"- UF: {v}")
        elif k == "ESCOLARIDADE":
            lines.append(f"- escolaridade: {label(str(v))}")
    return "\n".join(lines) or "- (ficha quase vazia)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    df = pd.read_parquet(PROCESSED_PATH)
    # stratified: half with referral, half without; prefer informative rows
    random.Random(args.seed)
    pos = df[df[TARGET] == 1].sample(args.n // 2, random_state=args.seed)
    neg = df[df[TARGET] == 0].sample(args.n - args.n // 2, random_state=args.seed + 1)
    sample = pd.concat([pos, neg]).sample(frac=1, random_state=args.seed)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    with OUT.open("a", encoding="utf-8") as f:
        for i, (_, row) in enumerate(sample.iterrows()):
            rid = f"syn-{args.seed}-{i:03d}"
            if rid in done:
                continue
            case = {k: (None if pd.isna(row[k]) else (int(row[k]) if k in VIOL or k in AG or k in REL_GROUPS else row[k])) for k in FEATURE_NAMES
                    if k not in ("ESCOLARIDADE_ORD", "ESCOLARIDADE_IGN", "IDADE_IGN")}
            case["IDADE_ANOS"] = None if pd.isna(row["IDADE_ANOS"]) else int(row["IDADE_ANOS"])
            esc = {0: "FUNDAMENTAL_INCOMPLETO", 1: "ENSINO_MEDIO_INCOMPLETO", 2: "ENSINO_MEDIO_COMPLETO", 3: "ENSINO_SUPERIOR"}
            case["ESCOLARIDADE"] = None if pd.isna(row["ESCOLARIDADE_ORD"]) else esc[int(row["ESCOLARIDADE_ORD"])]
            case = {k: (v.item() if hasattr(v, "item") else v) for k, v in case.items()}
            t0 = time.time()
            res = chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Ficha:\n{describe(case)}\n\nEscreva o relato em JSON."}],
                       role="generator", schema=SCHEMA, schema_name="relato", temperature=0.8, max_tokens=400)
            if res.backend == "error" or not res.parsed:
                print(f"[{rid}] erro: {res.error}", flush=True)
                continue
            rec = {"id": rid, "case": case, "target": int(row[TARGET]), "relato": res.parsed["relato"].strip(), "model": res.model,
                   "latency_ms": res.latency_ms}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            print(f"[{rid}] alvo={rec['target']} {len(rec['relato'].split())} palavras {int(time.time()-t0)}s", flush=True)


if __name__ == "__main__":
    main()
