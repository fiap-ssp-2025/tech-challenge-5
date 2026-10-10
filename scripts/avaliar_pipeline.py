#!/usr/bin/env python3
"""Batch evaluation of the full pipeline on the synthetic reports.

For each report in data/synthetic/relatos.jsonl:
  1. F2 structuring  → agreement with the real fields (binaries + OUT_VEZES) → precision/recall per field group
  2. F1 full graph   → judge verdict, blocklist, latency, number of cited sources, risk level
Writes docs/eval_pipeline.json and prints a summary table for the report.

Usage: uv run python scripts/avaliar_pipeline.py [--limit 30] [--skip-f1]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from guardia.data.features import AG, REL_GROUPS, VIOL
from guardia.graph.flow import run_case
from guardia.llm.structurer import structure_report

IN = ROOT / "data" / "synthetic" / "relatos.jsonl"
OUT = ROOT / "docs" / "eval_pipeline.json"
BIN_FIELDS = list(VIOL) + list(AG) + list(REL_GROUPS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--skip-f1", action="store_true")
    args = ap.parse_args()
    recs = [json.loads(l) for l in IN.read_text(encoding="utf-8").splitlines() if l.strip()][: args.limit]
    f2_rows, f1_rows = [], []
    tp = fp = fn = 0
    out_vezes_ok = 0
    for r in recs:
        t0 = time.time()
        st = structure_report(r["relato"])
        if st.ok:
            truth = r["case"]
            for k in BIN_FIELDS:
                a, b = int(bool(st.case.get(k))), int(bool(truth.get(k)))
                tp += a and b; fp += a and not b; fn += (not a) and b
            out_vezes_ok += int(str(st.case.get("OUT_VEZES")) == str(truth.get("OUT_VEZES")))
        f2_rows.append({"id": r["id"], "ok": st.ok, "latency_ms": st.latency_ms, "uncertain": st.uncertain, "error": st.error})
        print(f"[{r['id']}] F2 {'ok' if st.ok else 'ERRO'} {st.latency_ms} ms", flush=True)
        if args.skip_f1:
            continue
        s = run_case(r["relato"], st.case if st.ok else r["case"], st.signals if st.ok else {})
        log = s.get("audit_log") or {}
        judge = log.get("llm_judge") or {}
        resp = s.get("response") or {}
        f1_rows.append({
            "id": r["id"], "target": r["target"], "risk_level": s.get("risk_level"), "ml_probability": (s.get("ml") or {}).get("probability"),
            "ml_suggest": (s.get("ml") or {}).get("suggest_referral"), "llm_backend": log.get("llm_backend"), "llm_latency_ms": log.get("llm_latency_ms"),
            "status": log.get("safety_validation"), "judge_backend": judge.get("backend"), "judge_approved": judge.get("approved"),
            "judge_overrides": judge.get("overrides"), "n_sources": len(s.get("protocol_sources") or []), "n_cited": len(resp.get("fontes_citadas") or []),
            "audit_id": s.get("audit_id"), "total_s": round(time.time() - t0, 1),
        })
        print(f"   F1 risco={s.get('risk_level')} p={(s.get('ml') or {}).get('probability')} status={log.get('safety_validation')} "
              f"juiz={judge.get('approved')} fontes={len(s.get('protocol_sources') or [])} citadas={len(resp.get('fontes_citadas') or [])} "
              f"{f1_rows[-1]['total_s']}s", flush=True)

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    n_ok = sum(1 for x in f2_rows if x["ok"])
    summary = {
        "n": len(recs),
        "f2": {"ok": n_ok, "field_precision": round(prec, 3), "field_recall": round(rec, 3), "out_vezes_accuracy": round(out_vezes_ok / max(n_ok, 1), 3),
               "median_latency_ms": statistics.median([x["latency_ms"] for x in f2_rows if x["ok"]] or [0])},
    }
    if f1_rows:
        judged = [x for x in f1_rows if x["judge_approved"] is not None]
        ml_rows = [x for x in f1_rows if x["ml_suggest"] is not None]
        agree = sum(1 for x in ml_rows if int(bool(x["ml_suggest"])) == int(x["target"]))
        summary["f1"] = {
            "n": len(f1_rows), "llm_ok": sum(1 for x in f1_rows if x["status"] in ("approved", "blocked_by_judge")),
            "judge_evaluated": len(judged), "judge_approved": sum(1 for x in judged if x["judge_approved"]),
            "blocked": sum(1 for x in f1_rows if str(x["status"]).startswith("blocked")),
            "median_llm_latency_ms": statistics.median([x["llm_latency_ms"] or 0 for x in f1_rows]),
            "median_total_s": statistics.median([x["total_s"] for x in f1_rows]),
            "ml_agreement_with_sinan_label": round(agree / max(len(ml_rows), 1), 3),
            "mean_sources": round(statistics.mean([x["n_sources"] for x in f1_rows]), 2),
            "mean_cited": round(statistics.mean([x["n_cited"] for x in f1_rows]), 2),
            "risk_distribution": {lvl: sum(1 for x in f1_rows if x["risk_level"] == lvl) for lvl in ("baixo", "moderado", "alto", "critico")},
        }
    OUT.write_text(json.dumps({"summary": summary, "f2": f2_rows, "f1": f1_rows}, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    _write_report_block(summary)


def _write_report_block(summary: dict) -> None:
    """Fill the EVAL block of docs/relatorio_tecnico.md."""
    import re

    report = ROOT / "docs" / "relatorio_tecnico.md"
    if not report.exists():
        return
    f2 = summary["f2"]
    lines = [f"Relatos sintéticos avaliados: **{summary['n']}**.", "",
             (f"**F2 (estruturar relato).** {f2['ok']}/{summary['n']} extrações válidas; concordância por campo binário com a ficha de origem: "
             f"precisão {f2['field_precision']:.3f}, recall {f2['field_recall']:.3f}; acerto de `OUT_VEZES` {f2['out_vezes_accuracy']:.3f}; "
             f"latência mediana {f2['median_latency_ms']:.0f} ms.")]
    if "f1" in summary:
        f1 = summary["f1"]
        lines += ["", (f"**F1 (grafo completo).** {f1['llm_ok']}/{f1['n']} notas geradas pela LLM; juiz avaliou {f1['judge_evaluated']} e aprovou "
                  f"{f1['judge_approved']}; {f1['blocked']} bloqueadas (blocklist ou juiz). Latência mediana da LLM {f1['median_llm_latency_ms']:.0f} ms, "
                  f"tempo mediano total {f1['median_total_s']:.1f} s. Fontes recuperadas por nota: {f1['mean_sources']:.1f} (citadas {f1['mean_cited']:.1f}). "
                  f"Concordância entre sugestão do modelo e encaminhamento real da ficha: {f1['ml_agreement_with_sinan_label']:.3f}. "
                  f"Distribuição de risco por regras: {f1['risk_distribution']}.")]
    block = "\n".join(lines)
    text = report.read_text(encoding="utf-8")
    text = re.sub(r"<!-- EVAL:START -->.*?<!-- EVAL:END -->", f"<!-- EVAL:START -->\n{block}\n<!-- EVAL:END -->", text, flags=re.DOTALL)
    report.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
