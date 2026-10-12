#!/usr/bin/env python3
"""Retrieval evaluation of the protocol RAG.

Two sets, no LLM involved:
  * chat (F3): questions with the expected document and a pattern the chunk must contain → hit in the top-6
  * F1: the 60 synthetic reports, queried like the graph node does (report + risk factors) → an expected
    document in the top-5 (sexual violence → MS technical norm; otherwise a Maria da Penha / domestic violence text)

Usage: uv run python scripts/avaliar_rag.py [--mode hybrid|vector|bm25] [--json docs/eval_rag.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from guardia.graph.flow import assess_risk
from guardia.rag import retriever

MS = "ms_violencia_sexual.txt"
DOMESTIC = {"medidas_protetivas.txt", "lei_maria_da_penha.txt", "lei_maria_da_penha_resumo.txt", "protocolo_violencia.txt",
            "tipos_violencia_maria_da_penha.txt"}
DF = {"rede_protecao_mulher_df.txt", "canais_apoio_mulher_df.txt", "protocolo_rede_protecao_df.txt"}
CANAIS = {"canais_atendimento_mulher_brasil.txt", "canais_apoio_mulher_df.txt"}

# (id, query as the chat builds it — previous question, question, record facts —, expected files, pattern in the chunk)
CHAT = [
    ("caso-sexual", "Violência sexual; Ameaça de morte; Parceiro íntimo\nO que os protocolos indicam para este caso?", {MS}, r"72 ?h"),
    ("primeiras-horas", "O que fazer nas primeiras horas após violência sexual?", {MS}, r"72 ?h"),
    ("roteiro-72h", "O que fazer nas primeiras 72 h em violência sexual?", {MS}, r"72 ?h"),
    ("roteiro-medida", "Como solicitar medida protetiva de urgência?", {"medidas_protetivas.txt", "lei_maria_da_penha.txt",
                                                                       "lei_maria_da_penha_resumo.txt"}, r"medidas? protetivas?"),
    ("roteiro-df", "Como solicitar medida protetiva de urgência?\ne no DF, quais serviços?", DF, r"Distrito Federal|DF"),
    ("tipos", "Quais são os tipos de violência previstos na Lei Maria da Penha?",
     {"tipos_violencia_maria_da_penha.txt", "lei_maria_da_penha.txt"}, r"patrimonial"),
    ("patrimonial", "O que caracteriza violência patrimonial?", {"tipos_violencia_maria_da_penha.txt"}, r"patrimonial"),
    ("ligue-180", "Qual número ligar para a Central de Atendimento à Mulher?", CANAIS, r"180"),
    ("dna", "Como coletar vestígios biológicos para exame de DNA?", {MS}, r"DNA|vest[íi]gio"),
    ("contracepcao", "Quando fazer contracepção de emergência após estupro?", {MS}, r"contracep|anticoncep"),
    ("profilaxia", "Profilaxia de HIV e hepatite B após violência sexual", {MS}, r"profil[aá]x|profil[aá]t|retrovira"),
    ("boletim", "A vítima precisa registrar boletim de ocorrência para ser atendida na saúde?", {MS}, r"boletim"),
    ("notificacao", "Como fazer a notificação compulsória do caso de violência sexual?", {MS}, r"notifica"),
    ("aborto", "A mulher tem direito à interrupção da gravidez resultante de estupro?", {MS}, r"aborto|interrup"),
    ("lgpd", "Posso compartilhar os dados da atendida com outros serviços?", {"lgpd_dados_sensiveis.txt"}, r"compartilhamento|LGPD"),
    ("medidas-juiz", "Quais medidas protetivas podem ser aplicadas ao agressor?", {"medidas_protetivas.txt"}, r"afastamento"),
    ("caso-fisica", "Violência física; Objeto cortante; Parceiro íntimo\nO que os protocolos indicam para este caso?", DOMESTIC,
     r"medidas? protetivas?|acolhimento|risco"),
    ("adolescente", "O que fazer nas primeiras horas após violência sexual?\ne se for adolescente, o que muda?", {MS},
     r"adolescente"),
]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\xad\s*", "", text))


def run_chat(mode: str, k: int = 6) -> list[dict]:
    rows = []
    for qid, query, files, pattern in CHAT:
        hits = retriever.search(query, k=k, mode=mode)
        rank = next((i + 1 for i, h in enumerate(hits) if h.source in files and re.search(pattern, _norm(h.text), re.IGNORECASE)), None)
        rows.append({"id": qid, "hit": rank is not None, "rank": rank, "sources": [h.source for h in hits]})
    return rows


def run_f1(mode: str, k: int = 5) -> dict:
    recs = [json.loads(line) for line in (ROOT / "data/synthetic/relatos.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    hits = distinct = 0
    sexual = sexual_hit = sexual_72h = 0
    for r in recs:
        factors = assess_risk({"user_input": r["relato"], "case": r["case"], "signals": {}})["risk_factors"]
        res = retriever.search(r["relato"] + " " + " ".join(factors), k=k, mode=mode)
        got = {h.source for h in res}
        distinct += len(got)
        if r["case"].get("VIOL_SEXU"):
            sexual += 1
            ok = MS in got
            sexual_hit += ok
            sexual_72h += any(h.source == MS and re.search(r"72 ?h", _norm(h.text)) for h in res)
        else:
            ok = bool(got & DOMESTIC)
        hits += ok
    return {"n": len(recs), "hit": hits, "sexual": sexual, "sexual_hit": sexual_hit, "sexual_72h": sexual_72h,
            "distinct_sources_avg": round(distinct / len(recs), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "vector", "bm25"])
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    chat = run_chat(args.mode)
    f1 = run_f1(args.mode)
    print(f"modo={args.mode}  chunks={len(retriever.chunks())}")
    for r in chat:
        print(f"  {r['id']:<16} {'ok ' if r['hit'] else '-- '} rank={r['rank'] or '-'}  {', '.join(s.removesuffix('.txt') for s in r['sources'])}")
    print(f"chat: {sum(r['hit'] for r in chat)}/{len(chat)} no top-6")
    print(f"F1: {f1['hit']}/{f1['n']} com documento esperado no top-5 (violência sexual: {f1['sexual_hit']}/{f1['sexual']}, "
          f"com trecho das 72 h: {f1['sexual_72h']}/{f1['sexual']}); "
          f"fontes distintas/consulta: {f1['distinct_sources_avg']}")
    if args.json:
        args.json.write_text(json.dumps({"mode": args.mode, "chat": chat, "f1": f1}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
