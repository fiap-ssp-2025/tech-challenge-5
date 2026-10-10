"""`guardia` command line: build-data, train, serve, status, demo."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="guardia")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-data", help="SINAN bruto → tabela de treino com o alvo de encaminhamento")
    t = sub.add_parser("train", help="treina e compara os modelos"); t.add_argument("--ga", action="store_true"); t.add_argument("--quick", action="store_true")
    s = sub.add_parser("serve", help="sobe a interface web"); s.add_argument("--host", default="0.0.0.0"); s.add_argument("--port", type=int, default=8000); s.add_argument("--reload", action="store_true")
    sub.add_parser("status", help="checa LLM, juiz, modelo e RAG")
    d = sub.add_parser("demo", help="roda o grafo num relato de exemplo"); d.add_argument("--text", default=None)
    args = p.parse_args(argv)

    if args.cmd == "build-data":
        from guardia.data.features import TARGET, build_and_save
        t_ = build_and_save(); print(f"{len(t_):,} linhas · prevalência {t_[TARGET].mean():.3f}")
    elif args.cmd == "train":
        from guardia.ml import train
        sys.argv = ["train"] + (["--ga"] if args.ga else []) + (["--quick"] if args.quick else [])
        train.main()
    elif args.cmd == "serve":
        import uvicorn
        uvicorn.run("guardia.web.app:app", host=args.host, port=args.port, reload=args.reload)
    elif args.cmd == "status":
        from guardia.web.status import full_status
        print(json.dumps(full_status(), indent=2, ensure_ascii=False))
    elif args.cmd == "demo":
        from guardia.graph.flow import run_case
        text = args.text or ("Meu companheiro me ameaçou com uma faca ontem à noite. Não é a primeira vez, já aconteceu "
                             "outras vezes. Tenho dois filhos pequenos e não tenho para onde ir.")
        case = {"VIOL_FISIC": 1, "VIOL_PSICO": 1, "AG_AMEACA": 1, "AG_CORTE": 1, "REL_PARCEIRO_INTIMO": 1,
                "OUT_VEZES": "SIM", "AUTOR_SEXO": "MASCULINO", "CS_RACA": "PARDA", "CICL_VID": "ADULTA",
                "SIT_CONJUG": "CASADA_UNIAO", "LOCAL_OCOR": "RESIDENCIA", "SG_UF": "DF", "IDADE_ANOS": 29}
        out = run_case(text, case, {"arma_mencionada": True, "filhos_envolvidos": True, "local_seguro": False})
        keep = {k: out.get(k) for k in ("risk_level", "risk_factors", "ml", "protocol_sources", "response", "llm_metadata", "audit_id")}
        print(json.dumps(keep, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
