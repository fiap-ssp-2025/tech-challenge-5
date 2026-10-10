"""RAG over the security protocols (Chroma + multilingual MiniLM).

Differences from TC3: one persistent collection for the whole corpus (built once, reused),
chunks carry their source file as metadata, and search returns (chunk, source, score) so the
answer can cite the document that grounded it.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parents[3]
PROTOCOLS_DIR = ROOT / "data" / "protocols"
PERSIST_DIR = ROOT / "data" / "generated" / "chroma"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

SOURCE_TITLES = {
    "protocolo_violencia.txt": "Protocolo de atendimento em violência doméstica",
    "lei_maria_da_penha.txt": "Lei Maria da Penha (Lei 11.340/2006)",
    "lei_maria_da_penha_resumo.txt": "Lei Maria da Penha — resumo institucional",
    "tipos_violencia_maria_da_penha.txt": "Tipos de violência (Lei Maria da Penha, art. 7º)",
    "medidas_protetivas.txt": "Medidas protetivas de urgência",
    "canais_atendimento_mulher_brasil.txt": "Canais de atendimento à mulher — Brasil",
    "canais_apoio_mulher_df.txt": "Canais de apoio à mulher — Distrito Federal",
    "rede_protecao_mulher_df.txt": "Rede de proteção à mulher — DF",
    "protocolo_rede_protecao_df.txt": "Protocolo da rede de proteção — DF",
    "lgpd_dados_sensiveis.txt": "LGPD — dados sensíveis",
    "ms_violencia_sexual.txt": "MS/MJ — Norma técnica: atenção humanizada à violência sexual",
}


@dataclass
class Hit:
    text: str
    source: str
    title: str
    score: float


def _corpus_fingerprint() -> str:
    h = hashlib.sha1()
    for p in sorted(PROTOCOLS_DIR.glob("*.txt")):
        h.update(p.name.encode()); h.update(str(p.stat().st_size).encode())
    h.update(EMBEDDING_MODEL.encode())
    return h.hexdigest()[:10]


@lru_cache(maxsize=1)
def embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _load_documents() -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=80)
    docs: list[Document] = []
    for p in sorted(PROTOCOLS_DIR.glob("*.txt")):
        text = p.read_text(encoding="utf-8", errors="ignore").replace("﻿", "")
        for i, chunk in enumerate(splitter.split_text(text)):
            if chunk.strip():
                docs.append(Document(page_content=chunk.strip(), metadata={"source": p.name, "chunk": i,
                                                                             "title": SOURCE_TITLES.get(p.name, p.stem)}))
    return docs


@lru_cache(maxsize=1)
def vectorstore() -> Chroma:
    name = f"protocolos_{_corpus_fingerprint()}"
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    store = Chroma(collection_name=name, embedding_function=embeddings(), persist_directory=str(PERSIST_DIR),
                   collection_metadata={"hnsw:space": "cosine"})
    if store._collection.count() == 0:  # first run: index the corpus once
        docs = _load_documents()
        store.add_documents(docs, ids=[f"{d.metadata['source']}::{d.metadata['chunk']}" for d in docs])
    return store


def search(query: str, k: int = 6, max_per_source: int = 2) -> list[Hit]:
    results = vectorstore().similarity_search_with_score(query, k=k * 3)  # cosine distance, lower = closer
    hits: list[Hit] = []
    per_source: dict[str, int] = {}
    for doc, distance in results:
        score = 1.0 - float(distance)  # cosine similarity in [−1, 1]
        src = doc.metadata["source"]
        if per_source.get(src, 0) >= max_per_source:
            continue
        per_source[src] = per_source.get(src, 0) + 1
        hits.append(Hit(doc.page_content, src, doc.metadata.get("title", src), float(score)))
        if len(hits) >= k:
            break
    return hits


def corpus_summary() -> list[dict]:
    return [{"file": p.name, "title": SOURCE_TITLES.get(p.name, p.stem), "words": len(p.read_text(encoding='utf-8', errors='ignore').split())}
            for p in sorted(PROTOCOLS_DIR.glob("*.txt"))]
