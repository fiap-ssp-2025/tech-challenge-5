"""RAG over the security protocols (Chroma + multilingual MiniLM, plus BM25).

Differences from TC3: one persistent collection for the whole corpus (built once, reused),
chunks carry their source file as metadata, and search returns (chunk, source, score) so the
answer can cite the document that grounded it.

Spec 005: the PDF-extracted text is normalized before chunking (soft hyphens, wrapped lines, page
numbers; form and table-of-contents chunks are dropped), chunks are smaller, and search fuses the
vector ranking with a BM25 ranking (reciprocal rank fusion), so exact terms such as "72 horas" are
found inside long documents. A theme mentioned in the query can also pin its key passage (THEMES).
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import threading
import unicodedata
from collections import Counter
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


INDEX_VERSION = "3"  # bump when chunking changes, so a new Chroma collection is built
CHUNK_SIZE, CHUNK_OVERLAP = 450, 60
RRF_K, POOL = 60, 30  # reciprocal rank fusion constant; candidates taken from each ranking

# A theme found in the query pins its key passage (best BM25 chunk for the passage query) in the last slot,
# when the passage did not come up on its own. Generic case queries ("Violência sexual; ... O que os
# protocolos indicam?") otherwise miss the point the protocol is built around.
THEMES = [
    (re.compile(r"viol[eê]ncia sexual|estupr|abuso sexual|rela[cç][aã]o (sexual )?for[cç]ada", re.IGNORECASE),
     "atendimento em até 72 horas profilaxia contracepção de emergência coleta de vestígios"),
]

_STOPWORDS = frozenset([
    "a", "o", "as", "os", "de", "da", "do", "das", "dos", "e", "em", "no", "na", "nos", "nas", "um", "uma", "para", "por",
    "com", "que", "se", "ao", "aos", "ou", "como", "qual", "quais", "este", "esta", "esse", "essa", "isso", "caso",
])


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
    h.update(EMBEDDING_MODEL.encode()); h.update(INDEX_VERSION.encode())
    return h.hexdigest()[:10]


@lru_cache(maxsize=1)
def embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _normalize(text: str) -> str:
    """Undo PDF extraction artifacts: "agres\xad\nsão" → "agressão", lines wrapped mid-sentence, page numbers."""
    text = text.replace("\ufeff", "")
    text = re.sub(r"\xad\s*\n\s*", "", text).replace("\xad", "")
    text = re.sub(r"(?m)^\s*\d{1,3}\s*$", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n(?=[a-zà-ú(])", " ", text)  # next line starts in lowercase: same sentence
    return re.sub(r"[ \t]{2,}", " ", text)


def _is_noise(chunk: str) -> bool:
    """Blank forms ("Sim ( )", "_____") and the table of contents ("....... 20") match many queries and answer none."""
    return chunk.count("( )") >= 3 or chunk.count("....") >= 3 or chunk.count("_" * 10) >= 2 or len(chunk) < 40


def _tokens(text: str) -> list[str]:
    """BM25 tokens: lowercase, no accents, digits split from letters ("72h" → "72", "h"), 6-char prefix as stem."""
    text = "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")
    return [t[:6] for t in re.findall(r"\d+|[^\W\d_]+", text) if t not in _STOPWORDS and (len(t) > 1 or t.isdigit() or t == "h")]


class _BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / max(len(docs), 1)
        df = Counter(t for d in docs for t in set(d))
        self.idf = {t: math.log(1 + (len(docs) - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.k1, self.b = k1, b

    def ranking(self, query: str) -> list[int]:
        q = [t for t in set(_tokens(query)) if t in self.idf]
        scores = []
        for i, tf in enumerate(self.tf):
            s = sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
                    for t in q if t in tf)
            if s > 0:
                scores.append((s, i))
        return [i for _, i in sorted(scores, reverse=True)]


def _load_documents() -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                                              separators=["\n\n", "\n", ". ", " ", ""], keep_separator="end")
    docs: list[Document] = []
    for p in sorted(PROTOCOLS_DIR.glob("*.txt")):
        text = _normalize(p.read_text(encoding="utf-8", errors="ignore"))
        for i, chunk in enumerate(splitter.split_text(text)):
            if chunk.strip() and not _is_noise(chunk):
                docs.append(Document(page_content=chunk.strip(), metadata={"source": p.name, "chunk": i,
                                                                             "title": SOURCE_TITLES.get(p.name, p.stem)}))
    return docs


@lru_cache(maxsize=1)
def chunks() -> list[Document]:
    return _load_documents()


def _doc_id(doc: Document) -> str:
    return f"{doc.metadata['source']}::{doc.metadata['chunk']}"


@lru_cache(maxsize=1)
def _bm25() -> _BM25:
    return _BM25([_tokens(d.page_content) for d in chunks()])


_STORE_LOCK = threading.Lock()


def vectorstore() -> Chroma:
    """One Chroma client per process. The status thread and the first request reach this together right after
    startup; two clients created at once on the same directory broke chromadb's client cache (KeyError on the path)."""
    with _STORE_LOCK:
        return _vectorstore()


@lru_cache(maxsize=1)
def _vectorstore() -> Chroma:
    name = f"protocolos_{_corpus_fingerprint()}"
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    store = Chroma(collection_name=name, embedding_function=embeddings(), persist_directory=str(PERSIST_DIR),
                   collection_metadata={"hnsw:space": "cosine"})
    if store._collection.count() == 0:  # first run: index the corpus once
        docs = chunks()
        store.add_documents(docs, ids=[_doc_id(d) for d in docs])
    return store


def search(query: str, k: int = 6, max_per_source: int = 2, mode: str = "hybrid") -> list[Hit]:
    """``mode``: "hybrid" (vector + BM25, reciprocal rank fusion, theme passage), "vector" or "bm25" (for evaluation).
    At most ``max_per_source`` chunks per document, so short texts are not drowned by the long technical norm."""
    docs = chunks()
    index = {_doc_id(d): i for i, d in enumerate(docs)}
    # the corpus is small (~230 chunks): rank all of it, so the per-source cap never runs out of candidates
    vec = [(index[_doc_id(d)], 1.0 - float(dist)) for d, dist in vectorstore().similarity_search_with_score(query, k=len(docs))
           if _doc_id(d) in index]  # cosine distance → similarity in [−1, 1]
    cosine = dict(vec)
    if mode == "vector":
        order = [i for i, _ in vec]
    elif mode == "bm25":
        order = _bm25().ranking(query)
    else:
        fused: Counter[int] = Counter()
        for ranking in ([i for i, _ in vec][:POOL], _bm25().ranking(query)[:POOL]):
            for r, i in enumerate(ranking):
                fused[i] += 1.0 / (RRF_K + r + 1)
        order = [i for i, _ in fused.most_common()]
        order += [i for i, _ in vec if i not in fused]  # fill up to k when the cap cuts the fused candidates
    picked: list[int] = []
    per_source: Counter[str] = Counter()
    for i in order:
        src = docs[i].metadata["source"]
        if per_source[src] < max_per_source:
            per_source[src] += 1
            picked.append(i)
            if len(picked) >= k:
                break
    if mode == "hybrid":
        for trigger, passage in THEMES:
            if trigger.search(query):
                best = _bm25().ranking(passage)[:1]
                if best and best[0] not in picked:
                    picked = picked[: k - 1] + best
    return [Hit(docs[i].page_content, docs[i].metadata["source"], docs[i].metadata.get("title", docs[i].metadata["source"]),
                cosine.get(i, 0.0)) for i in picked]


def corpus_summary() -> list[dict]:
    return [{"file": p.name, "title": SOURCE_TITLES.get(p.name, p.stem), "words": len(p.read_text(encoding='utf-8', errors='ignore').split())}
            for p in sorted(PROTOCOLS_DIR.glob("*.txt"))]
