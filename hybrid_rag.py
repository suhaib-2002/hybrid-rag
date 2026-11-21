"""
Hybrid RAG backend (FastAPI): dense (embeddings) + keyword (BM25) search fused with RRF.
Run:  uvicorn hybrid_rag:app --reload      then open http://localhost:8000/docs
"""
import json
import os
import re
from pathlib import Path

import chromadb
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from groq import Groq
from pydantic import BaseModel
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

load_dotenv()
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
DATA_DIR, INDEX_DIR = Path("data"), Path("index")
STOP = set("a an the and or of to in on is are was were do does did what which who how why this that it for with".split())

app = FastAPI(title="Hybrid RAG (Dense + BM25 + RRF)")
embedder = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
chunks, by_id, bm25, col = [], {}, None, None


def tokenize(text):
    return [w for w in re.findall(r"\w+", text.lower()) if w not in STOP]


# ---------- indexing ----------
def load_docs():
    docs = []
    for f in sorted(DATA_DIR.glob("*")):
        if f.suffix.lower() == ".pdf":
            docs.append((f.name, " ".join(p.extract_text() or "" for p in PdfReader(f).pages)))
        elif f.suffix.lower() in (".txt", ".md"):
            docs.append((f.name, f.read_text(encoding="utf-8", errors="ignore")))
    return docs


def split(text, size=800, overlap=150):
    text = re.sub(r"\s+", " ", text)
    return [text[i:i + size] for i in range(0, len(text), size - overlap)]


def build_index():
    items = [{"id": f"{name}::{i}", "source": name, "text": c}
             for name, text in load_docs() for i, c in enumerate(split(text))]
    if not items:
        raise HTTPException(400, "No documents found in the data/ folder.")
    INDEX_DIR.mkdir(exist_ok=True)
    (INDEX_DIR / "chunks.json").write_text(json.dumps(items), encoding="utf-8")

    client = chromadb.PersistentClient(str(INDEX_DIR / "chroma"))
    if "chunks" in [c.name for c in client.list_collections()]:
        client.delete_collection("chunks")
    store = client.create_collection("chunks", metadata={"hnsw:space": "cosine"})
    vectors = embedder.encode([c["text"] for c in items], normalize_embeddings=True).tolist()
    for i in range(0, len(items), 500):
        part = items[i:i + 500]
        store.add(ids=[c["id"] for c in part], documents=[c["text"] for c in part], embeddings=vectors[i:i + 500])
    load_index()
    return len(items)


def load_index():
    global chunks, by_id, bm25, col
    chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in chunks}
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    col = chromadb.PersistentClient(str(INDEX_DIR / "chroma")).get_collection("chunks")


# ---------- retrieval ----------
def dense(query, n=20):
    vec = embedder.encode([query], normalize_embeddings=True).tolist()
    return col.query(query_embeddings=vec, n_results=min(n, len(chunks)))["ids"][0]


def keyword(query, n=20):
    scores = bm25.get_scores(tokenize(query))
    top = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
    return [chunks[i]["id"] for i in top if scores[i] > 0]


def rrf(rankings, k=60):
    """Reciprocal Rank Fusion: score = sum of 1 / (k + rank) over each ranking."""
    score = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, 1):
            score[cid] = score.get(cid, 0) + 1 / (k + rank)
    return sorted(score, key=score.get, reverse=True)


def search(query, mode="hybrid", top_k=5):
    if mode == "dense":
        ids = dense(query)
    elif mode == "bm25":
        ids = keyword(query)
    else:
        ids = rrf([dense(query), keyword(query)])
    return [by_id[i] for i in ids[:top_k]]


# ---------- generation ----------
def generate(question, hits):
    key = os.getenv("GROQ_API_KEY", "")
    if not key or key.startswith("your_"):
        raise HTTPException(500, "Set your GROQ_API_KEY in the .env file.")
    context = "\n\n".join(f"[{i}] {h['text']}" for i, h in enumerate(hits, 1))
    prompt = ("Answer using only the context below and cite sources like [1]. "
              f"If the answer is not in the context, say you don't know.\n\nContext:\n{context}\n\nQuestion: {question}")
    for _ in range(3):  # the model sometimes returns an empty reply
        reply = Groq(api_key=key).chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}])
        if reply.choices[0].message.content:
            return reply.choices[0].message.content
    return "No answer returned, please try again."


# ---------- API ----------
class Question(BaseModel):
    question: str
    mode: str = "hybrid"  # dense | bm25 | hybrid
    top_k: int = 5


@app.get("/health")
def health():
    return {"status": "ok", "indexed_chunks": len(chunks), "documents": [f.name for f in DATA_DIR.glob("*")]}


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    """Save a .pdf/.txt/.md file into data/. Call /ingest afterwards to index it."""
    name = Path(file.filename).name
    if Path(name).suffix.lower() not in (".pdf", ".txt", ".md"):
        raise HTTPException(400, "Only .pdf, .txt and .md files are supported.")
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / name).write_bytes(file.file.read())
    return {"saved": name}


@app.post("/ingest")
def ingest():
    """(Re)build the dense and BM25 indexes from everything in data/."""
    return {"indexed_chunks": build_index()}


@app.post("/ask")
def ask(q: Question):
    if not chunks:
        raise HTTPException(400, "Index is empty. Call POST /ingest first.")
    if q.mode not in ("dense", "bm25", "hybrid"):
        raise HTTPException(400, "mode must be dense, bm25 or hybrid.")
    hits = search(q.question, q.mode, q.top_k)
    return {"answer": generate(q.question, hits),
            "sources": [{"source": h["source"], "text": h["text"][:200]} for h in hits]}


if (INDEX_DIR / "chunks.json").exists():
    load_index()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
