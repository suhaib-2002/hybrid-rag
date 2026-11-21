# Hybrid RAG (Dense + BM25 + Reciprocal Rank Fusion)

A FastAPI backend that combines **semantic (embedding) search** with **keyword (BM25) search**, merges the rankings with **Reciprocal Rank Fusion (RRF)**, and asks a Groq LLM to answer with citations.

## Why hybrid?
| Retriever | Good at | Weak at |
|---|---|---|
| Dense (embeddings) | Meaning, paraphrases | Exact names, acronyms, rare terms |
| BM25 (keywords) | Exact term matches | Synonyms, reworded questions |

## How it works
```
Indexing (POST /ingest):  data/ -> 800-char chunks -> embeddings in ChromaDB  +  BM25 index

Query (POST /ask):        question -> dense top 20 ─┐
                                      BM25 top 20 ─┴─> RRF fusion -> top 5 chunks -> Groq LLM -> answer with [n] citations
```
RRF score of a chunk = sum of `1 / (60 + rank)` over both rankings.

## Project files
```
hybrid_rag.py       the whole backend (indexing, retrieval, generation, FastAPI)
start.bat           double-click to create the venv, install packages and start the server
requirements.txt    dependencies
COMMANDS.txt        step-by-step run commands
.env                Groq API key and model settings
data/               documents to search (14 sample Wikipedia files included)
```

## Quick start
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
# a temporary Groq key is already in .env (replace it with your own if you like)
uvicorn hybrid_rag:app --reload
```
On Windows you can also just double-click `start.bat`. Then open http://localhost:8000/docs. Full commands are in [COMMANDS.txt](COMMANDS.txt).

## How to use the API
Open http://localhost:8000/docs, click an endpoint, then **Try it out** and **Execute**.

| Endpoint | What to enter | Result |
|---|---|---|
| `GET /health` | nothing | Indexed chunk count and the files in `data/` |
| `POST /upload` | **Choose File**: a `.pdf`, `.txt` or `.md` | Saved to `data/`. Not searchable until `/ingest` |
| `POST /ingest` | nothing | Rebuilds the index from `data/`. Run first, and after every upload |
| `POST /ask` | the JSON below | `answer` and `sources` |

```json
{"question": "What does BM25 stand for?", "mode": "hybrid", "top_k": 5}
```
`mode` is `hybrid` (default), `dense` or `bm25`. `top_k` is how many chunks go to the LLM.

**Suggested test order:** `/ingest`, then `/ask` the same question in each mode and compare the sources, then `/upload` your own file, `/ingest`, and `/ask` about it. Ask something not in the documents to confirm it says it doesn't know.

## Groq API key
A temporary key is included in `.env` so the code runs out of the box for testing. To use your own, set `GROQ_API_KEY` in `.env` (free at https://console.groq.com) and restart the server.

## Tech stack
Python 3.10+, FastAPI, sentence-transformers (all-MiniLM-L6-v2), ChromaDB, rank-bm25, pypdf, Groq (`openai/gpt-oss-20b`).
