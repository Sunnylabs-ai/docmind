# DocMind 🧠

Chat with your documents. A Retrieval-Augmented Generation (RAG) system built from scratch in Python — no LangChain, no frameworks — to understand every moving part of the pipeline.

> 🚧 **Work in progress** — building in public. Follow the commit history to see it grow phase by phase.

## How it works

```
documents → chunk → embed → vector store
                                  ↓
question  → embed → find nearest chunks → LLM answers from those chunks
```

## Tech stack

- **Python** — pipeline and glue
- **Gemini API** — answer generation + embeddings
- **ChromaDB** — persistent vector store
- **Streamlit** — chat UI *(coming in Phase 3)*

## Run it yourself

```bash
git clone https://github.com/Sunnylabs-ai/docmind && cd docmind
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then put your Gemini API key inside
python ingest.py            # index the documents in docs/
python query.py "How many vacation days do I get?"   # CLI
streamlit run app.py                                  # chat UI
```

Drop your own `.md` files into `docs/` and re-run `ingest.py` to chat with anything.

## Project structure

| File | Role |
|---|---|
| `app.py` | Streamlit chat UI — history, spinner, per-answer source chunks |
| `ingest.py` | Load → chunk → embed → store in ChromaDB (run when docs change) |
| `query.py` | Embed question → retrieve top chunks → generate cited answer |
| `chunker.py` | Overlapping, heading-aware chunking strategy |
| `llm.py` | All Gemini calls in one swappable module |
| `rag.py` | The Phase 1 single-file pipeline — kept as the project's origin story |

## Lessons learned (so far)

- **Chunking quality beats database choice.** In Phase 1, "who founded the company?" retrieved a *menu item* as the top match, because lone paragraphs lack context. Grouping paragraphs under their headings fixed the ranking — before touching any database.
- **Top-k retrieval is a safety net.** Even when the #1 chunk was wrong, the right one was in the top 3, so the answer survived.
- **Grounding works.** Asking for the (nonexistent) wifi password returns "I don't know" instead of a hallucination — enforced by one line in the prompt.

## Roadmap

- [x] Phase 0 — Project setup
- [x] Phase 1 — Minimal RAG pipeline in a single script (`rag.py`)
- [x] Phase 2 — Real chunking + persistent vector store (ChromaDB)
- [x] Phase 3 — Streamlit chat UI with source citations
- [ ] Phase 4 — Full documentation + architecture diagram
- [ ] Phase 5 — Live deployment
