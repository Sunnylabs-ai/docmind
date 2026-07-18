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
- **Gemini API** — answer generation
- **ChromaDB** — vector store + embeddings
- **Streamlit** — chat UI *(coming in Phase 3)*

## Roadmap

- [x] Phase 0 — Project setup
- [ ] Phase 1 — Minimal RAG pipeline in a single script
- [ ] Phase 2 — Real chunking + persistent vector store (ChromaDB)
- [ ] Phase 3 — Streamlit chat UI with source citations
- [ ] Phase 4 — Full documentation + architecture diagram
- [ ] Phase 5 — Live deployment
