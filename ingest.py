"""Build (or rebuild) the vector index from everything in docs/.

Run this once whenever documents change:   python ingest.py

Why a vector DATABASE instead of Phase 1's embed-on-every-question?
Embedding the corpus costs time and API calls. Doing it once and storing
the vectors on disk means questions only need ONE embedding call (the
question itself) — this is what makes RAG fast and cheap at scale.
"""

from pathlib import Path

import chromadb

from chunker import chunk_text
from llm import embed_texts

# Anchor paths to this file's folder so ingest/query/UI work no matter
# which directory they're launched from.
BASE_DIR = Path(__file__).parent
DB_PATH = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "docmind"
DOCS_FOLDER = str(BASE_DIR / "docs")


def main() -> None:
    # 1. Load + chunk
    chunks: list[dict] = []
    doc_paths = sorted(Path(DOCS_FOLDER).glob("*.md"))
    for path in doc_paths:
        chunks.extend(chunk_text(path.read_text(), source=path.name))
    print(f"Loaded {len(doc_paths)} documents -> {len(chunks)} chunks")

    # 2. Embed every chunk (one batched API call for small corpora)
    vectors = embed_texts([c["text"] for c in chunks])
    print(f"Embedded {len(vectors)} chunks ({len(vectors[0])} dimensions each)")

    # 3. Store in ChromaDB. Delete-and-recreate keeps re-ingesting idempotent:
    #    running this script twice never duplicates chunks.
    db = chromadb.PersistentClient(path=DB_PATH)
    try:
        db.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # first run — nothing to delete
    collection = db.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # score matches with cosine similarity
    )
    collection.add(
        ids=[f"{c['source']}-{i}" for i, c in enumerate(chunks)],
        embeddings=vectors,
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"]} for c in chunks],
    )
    print(f"Index written to {DB_PATH}/ — ready for query.py")


if __name__ == "__main__":
    main()
