"""Ask DocMind a question:   python query.py "How many vacation days do I get?"

Assumes the index exists — run `python ingest.py` first (and after any
change to the docs/ folder).
"""

import sys

import chromadb

from ingest import DB_PATH, COLLECTION_NAME
from llm import embed_texts, generate

TOP_K = 3


def retrieve(question: str) -> list[dict]:
    """Embed the question and let ChromaDB find the nearest chunks.

    Chroma returns cosine DISTANCE (0 = identical); we convert to
    similarity (1 = identical) to keep the same mental model as Phase 1.
    """
    db = chromadb.PersistentClient(path=DB_PATH)
    try:
        collection = db.get_collection(COLLECTION_NAME)
    except Exception:
        sys.exit("No index found — run `python ingest.py` first.")

    question_vector = embed_texts([question])[0]
    result = collection.query(
        query_embeddings=[question_vector],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"text": text, "source": meta["source"], "score": 1 - distance}
        for text, meta, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


def answer(question: str, retrieved: list[dict]) -> str:
    context = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in retrieved)
    prompt = f"""Answer the question using ONLY the context below.
If the context doesn't contain the answer, say "I don't know based on my documents."
Cite the source file name in your answer.

Context:
{context}

Question: {question}"""
    return generate(prompt)


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "How many vacation days do I get?"

    retrieved = retrieve(question)
    print(f'Question: "{question}"\n')
    print("Top matches (cosine similarity):")
    for c in retrieved:
        preview = c["text"][:70].replace("\n", " ")
        print(f'  {c["score"]:.3f}  [{c["source"]}]  {preview}...')

    print("\nAnswer:")
    print(answer(question, retrieved))


if __name__ == "__main__":
    main()
