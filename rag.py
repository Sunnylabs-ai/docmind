"""DocMind — Phase 1: a complete RAG pipeline in one readable file.

The whole idea in one sentence: instead of asking the LLM to answer from
memory, we first FIND the most relevant passages in our own documents,
then hand those passages to the LLM and say "answer using only this."

Pipeline:  load -> chunk -> embed -> retrieve -> generate

Run it:    python rag.py "How many vacation days do I get?"
"""

import math
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai

# Reads .env and puts GOOGLE_API_KEY into the environment,
# where the Gemini client automatically finds it.
load_dotenv()
client = genai.Client()

EMBED_MODEL = "gemini-embedding-001"   # turns text into vectors
CHAT_MODEL = "gemini-3.5-flash"        # writes the final answer
TOP_K = 3                              # how many chunks to retrieve


# ── Stage 1: LOAD ─────────────────────────────────────────────────────────────
def load_documents(folder: str = "docs") -> list[dict]:
    """Read every .md file in the folder. Returns [{source, text}, ...]."""
    documents = []
    for path in sorted(Path(folder).glob("*.md")):
        documents.append({"source": path.name, "text": path.read_text()})
    return documents


# ── Stage 2: CHUNK ────────────────────────────────────────────────────────────
def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split each document into paragraphs.

    Why chunk at all? Two reasons:
    1. Retrieval precision — a whole file about 10 topics matches every
       question a little; a paragraph about one topic matches its question a lot.
    2. Context size — we only want to send the LLM the relevant parts.

    Splitting on blank lines is the simplest possible strategy. Phase 2
    replaces this with smarter chunking (overlap, size limits).
    """
    chunks = []
    for doc in documents:
        for paragraph in doc["text"].split("\n\n"):
            paragraph = paragraph.strip()
            if len(paragraph) > 50:  # skip headings and tiny fragments
                chunks.append({"source": doc["source"], "text": paragraph})
    return chunks


# ── Stage 3: EMBED ────────────────────────────────────────────────────────────
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Turn each text into a vector — a list of ~3000 numbers.

    The magic property: texts with similar MEANING get vectors that point
    in similar directions, even if they share no words. "How much time off
    do I get?" lands near "23 paid vacation days per year".
    """
    result = client.models.embed_content(model=EMBED_MODEL, contents=texts)
    return [embedding.values for embedding in result.embeddings]


# ── Stage 4: RETRIEVE ─────────────────────────────────────────────────────────
def cosine_similarity(a: list[float], b: list[float]) -> float:
    """How similar do two vectors point? 1.0 = same direction, 0 = unrelated.

    This is the entire 'search engine' of a RAG system — just a dot
    product divided by the vectors' lengths. No magic.
    """
    dot = sum(x * y for x, y in zip(a, b))
    length_a = math.sqrt(sum(x * x for x in a))
    length_b = math.sqrt(sum(y * y for y in b))
    return dot / (length_a * length_b)


def retrieve(question: str, chunks: list[dict], chunk_vectors: list[list[float]]) -> list[dict]:
    """Embed the question, score it against every chunk, return the top K."""
    question_vector = embed_texts([question])[0]
    scored = []
    for chunk, vector in zip(chunks, chunk_vectors):
        score = cosine_similarity(question_vector, vector)
        scored.append({**chunk, "score": score})
    scored.sort(key=lambda c: c["score"], reverse=True)
    return scored[:TOP_K]


# ── Stage 5: GENERATE ─────────────────────────────────────────────────────────
def generate_answer(question: str, retrieved: list[dict]) -> str:
    """Hand the retrieved chunks to the LLM and ask it to answer from them."""
    context = "\n\n".join(
        f"[{c['source']}]\n{c['text']}" for c in retrieved
    )
    prompt = f"""Answer the question using ONLY the context below.
If the context doesn't contain the answer, say "I don't know based on my documents."
Cite the source file name in your answer.

Context:
{context}

Question: {question}"""
    response = client.models.generate_content(model=CHAT_MODEL, contents=prompt)
    return response.text


# ── Putting it all together ───────────────────────────────────────────────────
def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "How many vacation days do I get?"

    documents = load_documents()
    chunks = chunk_documents(documents)
    print(f"Loaded {len(documents)} documents -> {len(chunks)} chunks\n")

    chunk_vectors = embed_texts([c["text"] for c in chunks])

    retrieved = retrieve(question, chunks, chunk_vectors)
    print(f'Question: "{question}"\n')
    print("Top matches (cosine similarity):")
    for c in retrieved:
        preview = c["text"][:70].replace("\n", " ")
        print(f'  {c["score"]:.3f}  [{c["source"]}]  {preview}...')

    print("\nAnswer:")
    print(generate_answer(question, retrieved))


if __name__ == "__main__":
    main()
