"""One place for every Gemini API call.

Keeping the LLM behind this thin wrapper means the rest of DocMind doesn't
know or care which provider we use — swapping Gemini for Claude, OpenAI,
or a local model would be a change to this one file.
"""

from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()  # finds GOOGLE_API_KEY in the environment

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash"

# The embedding endpoint caps how many texts one request may contain,
# so we send large corpora in slices.
EMBED_BATCH_SIZE = 50


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Turn texts into vectors, batching politely for large corpora."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        result = client.models.embed_content(model=EMBED_MODEL, contents=batch)
        vectors.extend(e.values for e in result.embeddings)
    return vectors


def generate(prompt: str) -> str:
    """Ask the chat model for a completion."""
    response = client.models.generate_content(model=CHAT_MODEL, contents=prompt)
    return response.text
