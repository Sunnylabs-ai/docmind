"""One place for every Gemini API call.

Keeping the LLM behind this thin wrapper means the rest of DocMind doesn't
know or care which provider we use — swapping Gemini for Claude, OpenAI,
or a local model would be a change to this one file.
"""

import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv()
client = genai.Client()  # finds GOOGLE_API_KEY in the environment

EMBED_MODEL = "gemini-embedding-001"
# flash-lite: same grounded-answer quality for RAG, much larger free-tier
# daily quota (gemini-3.5-flash allows only 20 requests/day free)
CHAT_MODEL = "gemini-3.1-flash-lite"

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


def generate(prompt: str, retries: int = 3) -> str:
    """Ask the chat model for a completion, riding out rate limits.

    The free tier allows 5 requests/minute. Instead of crashing on a
    429 "slow down" response, wait and try again with growing pauses —
    the standard pattern (exponential backoff) every production system
    uses for talking to APIs.
    """
    for attempt in range(retries + 1):
        try:
            response = client.models.generate_content(
                model=CHAT_MODEL, contents=prompt
            )
            return response.text
        except errors.ClientError as e:
            if getattr(e, "code", None) == 429 and attempt < retries:
                wait = 20 * (attempt + 1)
                print(f"  (rate limited — waiting {wait}s, attempt {attempt + 1}/{retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("unreachable")
