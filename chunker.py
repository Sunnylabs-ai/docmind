"""Split documents into overlapping chunks.

Phase 1 used one-paragraph chunks. Two problems with that:
1. A lone paragraph can lack context ("It costs $6.25" — what does?).
   Grouping paragraphs keeps headings attached to their sections.
2. A fact can straddle a boundary. OVERLAP fixes this: each chunk starts
   with the last paragraph of the previous one, so boundary-facts appear
   in two chunks and can't fall through the crack.
"""

CHUNK_SIZE = 700  # target characters per chunk — big enough for context,
                  # small enough that one chunk covers one topic


def chunk_text(text: str, source: str) -> list[dict]:
    """Group consecutive paragraphs into ~CHUNK_SIZE chunks with 1-paragraph overlap."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[dict] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        # Would adding this paragraph overflow the chunk? Close it out first.
        if current and current_len + len(para) > CHUNK_SIZE:
            chunks.append({"source": source, "text": "\n\n".join(current)})
            current = [current[-1]]           # the overlap: carry last paragraph forward
            current_len = len(current[0])
        current.append(para)
        current_len += len(para)

    if current:
        chunks.append({"source": source, "text": "\n\n".join(current)})
    return chunks
