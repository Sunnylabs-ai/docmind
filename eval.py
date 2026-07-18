"""Measure DocMind's quality instead of guessing at it.

Run:   python eval.py

Every question below has a known correct source and a fact the answer
must contain. Three metrics come out:

  top-1  — did the BEST-ranked chunk come from the right file?
  top-3  — was the right file anywhere in the retrieved chunks?
  answer — did the final answer contain the expected fact?

Why this matters: RAG systems fail silently. Retrieval can rank the
wrong chunk first (it happened in Phase 1 — see README) and the answer
may still look plausible. A test set turns "seems fine" into numbers,
and protects against regressions every time chunking or models change.
"""

import time

from query import answer, retrieve

# Each case: the question, which file the answer lives in
# (None = deliberately unanswerable), and a fact the reply must contain.
EVAL_SET = [
    {"question": "How many vacation days do full-time employees get?",
     "source": "handbook.md", "expect": "23"},
    {"question": "Which days can office staff work remotely?",
     "source": "handbook.md", "expect": "tuesday"},
    {"question": "Which dog is allowed inside, and everywhere including the roastery?",
     "source": "handbook.md", "expect": "biscuit"},
    {"question": "What is the best-selling drink and what does it cost?",
     "source": "menu.md", "expect": "6.25"},
    {"question": "During which months can I order a Solar Flare?",
     "source": "menu.md", "expect": "april"},
    {"question": "Who founded Aurora Coffee?",
     "source": "history.md", "expect": "mara ellison"},
    {"question": "How much did the founder pay for the first shop at auction?",
     "source": "history.md", "expect": "dollar"},
    {"question": "What is the wifi password?",
     "source": None, "expect": "don't know"},
]


def run_eval() -> None:
    top1_hits = top3_hits = answer_hits = answerable = 0

    print(f"Running {len(EVAL_SET)} test cases...\n")
    print(f"{'top1':>4} {'top3':>4} {'ans':>4}  question")
    print("-" * 70)

    for case in EVAL_SET:
        retrieved = retrieve(case["question"])
        reply = answer(case["question"], retrieved)
        sources = [r["source"] for r in retrieved]

        # Retrieval metrics only make sense when an answer exists in the corpus
        if case["source"] is not None:
            answerable += 1
            top1 = sources[0] == case["source"]
            top3 = case["source"] in sources
            top1_hits += top1
            top3_hits += top3
            t1, t3 = ("✓" if top1 else "✗"), ("✓" if top3 else "✗")
        else:
            t1 = t3 = "–"

        answered = case["expect"].lower() in reply.lower()
        answer_hits += answered

        print(f"{t1:>4} {t3:>4} {'✓' if answered else '✗':>4}  {case['question']}")
        time.sleep(13)  # free tier allows 5 requests/min — pace at ~1 per 13s

    print("-" * 70)
    print(f"Retrieval top-1: {top1_hits}/{answerable}"
          f"  |  top-3: {top3_hits}/{answerable}"
          f"  |  answer accuracy: {answer_hits}/{len(EVAL_SET)}")


if __name__ == "__main__":
    run_eval()
