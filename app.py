"""DocMind — chat UI.

Run:   streamlit run app.py

A thin layer over the same pipeline the CLI uses: every question goes
through query.retrieve() and query.answer(). The UI's job is chat history
and showing WHICH chunks each answer came from — the receipts.
"""

from pathlib import Path

import streamlit as st

from query import answer, retrieve

st.set_page_config(page_title="DocMind", page_icon="🧠")


# ── Sidebar: what am I looking at? ───────────────────────────────────────────
with st.sidebar:
    st.title("🧠 DocMind")
    st.markdown(
        "Chat with your documents. Answers come **only** from the files "
        "below — with sources shown for every reply."
    )
    st.subheader("Indexed documents")
    for path in sorted((Path(__file__).parent / "docs").glob("*.md")):
        st.caption(f"📄 {path.name}")
    st.divider()
    st.caption(
        "Pipeline: question → embedding → vector search (ChromaDB) → "
        "Gemini answers from the retrieved chunks."
    )


def show_sources(sources: list[dict]) -> None:
    """Render the retrieved chunks — the evidence behind an answer."""
    with st.expander(f"📚 Sources ({len(sources)} chunks)"):
        for s in sources:
            st.markdown(f"**{s['source']}** — similarity `{s['score']:.3f}`")
            st.text(s["text"][:400])
            st.divider()


# ── Chat history (Streamlit reruns this script on every interaction, so
#    the conversation lives in session_state) ─────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"])


# ── Handle a new question ────────────────────────────────────────────────────
if question := st.chat_input("Ask about your documents…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your documents…"):
            sources = retrieve(question)
            reply = answer(question, sources)
        st.markdown(reply)
        show_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "sources": sources}
    )
