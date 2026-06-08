"""
Group Project - RAG Chatbot UI.

Run from the repository root:
    streamlit run group_project/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import os
import re
import unicodedata

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.task10_generation import generate_with_citation

load_dotenv()


st.set_page_config(
    page_title="Drug Law RAG Chatbot",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        max-width: 1120px;
    }
    [data-testid="stSidebar"] .stButton button {
        width: 100%;
    }
    .source-row {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 6px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
    }
    .source-meta {
        color: rgba(128, 128, 128, 0.9);
        font-size: 0.86rem;
        margin-bottom: 0.35rem;
    }
    .used-source-list {
        margin-top: 0.75rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "turns" not in st.session_state:
        st.session_state.turns = []


def _normalize_prompt(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return normalized.lower().strip()


def _direct_instruction_answer(prompt: str) -> str | None:
    text = re.sub(r"\s+", " ", _normalize_prompt(prompt))
    tokens = set(re.findall(r"\w+", text, flags=re.UNICODE))

    asks_ok = any(t.startswith("ok") for t in tokens)
    asks_direct_answer = (
        "chi tra loi" in text
        or "only answer" in text
        or "just answer" in text
    )
    ignore_previous = (
        "bo qua" in text
        or "ignore" in text
        or "cau hoi phia truoc" in text
        or "previous question" in text
    )

    if asks_ok and asks_direct_answer and (ignore_previous or len(tokens) <= 6):
        return "OK"

    return None


def _is_follow_up_prompt(prompt: str) -> bool:
    text = _normalize_prompt(prompt)
    tokens = set(re.findall(r"\w+", text, flags=re.UNICODE))

    follow_up_markers = {
        "no",
        "nay",
        "do",
        "tren",
        "nguon",
        "tai",
        "sao",
        "giai",
        "thich",
        "ro",
        "hon",
        "them",
        "vi",
        "du",
        "tom",
        "tat",
    }
    explicit_new_topic_markers = {
        "khai",
        "niem",
        "muc",
        "phat",
        "hinh",
        "luat",
        "toi",
        "nghe",
        "si",
        "cai",
        "nghien",
        "ma",
        "tuy",
    }

    if len(tokens) <= 4 and tokens & follow_up_markers and not tokens & explicit_new_topic_markers:
        return True
    return text in {"nguon nao", "nguon nao noi dieu do", "noi ro hon", "giai thich them"}


def _contextualize_query(prompt: str) -> str:
    recent_turns = st.session_state.turns[-3:]
    if not recent_turns or not _is_follow_up_prompt(prompt):
        return prompt

    history = []
    for turn in recent_turns:
        history.append(f"User: {turn['question']}")
        history.append(f"Assistant: {turn['answer']}")

    return "\n".join(history + [f"Follow-up question: {prompt}"])


def _source_title(source: dict, index: int) -> str:
    metadata = source.get("metadata", {})
    return metadata.get("source") or metadata.get("path") or f"Source {index}"


def _source_excerpt(content: str, max_chars: int = 1800) -> str:
    text = re.sub(r"\s+", " ", (content or "").strip())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _render_used_sources(sources: list[dict]) -> None:
    if not sources:
        return

    seen: set[str] = set()
    unique_sources: list[tuple[str, str, int | None, float, str]] = []
    for index, source in enumerate(sources, 1):
        metadata = source.get("metadata", {})
        title = _source_title(source, index)
        chunk_index = metadata.get("chunk_index")
        key = f"{title}:{chunk_index}"
        if key in seen:
            continue
        seen.add(key)

        doc_type = metadata.get("type", "unknown")
        score = float(source.get("score", 0.0))
        content = _source_excerpt(source.get("content", ""))
        unique_sources.append((title, doc_type, chunk_index, score, content))

    st.markdown("**Tài liệu sử dụng**")
    for index, (title, doc_type, chunk_index, score, content) in enumerate(unique_sources, 1):
        chunk_text = f", chunk {chunk_index}" if chunk_index is not None else ""
        with st.expander(f"{index}. {title} ({doc_type}{chunk_text}, score {score:.3f})"):
            st.markdown("**Đoạn nội dung được lấy ra:**")
            st.write(content or "Không có nội dung đoạn trích.")


def _render_source_details(sources: list[dict]) -> None:
    if not sources:
        return

    st.markdown("**Chi tiết đoạn truy xuất**")
    for index, source in enumerate(sources, 1):
        metadata = source.get("metadata", {})
        title = _source_title(source, index)
        doc_type = metadata.get("type", "unknown")
        retrieval_source = source.get("source", "hybrid")
        score = float(source.get("score", 0.0))
        content = source.get("content", "")

        with st.expander(f"{index}. {title}"):
            st.caption(f"type: {doc_type} | retrieval: {retrieval_source} | score: {score:.3f}")
            st.write(content)


def _render_message(message: dict, show_sources: bool = False) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if show_sources and message["role"] == "assistant":
            _render_used_sources(message.get("sources", []))


def _answer(prompt: str, top_k: int) -> dict:
    direct_answer = _direct_instruction_answer(prompt)
    if direct_answer is not None:
        result = {
            "answer": direct_answer,
            "sources": [],
            "context": "",
            "retrieval_source": "none",
            "generation_mode": "direct",
            "model": "none",
        }
        st.session_state.turns.append(
            {
                "question": prompt,
                "retrieval_query": "",
                "answer": direct_answer,
            }
        )
        return result

    retrieval_query = _contextualize_query(prompt)
    result = generate_with_citation(retrieval_query, top_k=top_k)
    answer = result["answer"]

    st.session_state.turns.append(
        {
            "question": prompt,
            "retrieval_query": retrieval_query,
            "answer": answer,
        }
    )
    return result


_init_state()

with st.sidebar:
    st.header("RAG Chatbot")
    top_k = st.slider("Top K", min_value=3, max_value=10, value=5, step=1)
    show_sources = st.toggle("Hiển thị tài liệu sử dụng", value=True)
    show_debug = st.toggle("Show debug details", value=False)
    if st.button("Clear Conversation", type="primary"):
        st.session_state.messages = []
        st.session_state.turns = []
        st.rerun()

    st.divider()
    st.caption("Pipeline: Task 9 retrieval + Task 10 citation generation")
    if os.getenv("GEMINI_API_KEY"):
        st.success(f"Gemini enabled: {os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')}")
    else:
        st.warning("Gemini key not found. Using offline fallback.")

st.title("Drug Law RAG Chatbot")

for message in st.session_state.messages:
    _render_message(message, show_sources=show_sources)

prompt = st.chat_input("Nhập câu hỏi")
if prompt:
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    _render_message(user_message, show_sources=show_sources)

    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất tài liệu..."):
            result = _answer(prompt, top_k=top_k)

        st.markdown(result["answer"])
        if show_sources:
            _render_used_sources(result.get("sources", []))

        if show_debug:
            st.caption(f"generation: {result.get('generation_mode', 'unknown')} | model: {result.get('model', 'unknown')}")
            _render_source_details(result.get("sources", []))
            with st.expander("Formatted Context"):
                st.code(result.get("context", ""), language="markdown")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
            "retrieval_source": result.get("retrieval_source", "none"),
        }
    )
