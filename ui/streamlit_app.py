"""
Interface Streamlit pour l'assistant juridique IA.

Lancer avec :
    streamlit run ui/streamlit_app.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st

from app.rag.pipeline import ask_question
from app.core.logging import setup_logging

setup_logging()

st.set_page_config(
    page_title="Assistant Juridique IA — Droit du Travail",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Custom CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: #000000;
    }

    /* ── Hide Streamlit defaults ── */
    #MainMenu, header, footer { visibility: hidden; }
    .block-container { padding-top: 1rem; padding-bottom: 0; }

    /* ── Header ── */
    .app-header {
        text-align: center;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        background: rgba(20, 20, 20, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        backdrop-filter: blur(12px);
    }
    .app-header h1 {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .app-header p {
        color: #a0a0a0;
        font-size: 0.9rem;
        font-weight: 300;
        letter-spacing: 0.5px;
    }

    /* ── Chat area ── */
    .stChatMessage {
        border-radius: 12px !important;
        margin-bottom: 0.5rem !important;
        animation: fadeIn 0.4s ease-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* User messages */
    [data-testid="stChatMessage"][aria-label="user"] {
        background: rgba(40, 40, 40, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    /* Assistant messages */
    [data-testid="stChatMessage"][aria-label="assistant"] {
        background: rgba(20, 20, 20, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
    }

    /* ── Chat input ── */
    .stChatInput > div {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        background: rgba(20, 20, 20, 0.8) !important;
        transition: border-color 0.3s ease;
    }
    .stChatInput > div:focus-within {
        border-color: rgba(255, 255, 255, 0.6) !important;
        box-shadow: 0 0 15px rgba(255, 255, 255, 0.1) !important;
    }

    /* ── Sources panel ── */
    .sources-panel {
        background: rgba(15, 15, 15, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.2rem;
        backdrop-filter: blur(8px);
        max-height: 600px;
        overflow-y: auto;
    }
    .sources-panel::-webkit-scrollbar { width: 4px; }
    .sources-panel::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 4px;
    }
    .sources-panel::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.4);
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: rgba(25, 25, 25, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #a0a0a0 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ── Section titles ── */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid rgba(255, 255, 255, 0.2);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Source item ── */
    .source-item {
        background: rgba(30, 30, 30, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 0.8rem;
        margin-bottom: 0.6rem;
        transition: border-color 0.2s ease;
    }
    .source-item:hover {
        border-color: rgba(255, 255, 255, 0.3);
    }
    .source-title {
        color: #ffffff;
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 0.3rem;
    }
    .source-article {
        color: #cccccc;
        font-size: 0.8rem;
        font-weight: 500;
        margin-bottom: 0.3rem;
    }
    .source-preview {
        color: #a0a0a0;
        font-size: 0.75rem;
        line-height: 1.4;
        border-left: 2px solid rgba(255, 255, 255, 0.3);
        padding-left: 0.6rem;
        margin-top: 0.3rem;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: rgba(30, 30, 30, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        background: rgba(50, 50, 50, 0.8) !important;
        border-color: rgba(255, 255, 255, 0.4) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(255, 255, 255, 0.1) !important;
    }

    /* ── Spinner ── */
    .stSpinner > div > div {
        border-top-color: #ffffff !important;
    }

    /* ── Divider ── */
    hr {
        border-color: rgba(255, 255, 255, 0.1) !important;
    }

    /* ── Info message ── */
    .stAlert {
        background: rgba(40, 40, 40, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        color: #cccccc !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header 
st.markdown(
    """
    <div class="app-header">
        <h1>Assistant Juridique IA</h1>
        <p>Droit du Travail Français · Code du Travail · Conventions Collectives · Sécurité Sociale</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "sources_html" not in st.session_state:
    st.session_state.sources_html = ""
if "last_latency" not in st.session_state:
    st.session_state.last_latency = 0.0

# Layout
col_chat, col_sources = st.columns([2, 1])
#  Chat Column 
with col_chat:
    st.markdown(
        '<div class="section-title"> Conversation</div>',
        unsafe_allow_html=True,
    )

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Posez votre question juridique...")

    if user_input:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner(" Analyse juridique en cours..."):
                response = ask_question(
                    question=user_input,
                    chat_history=st.session_state.chat_history[:-1],
                )
            st.markdown(response.answer)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": response.answer}
        )
        st.session_state.last_latency = response.latency_ms

        if response.sources:
            html_parts = []
            for i, src in enumerate(response.sources, 1):
                preview = ""
                if src.content_preview:
                    preview_text = src.content_preview[:200].replace("<", "&lt;")
                    preview = f'<div class="source-preview">{preview_text}...</div>'
                html_parts.append(
                    f'<div class="source-item">'
                    f'  <div class="source-title">{i}. {src.source}</div>'
                    f'  <div class="source-article"> {src.article or "N/A"}</div>'
                    f'  {preview}'
                    f'</div>'
                )
            st.session_state.sources_html = "\n".join(html_parts)
        else:
            st.session_state.sources_html = ""

        st.rerun()

#  Sources Column
with col_sources:
    st.markdown(
        '<div class="section-title"> Sources & Métadonnées</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.last_latency > 0:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            latency = st.session_state.last_latency
            label = f"{latency / 1000:.1f}s" if latency >= 1000 else f"{latency:.0f}ms"
            st.metric(" Latence", label)
        with col_m2:
            n_msgs = len(
                [m for m in st.session_state.chat_history if m["role"] == "user"]
            )
            st.metric("Questions", str(n_msgs))

    if st.session_state.sources_html:
        st.markdown(
            f'<div class="sources-panel">{st.session_state.sources_html}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Les sources juridiques apparaîtront ici après votre question.")

    st.divider()
    if st.button(" Effacer l'historique", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.sources_html = ""
        st.session_state.last_latency = 0.0
        st.rerun()
