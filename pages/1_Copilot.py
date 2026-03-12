import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict

import streamlit as st

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from agents.orchestrator import run_copilot  # noqa: E402


# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Baselstrasse Supply Chain Copilot",
    page_icon="📦",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Baselstrasse dark theme — matches exact palette from baselstrasse.py
#   BG      = #0f172a   (15,  23,  42)
#   ACCENT  = #63b3ed   (99, 179, 237)
#   WHITE   = #ffffff
#   MUTED   = #a0aec0   (160, 174, 192)
#   SURFACE = #1e293b   (30,  41,  59)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Global background ── */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #0f172a;
        color: #ffffff;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] * {
        color: #a0aec0 !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #63b3ed !important;
    }

    /* ── Headings ── */
    h1, h2, h3, h4 {
        color: #ffffff !important;
    }

    /* ── Accent bar on page title ── */
    .bs-title {
        border-left: 4px solid #63b3ed;
        padding-left: 14px;
        margin-bottom: 4px;
    }
    .bs-brand {
        color: #a0aec0;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding-left: 18px;
        margin-bottom: 20px;
    }

    /* ── Cards / surfaces ── */
    .bs-card {
        background: #1e293b;
        border-left: 4px solid #63b3ed;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }
    .bs-card-label {
        color: #63b3ed;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .bs-card-value {
        color: #ffffff;
        font-size: 1rem;
        line-height: 1.55;
    }
    .bs-muted {
        color: #a0aec0;
        font-size: 0.85rem;
    }

    /* ── Risk badge ── */
    .bs-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.06em;
    }
    .bs-badge-high   { background: #7f1d1d; color: #fca5a5; }
    .bs-badge-medium { background: #78350f; color: #fcd34d; }
    .bs-badge-low    { background: #14532d; color: #86efac; }
    .bs-badge-info   { background: #1e3a5f; color: #93c5fd; }

    /* ── Divider ── */
    .bs-divider {
        border: none;
        border-top: 1px solid #1e293b;
        margin: 18px 0;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background-color: #1e293b !important;
        border-radius: 8px !important;
        margin-bottom: 8px !important;
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] textarea {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: 1px solid #63b3ed !important;
        border-radius: 6px !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background-color: #1e293b !important;
        color: #63b3ed !important;
        border: 1px solid #63b3ed !important;
        border-radius: 5px !important;
        font-size: 0.82rem !important;
        transition: background 0.18s;
    }
    .stButton > button:hover {
        background-color: #63b3ed !important;
        color: #0f172a !important;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background-color: #1e293b !important;
        border: 1px solid #1e3a5f !important;
        border-radius: 6px !important;
    }

    /* ── Code block (email draft) ── */
    .stCode {
        background-color: #1e293b !important;
        border: 1px solid #1e3a5f !important;
    }

    /* ── Loading message ── */
    .bs-loading {
        color: #63b3ed;
        font-style: italic;
        font-size: 0.95rem;
        padding: 10px 0;
    }

    /* ── Metric-style row ── */
    .bs-metric-row {
        display: flex;
        gap: 10px;
        margin-bottom: 14px;
        flex-wrap: wrap;
    }
    .bs-metric {
        background: #1e293b;
        border-top: 2px solid #63b3ed;
        border-radius: 5px;
        padding: 10px 16px;
        min-width: 130px;
        flex: 1;
    }
    .bs-metric-label {
        color: #a0aec0;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .bs-metric-val {
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 700;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(
    '<div class="bs-title"><h1 style="margin:0;font-size:1.6rem;">Baselstrasse Supply Chain Copilot</h1></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="bs-brand">Baselstrasse Co. LTD &nbsp;|&nbsp; AI-Powered Supply Chain Intelligence</div>',
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None


# -----------------------------------------------------------------------------
# Loading messages (rotating while orchestrator runs)
# -----------------------------------------------------------------------------
_LOADING_MESSAGES = [
    "Fetching supply chain data...",
    "Analysing OTIF performance...",
    "Checking supplier lead times...",
    "Running forecast diagnostics...",
    "Reviewing open purchase orders...",
    "Cross-referencing evidence...",
    "Evaluating fill rate signals...",
    "Synthesising insights...",
    "Preparing your answer...",
]


def _run_with_rotating_messages(question: str) -> Dict[str, Any]:
    """Run the orchestrator in a background thread while cycling loading messages."""
    result_holder: list = [None]
    error_holder: list = [None]

    def _worker():
        try:
            result_holder[0] = run_copilot(question)
        except Exception as exc:  # noqa: BLE001
            error_holder[0] = exc

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    placeholder = st.empty()
    idx = 0
    while thread.is_alive():
        placeholder.markdown(
            f'<p class="bs-loading">⏳ {_LOADING_MESSAGES[idx % len(_LOADING_MESSAGES)]}</p>',
            unsafe_allow_html=True,
        )
        time.sleep(1.3)
        idx += 1

    thread.join()
    placeholder.empty()

    if error_holder[0] is not None:
        raise error_holder[0]
    return result_holder[0]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _risk_badge(risk_level: str) -> str:
    rl = (risk_level or "").lower()
    if "high" in rl:
        cls = "bs-badge-high"
    elif "medium" in rl or "mod" in rl:
        cls = "bs-badge-medium"
    elif "low" in rl:
        cls = "bs-badge-low"
    else:
        cls = "bs-badge-info"
    return f'<span class="bs-badge {cls}">{risk_level or "—"}</span>'


def render_result(result: Dict[str, Any]) -> None:
    summary = result.get("summary", "No summary returned.")
    causes = result.get("causes", [])
    actions = result.get("actions", [])
    email = result.get("email", "")
    evidence = result.get("evidence", {})
    risk_level = result.get("risk_level", "—")

    # ── Executive summary card ──
    st.markdown(
        f"""
        <div class="bs-card">
            <div class="bs-card-label">Executive Summary &nbsp;{_risk_badge(risk_level)}</div>
            <div class="bs-card-value">{summary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_l, col_r = st.columns(2)

    with col_l:
        # ── Root causes ──
        st.markdown(
            '<div class="bs-card-label" style="color:#63b3ed;letter-spacing:0.08em;">Likely Root Causes</div>',
            unsafe_allow_html=True,
        )
        if causes:
            for i, cause in enumerate(causes, 1):
                st.markdown(
                    f'<div class="bs-card" style="padding:10px 14px;margin-bottom:8px;">'
                    f'<span style="color:#63b3ed;font-weight:700;">{i}.</span>'
                    f'<span class="bs-card-value"> {cause}</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<p class="bs-muted">No root causes identified.</p>', unsafe_allow_html=True)

    with col_r:
        # ── Recommended actions ──
        st.markdown(
            '<div class="bs-card-label" style="color:#63b3ed;letter-spacing:0.08em;">Recommended Actions</div>',
            unsafe_allow_html=True,
        )
        if actions:
            for i, action in enumerate(actions, 1):
                st.markdown(
                    f'<div class="bs-card" style="padding:10px 14px;margin-bottom:8px;">'
                    f'<span style="color:#63b3ed;font-weight:700;">{i}.</span>'
                    f'<span class="bs-card-value"> {action}</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<p class="bs-muted">No actions recommended.</p>', unsafe_allow_html=True)

    # ── Email draft ──
    if email and email.strip():
        st.markdown('<hr class="bs-divider"/>', unsafe_allow_html=True)
        st.markdown(
            '<div class="bs-card-label" style="color:#63b3ed;letter-spacing:0.08em;">Draft Email</div>',
            unsafe_allow_html=True,
        )
        st.code(email, language="markdown")

    # ── Evidence (collapsed) ──
    if evidence:
        with st.expander("Evidence / Tool Output", expanded=False):
            st.json(evidence)


def handle_question(question: str) -> None:
    if not question.strip():
        st.warning("Please enter a question.")
        return

    st.session_state.chat_history.append({"role": "user", "content": question})

    try:
        result = _run_with_rotating_messages(question)
        st.session_state.last_result = result
        st.session_state.chat_history.append(
            {"role": "assistant", "content": result.get("summary", "Done.")}
        )
    except Exception as exc:
        msg = f"Copilot error: {exc}"
        st.session_state.chat_history.append({"role": "assistant", "content": msg})
        st.error(msg)


# -----------------------------------------------------------------------------
# Sidebar — quick prompts
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<h3 style="color:#63b3ed;border-left:3px solid #63b3ed;padding-left:10px;margin-bottom:16px;">'
        "Quick Prompts</h3>",
        unsafe_allow_html=True,
    )

    _QUICK_PROMPTS = [
        ("OTIF drop", "Why is OTIF down in Haircare this week?"),
        ("At-risk SKUs", "What SKUs are at risk next week?"),
        ("Forecast error", "Which SKUs have the worst forecast accuracy?"),
        ("Supplier delays", "Which suppliers are causing the most delays?"),
        ("Fill rate", "What is the fill rate for Haircare?"),
        ("Escalation email", "Draft an escalation email for late supplier deliveries."),
    ]

    for label, prompt in _QUICK_PROMPTS:
        if st.button(label, use_container_width=True, key=f"qp_{label}"):
            handle_question(prompt)
            st.rerun()

    st.markdown('<hr class="bs-divider"/>', unsafe_allow_html=True)

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_result = None
        st.rerun()

    st.markdown(
        '<p class="bs-muted" style="font-size:0.75rem;margin-top:20px;">'
        "Baselstrasse Co. LTD &nbsp;|&nbsp; Supply Chain Intelligence</p>",
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Main layout — conversation left, structured output right
# -----------------------------------------------------------------------------
left_col, right_col = st.columns([1, 1.5])

with left_col:
    st.markdown(
        '<div class="bs-card-label" style="color:#63b3ed;letter-spacing:0.1em;margin-bottom:10px;">'
        "Conversation</div>",
        unsafe_allow_html=True,
    )

    if not st.session_state.chat_history:
        st.markdown(
            '<div class="bs-card">'
            '<p class="bs-muted" style="margin:0;">Ask a supply chain question to begin. '
            "Try: <em>\"Why is OTIF dropping in Haircare?\"</em></p>"
            "</div>",
            unsafe_allow_html=True,
        )

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_question = st.chat_input("Ask a supply chain question...")
    if user_question:
        handle_question(user_question)
        st.rerun()

with right_col:
    st.markdown(
        '<div class="bs-card-label" style="color:#63b3ed;letter-spacing:0.1em;margin-bottom:10px;">'
        "Copilot Output</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.last_result:
        render_result(st.session_state.last_result)
    else:
        st.markdown(
            '<div class="bs-card">'
            '<p class="bs-muted" style="margin:0;">'
            "Your structured answer will appear here — executive summary, root causes, "
            "recommended actions, and draft email."
            "</p></div>",
            unsafe_allow_html=True,
        )