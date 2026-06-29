"""
app.py
-------
Streamlit frontend for the Company Assistant chatbot.

Run with:
    streamlit run app.py

The UI:
  - Accepts user queries via a text input
  - Calls the FastAPI /chat endpoint
  - Displays the AI-generated answer and retrieved source passages
  - Maintains a chat history for the session
"""

import time

import requests
import streamlit as st


# Configuration


API_BASE_URL = "http://localhost:8000"
CHAT_ENDPOINT = f"{API_BASE_URL}/chat"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"
DOCS_ENDPOINT = f"{API_BASE_URL}/documents"


# Page configuration


st.set_page_config(
    page_title="Company Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom CSS — clean, professional dark theme


st.markdown(
    """
    <style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid #334155;
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    /* ── Main area ── */
    .main .block-container {
        padding-top: 2rem;
        max-width: 900px;
    }

    /* ── Chat message cards ── */
    .user-bubble {
        background: linear-gradient(135deg, #1d4ed8, #2563eb);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 6px 0 6px 15%;
        font-size: 0.95rem;
        line-height: 1.6;
        box-shadow: 0 2px 12px rgba(37,99,235,0.3);
    }

    .assistant-bubble {
        background: #f8fafc;
        color: #0f172a;
        padding: 14px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 6px 15% 6px 0;
        font-size: 0.95rem;
        line-height: 1.7;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    /* ── Source cards ── */
    .source-card {
        background: #f1f5f9;
        border-left: 3px solid #3b82f6;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.82rem;
        color: #475569;
        font-family: 'JetBrains Mono', monospace;
    }
    .source-label {
        font-weight: 600;
        color: #1e40af;
        margin-bottom: 4px;
    }
    .score-badge {
        display: inline-block;
        background: #dbeafe;
        color: #1d4ed8;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-left: 6px;
    }

    /* ── Status indicator ── */
    .status-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-dot.green { background: #22c55e; }
    .status-dot.red   { background: #ef4444; }

    /* ── Metric cards ── */
    div[data-testid="metric-container"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 12px;
    }

    /* ── Input box ── */
    textarea {
        border-radius: 12px !important;
        border: 1.5px solid #cbd5e1 !important;
        font-family: 'Inter', sans-serif !important;
    }
    textarea:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.15) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Session state initialisation


if "messages" not in st.session_state:
    st.session_state.messages = []          # list of {"role": "user"|"assistant", "content": ..., "sources": [...], "latency": float}

if "api_status" not in st.session_state:
    st.session_state.api_status = None      # "ok" | "error"

if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []



# Helper functions


def check_api_health() -> bool:
    """Ping the backend and cache status in session state."""
    try:
        r = requests.get(HEALTH_ENDPOINT, timeout=5)
        if r.status_code == 200:
            data = r.json()
            st.session_state.api_status = data
            return True
    except Exception:
        pass
    st.session_state.api_status = None
    return False


def fetch_indexed_docs() -> list[str]:
    """Get the list of document sources from the backend."""
    try:
        r = requests.get(DOCS_ENDPOINT, timeout=5)
        if r.status_code == 200:
            return r.json().get("sources", [])
    except Exception:
        pass
    return []


def call_chat_api(query: str, top_k: int) -> dict | None:
    """POST to /chat and return the JSON response, or None on error."""
    try:
        r = requests.post(
            CHAT_ENDPOINT,
            json={"query": query, "top_k": top_k},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the API. Make sure the FastAPI server is running on port 8000.")
    except requests.exceptions.Timeout:
        st.error("⏱️ The request timed out. The model might be loading — try again in a moment.")
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API error: {e.response.status_code} — {e.response.text}")
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
    return None



# Sidebar


with st.sidebar:
    st.markdown("## 🤖 Company Assistant")
    st.markdown("*Powered by Claude + FAISS RAG*")
    st.divider()

    # API Health
    st.markdown("###  API Status")
    if st.button("Check Connection", use_container_width=True):
        ok = check_api_health()
        st.session_state.indexed_docs = fetch_indexed_docs() if ok else []

    status = st.session_state.api_status
    if status:
        st.markdown(
            f'<span class="status-dot green"></span> **Connected** — {status.get("index_size", "?")} passages indexed',
            unsafe_allow_html=True,
        )
        st.caption(f"Embedding model: `{status.get('model', '?')}`")
    else:
        st.markdown(
            '<span class="status-dot red"></span> *Not connected — click above*',
            unsafe_allow_html=True,
        )

    st.divider()

    # Settings
    st.markdown("###  Settings")
    top_k = st.slider(
        "Retrieved passages (top-k)",
        min_value=1, max_value=8, value=4,
        help="How many document chunks to retrieve for each query",
    )

    st.divider()

    # Indexed documents
    st.markdown("###  Indexed Documents")
    if st.session_state.indexed_docs:
        for doc in st.session_state.indexed_docs:
            st.markdown(f"- {doc}")
    else:
        st.caption("Connect to the API to see indexed documents.")

    st.divider()

    # Example questions
    st.markdown("###  Try asking …")
    example_questions = [
        "How many days of annual leave do I get?",
        "What is the commission structure for sales reps?",
        "How do I submit an expense report?",
        "What is the work-from-home policy?",
        "What are the password requirements?",
        "When are performance reviews conducted?",
        "What is the Enterprise pricing plan?",
        "What is the company's return policy?",   # not in docs — tests graceful failure
    ]
    for q in example_questions:
        if st.button(q, use_container_width=True, key=f"ex_{q[:20]}"):
            st.session_state["prefill_query"] = q

    st.divider()

    # Clear history
    if st.button(" Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()



# Main area — header


col_title, col_metrics = st.columns([2, 1])

with col_title:
    st.markdown("#  Company Assistant")
    st.markdown("Ask me anything about HR, Sales, Finance, IT, or company policies.")

with col_metrics:
    total_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
    st.metric("Questions asked", total_msgs)

st.divider()



# Chat history display


chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">👤 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="assistant-bubble">🤖 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )

            # Show source documents in an expander
            if msg.get("sources"):
                with st.expander(f"📄 {len(msg['sources'])} source passage(s) retrieved  |  ⚡ {msg.get('latency', 0):.0f} ms"):
                    for i, src in enumerate(msg["sources"], 1):
                        score_pct = round(src["score"] * 100, 1)
                        st.markdown(
                            f"""
                            <div class="source-card">
                                <div class="source-label">
                                    [{i}] {src["source"]}
                                    <span class="score-badge">score: {score_pct}%</span>
                                </div>
                                {src["text"][:350]}{"…" if len(src["text"]) > 350 else ""}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        st.markdown("<br>", unsafe_allow_html=True)



# Input area


# Handle pre-filled query from sidebar buttons
prefill = st.session_state.pop("prefill_query", "")

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_area(
        "Your question",
        value=prefill,
        placeholder="e.g. How many sick days am I entitled to?",
        height=90,
        label_visibility="collapsed",
    )
    submit = st.form_submit_button("Send ↗", use_container_width=False, type="primary")


if submit and user_input.strip():
    query = user_input.strip()

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": query})

    # Call API with a spinner
    with st.spinner(" Searching documents and generating answer …"):
        result = call_chat_api(query, top_k)

    if result:
        answer = result.get("answer", "No answer returned.")
        sources = result.get("sources", [])
        latency = result.get("latency_ms", 0)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "latency": latency,
        })

    st.rerun()



# Empty state prompt


if not st.session_state.messages:
    st.markdown(
        """
        <div style="text-align:center; padding: 60px 20px; color: #94a3b8;">
            <div style="font-size: 3rem;">🤖</div>
            <h3 style="color:#64748b; margin-top: 12px;">Ask your company assistant anything</h3>
            <p>Try one of the example questions in the sidebar, or type your own below.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
