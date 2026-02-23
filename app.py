"""
app.py  —  Graph RAG Chatbot with Auto-Updating Memory
Run with:  streamlit run app.py
"""

import os
import streamlit as st
from graph_memory import GraphMemory
from entity_extractor import extract_entities, parse_triples, is_question
from llm_client import LLMClient
from graph_visualizer import draw_graph

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Graph RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS  —  dark neural aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    background-color: #080c14;
    color: #c8d8f0;
}
.stApp { background: linear-gradient(135deg, #080c14 0%, #0d1826 100%); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0a1020;
    border-right: 1px solid #1e3050;
}

/* Chat bubbles */
.user-bubble {
    background: linear-gradient(135deg, #1a3a6e, #0f2248);
    border: 1px solid #2a5298;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 16px;
    margin: 8px 0 8px 60px;
    font-size: 0.92rem;
    line-height: 1.6;
    box-shadow: 0 2px 8px rgba(42,82,152,0.3);
}
.bot-bubble {
    background: linear-gradient(135deg, #0f1e30, #162840);
    border: 1px solid #1e4070;
    border-radius: 16px 16px 16px 4px;
    padding: 12px 16px;
    margin: 8px 60px 8px 0;
    font-size: 0.92rem;
    line-height: 1.6;
    box-shadow: 0 2px 8px rgba(15,30,50,0.5);
}
.bubble-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    opacity: 0.55;
    margin-bottom: 4px;
    letter-spacing: 0.08em;
}
.triple-badge {
    display: inline-block;
    background: #0a2040;
    border: 1px solid #1a4080;
    border-radius: 8px;
    padding: 3px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    margin: 2px 3px;
    color: #7abaff;
}
.stat-box {
    background: #0a1628;
    border: 1px solid #1a3050;
    border-radius: 10px;
    padding: 10px 14px;
    text-align: center;
    margin: 4px 0;
}
.stat-num { font-size: 1.8rem; font-weight: 700; color: #4F8EF7; }
.stat-label { font-size: 0.72rem; color: #6688aa; font-family: 'JetBrains Mono', monospace; }
h1 { font-weight: 700; color: #e8f4ff; letter-spacing: -0.02em; }
.stButton>button {
    background: linear-gradient(135deg, #1a3a6e, #0f2248);
    color: #c8d8f0;
    border: 1px solid #2a5298;
    border-radius: 8px;
    font-family: 'Sora', sans-serif;
    transition: all 0.2s;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #2a5298, #1a3a6e);
    border-color: #4F8EF7;
    color: #fff;
}
.stTextInput>div>div>input, .stSelectbox>div>div {
    background: #0a1628 !important;
    border: 1px solid #1e3050 !important;
    color: #c8d8f0 !important;
    border-radius: 8px !important;
}
.section-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #4F8EF7;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 8px;
    border-bottom: 1px solid #1e3050;
    padding-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_session(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def init_session():
    get_session("chat_history", [])
    get_session("llm", None)
    get_session("memory", None)
    get_session("user_id", "user_default")
    get_session("last_triples", [])
    get_session("api_key", "")


init_session()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Graph RAG Chatbot")
    st.markdown('<div class="section-title">Configuration</div>', unsafe_allow_html=True)

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=st.session_state.api_key,
        placeholder="gsk_...",
        help="Get a free key at console.groq.com",
    )
    if api_key:
        st.session_state.api_key = api_key

    user_id = st.text_input(
        "User Profile",
        value=st.session_state.user_id,
        placeholder="alice / bob / ...",
        help="Each user gets a separate knowledge graph",
    )

    model_choice = st.selectbox(
        "Model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "gemma2-9b-it"],
        index=0,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Load / Switch", use_container_width=True):
            if api_key and user_id:
                st.session_state.user_id = user_id
                st.session_state.llm = LLMClient(api_key, model=model_choice)
                st.session_state.memory = GraphMemory(user_id)
                st.session_state.chat_history = []
                st.session_state.last_triples = []
                st.success(f"Loaded profile: {user_id}")
            else:
                st.error("Enter API key + user ID first.")
    with col2:
        if st.button("🗑️ Clear Graph", use_container_width=True):
            if st.session_state.memory:
                st.session_state.memory.clear()
                st.session_state.chat_history = []
                st.success("Graph cleared!")

    # Graph stats
    if st.session_state.memory:
        stats = st.session_state.memory.stats()
        st.markdown('<div class="section-title" style="margin-top:16px">Graph Stats</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="stat-box"><div class="stat-num">{stats["nodes"]}</div><div class="stat-label">Entities</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-box"><div class="stat-num">{stats["edges"]}</div><div class="stat-label">Relations</div></div>', unsafe_allow_html=True)

        if stats["entities"]:
            st.markdown('<div class="section-title" style="margin-top:12px">Known Entities</div>', unsafe_allow_html=True)
            entity_str = " · ".join(stats["entities"][:30])
            st.caption(entity_str)

    st.markdown("---")
    st.markdown("""
**How it works:**
1. Type a *fact* → extracted & stored in your graph  
2. Ask a *question* → graph is searched, answer generated  
3. Each user has an isolated graph  

**Built with:** NetworkX · spaCy · Groq (Llama 3) · Streamlit
""")


# ─────────────────────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 🧠 Graph RAG Chatbot")
st.caption("Tell me facts. Ask me questions. I remember everything in a knowledge graph.")

tab_chat, tab_graph, tab_triples, tab_demo = st.tabs(
    ["💬 Chat", "🕸️ Graph", "🗂️ All Triples", "🎯 Demo Examples"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB: Chat
# ─────────────────────────────────────────────────────────────────────────────
with tab_chat:
    # Render chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            role = msg["role"]
            content = msg["content"]
            meta = msg.get("meta", {})

            if role == "user":
                st.markdown(
                    f'<div class="user-bubble"><div class="bubble-label">👤 YOU</div>{content}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="bot-bubble"><div class="bubble-label">🤖 ASSISTANT</div>{content}</div>',
                    unsafe_allow_html=True,
                )
                # Show triples if any were extracted
                if meta.get("triples"):
                    badge_html = "".join(
                        f'<span class="triple-badge">({s}, {r}, {o})</span>'
                        for s, r, o in meta["triples"]
                    )
                    st.markdown(
                        f'<div style="margin: -4px 0 8px 0; padding-left: 4px">📌 Stored: {badge_html}</div>',
                        unsafe_allow_html=True,
                    )
                if meta.get("context_triples"):
                    with st.expander("🔍 Graph context used", expanded=False):
                        for s, r, o in meta["context_triples"]:
                            st.caption(f"• {s}  —[{r}]→  {o}")

    st.markdown("---")

    # Input area
    if not st.session_state.memory:
        st.warning("⚠️ Enter your Groq API key, set a user profile, and click **Load / Switch** in the sidebar to start.")
    else:
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Your message",
                placeholder='e.g. "Alice works at Google" or "Where does Alice work?"',
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Send →", use_container_width=False)

        if submitted and user_input.strip():
            llm: LLMClient = st.session_state.llm
            mem: GraphMemory = st.session_state.memory

            user_msg = user_input.strip()
            st.session_state.chat_history.append({"role": "user", "content": user_msg})

            with st.spinner("Thinking…"):
                # Classify intent
                intent = llm.classify_intent(user_msg)
                # Also use local heuristic as tie-breaker
                if is_question(user_msg):
                    intent = "question"

                meta = {}

                if intent == "statement":
                    # ── Extract triples and store in graph ──
                    raw = llm.extract_triples_raw(user_msg)
                    triples = parse_triples(raw)

                    # Fallback: spaCy entities as loose triples
                    if not triples:
                        entities = extract_entities(user_msg)
                        for i in range(len(entities) - 1):
                            triples.append((entities[i], "related_to", entities[i + 1]))

                    added = mem.add_triples(triples, source_text=user_msg)
                    st.session_state.last_triples = triples
                    meta["triples"] = triples

                    if triples:
                        reply = (
                            f"Got it! I extracted **{len(triples)}** triple(s) and added "
                            f"**{added}** new fact(s) to your knowledge graph."
                        )
                    else:
                        reply = (
                            "I recorded your message, but couldn't extract structured triples. "
                            "Try stating facts like 'X is Y' or 'X works at Y'."
                        )

                else:
                    # ── RAG: retrieve context → answer ──
                    context = mem.get_context_for_query(user_msg)
                    meta["context_triples"] = context
                    reply = llm.answer_with_context(
                        user_msg, context, st.session_state.chat_history
                    )

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": reply,
                "meta": meta,
            })
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB: Graph visualisation
# ─────────────────────────────────────────────────────────────────────────────
with tab_graph:
    if st.session_state.memory:
        mem: GraphMemory = st.session_state.memory
        stats = mem.stats()

        if stats["nodes"] == 0:
            st.info("The knowledge graph is empty. Add some facts in the Chat tab first.")
        else:
            st.markdown(f"**{stats['nodes']} entities** · **{stats['edges']} relations** · user: `{st.session_state.user_id}`")

            png = draw_graph(mem.G, title=f"Knowledge Graph — {st.session_state.user_id}")
            if png:
                st.image(png, use_container_width=True)

            if st.button("↻ Refresh graph"):
                st.rerun()
    else:
        st.info("Load a user profile first (sidebar).")


# ─────────────────────────────────────────────────────────────────────────────
# TAB: All triples
# ─────────────────────────────────────────────────────────────────────────────
with tab_triples:
    if st.session_state.memory:
        triples = st.session_state.memory.get_all_triples()
        if not triples:
            st.info("No triples stored yet.")
        else:
            st.markdown(f"**{len(triples)} triples** in the graph:")
            search = st.text_input("Filter triples", placeholder="Search entity or relation…")
            for s, r, o in triples:
                row = f"({s}, {r}, {o})"
                if not search or search.lower() in row.lower():
                    st.markdown(
                        f'<span class="triple-badge">({s})</span>'
                        f'<span style="color:#888;font-family:monospace;font-size:0.8rem"> ──[{r}]──▶ </span>'
                        f'<span class="triple-badge">({o})</span>',
                        unsafe_allow_html=True,
                    )
    else:
        st.info("Load a user profile first (sidebar).")


# ─────────────────────────────────────────────────────────────────────────────
# TAB: Demo examples
# ─────────────────────────────────────────────────────────────────────────────
with tab_demo:
    st.markdown("### 📥 Example inputs that add information to the graph")
    st.markdown("Paste any of these into the chat to populate the graph:")

    examples_info = [
        ("1", "Alice works at Google as a senior software engineer."),
        ("2", "Bob is Alice's manager and lives in San Francisco."),
        ("3", "Charlie studies machine learning at MIT."),
        ("4", "Google was founded by Larry Page and Sergey Brin in 1998."),
        ("5", "Alice and Charlie were classmates at Stanford University."),
        ("6", "Bob's project is called Project Falcon and it uses Python."),
        ("7", "Diana is the CEO of Acme Corp and previously worked at Tesla."),
        ("8", "Project Falcon is partnered with Acme Corp for cloud services."),
        ("9", "Charlie published a paper on transformer models in 2023."),
        ("10", "Alice is friends with Diana and they both enjoy hiking."),
        ("11", "MIT is located in Cambridge, Massachusetts."),
        ("12", "Acme Corp is headquartered in New York City."),
    ]

    for num, text in examples_info:
        col1, col2 = st.columns([0.06, 0.94])
        with col1:
            st.markdown(f"**#{num}**")
        with col2:
            st.code(text, language=None)

    st.markdown("---")
    st.markdown("### ❓ Example queries that use the stored graph context")

    examples_query = [
        ("Q1", "Where does Alice work?"),
        ("Q2", "Who is Bob and what is his relationship to Alice?"),
        ("Q3", "Tell me about Project Falcon."),
        ("Q4", "What connections exist between Alice and Acme Corp?"),
        ("Q5", "What do you know about Charlie's academic work?"),
        ("Q6", "Who are the founders of Google and when was it started?"),
        ("Q7", "What companies are connected in the graph?"),
    ]

    for num, text in examples_query:
        col1, col2 = st.columns([0.08, 0.92])
        with col1:
            st.markdown(f"**{num}**")
        with col2:
            st.code(text, language=None)

    st.markdown("---")
    st.markdown("### 📝 Reflection")
    st.markdown("""
**What the pipeline handles well and where it breaks:**

The extraction pipeline performs reliably on clean, declarative sentences that follow 
subject-verb-object patterns (e.g. *"Alice works at Google"*, *"Bob is Alice's manager"*). 
The LLM triple extractor generalises well across professional, academic, and social 
relationship types, and can handle multi-fact sentences by producing several triples at once. 
spaCy's NER layer adds robustness as a fallback when the LLM output is malformed.

However, the pipeline struggles with **implicit or contextual facts** (e.g. *"She used to be 
his boss"* — pronoun resolution is absent), **negations** (*"Alice no longer works at Google"* 
might incorrectly store the old relation), **numerical/temporal data** (*"Alice joined in 2019"* 
produces inconsistent relation naming), and **highly colloquial input** where no clear 
subject-predicate-object structure exists. Complex compound sentences with conjunctions 
sometimes yield partial or duplicated triples.

**One limitation of graph-based memory:**

A key limitation is the **lack of temporal and contextual ordering**. A knowledge graph stores 
facts as static, timeless triples. It cannot natively represent that *"Alice worked at Google 
from 2019–2023 but is now at Anthropic"* — adding the new fact simply creates a conflicting 
edge rather than updating the old one. This makes the graph unreliable for dynamic, evolving 
personal information without a dedicated temporal-versioning layer (e.g. attaching timestamps 
and validity flags to each edge). Real-world memory is inherently sequential; a flat graph 
treats all facts as equally current.
""")