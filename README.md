# 🧠 Graph RAG Chatbot with Auto-Updating Memory

A Streamlit chatbot that uses a **NetworkX knowledge graph** as persistent, per-user memory.
When you tell it facts, it extracts triples and stores them. When you ask questions, it
retrieves relevant graph context and generates accurate, grounded answers.

---

## Architecture

```
User Input
    │
    ▼
Intent Classifier (LLM)
    │
    ├─── Statement ──▶ Triple Extractor (LLM + spaCy) ──▶ GraphMemory.add_triples()
    │                                                              │
    │                                                        NetworkX DiGraph
    │                                                        (persisted to JSON)
    │
    └─── Question ──▶ GraphMemory.get_context_for_query()
                              │
                        Relevant triples
                              │
                        LLM answer_with_context()
                              │
                         Final Answer
```

**Components:**
- `graph_memory.py` — NetworkX DiGraph, per-user persistence in `graph_data/`
- `entity_extractor.py` — spaCy NER + regex triple parser
- `llm_client.py` — Groq API wrapper (triple extraction, RAG answering, intent classification)
- `graph_visualizer.py` — Matplotlib graph rendering + optional pyvis interactive HTML
- `app.py` — Streamlit UI with Chat / Graph / Triples / Demo tabs

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Get a free Groq API key

Sign up at [console.groq.com](https://console.groq.com) — no credit card required.

### 3. Run the app

```bash
streamlit run app.py
```


### 4. In the sidebar

- Paste your Groq API key
- Set a User Profile name (each user gets an isolated graph)
- Click **Load / Switch**
- Start chatting!

---

## Example Inputs (adds to graph)

| # | Input |
|---|-------|
| 1 | Alice works at Google as a senior software engineer. |
| 2 | Bob is Alice's manager and lives in San Francisco. |
| 3 | Charlie studies machine learning at MIT. |
| 4 | Google was founded by Larry Page and Sergey Brin in 1998. |
| 5 | Alice and Charlie were classmates at Stanford University. |
| 6 | Bob's project is called Project Falcon and it uses Python. |
| 7 | Diana is the CEO of Acme Corp and previously worked at Tesla. |
| 8 | Project Falcon is partnered with Acme Corp for cloud services. |
| 9 | Charlie published a paper on transformer models in 2023. |
| 10 | Alice is friends with Diana and they both enjoy hiking. |
| 11 | MIT is located in Cambridge, Massachusetts. |
| 12 | Acme Corp is headquartered in New York City. |

## Example Queries (uses graph context)

| # | Query |
|---|-------|
| Q1 | Where does Alice work? |
| Q2 | Who is Bob and what is his relationship to Alice? |
| Q3 | Tell me about Project Falcon. |
| Q4 | What connections exist between Alice and Acme Corp? |
| Q5 | What do you know about Charlie's academic work? |
| Q6 | Who are the founders of Google and when was it started? |
| Q7 | What companies are connected in the graph? |

---

## Features

- ✅ Automatic triple extraction from natural language
- ✅ Per-user isolated knowledge graphs
- ✅ Graph persisted to disk (survives restarts)
- ✅ Context-grounded RAG answers (no hallucination on known facts)
- ✅ Live graph visualisation (Matplotlib static + optional pyvis interactive)
- ✅ Multi-user support via user profiles
- ✅ Searchable triple browser

---

## Reflection

### What the extraction pipeline handles well and where it breaks

The pipeline performs reliably on clean, declarative sentences following
subject-verb-object patterns. The LLM triple extractor generalises well across
professional, academic, and social relationship types, handling multi-fact
sentences by producing several triples at once. spaCy's NER adds robustness as
a fallback when LLM output is malformed.

However, the pipeline struggles with **implicit or contextual facts** (pronoun
resolution is absent), **negations** ("Alice no longer works at Google" may
incorrectly retain the old relation), **numerical/temporal data** (dates produce
inconsistent relation naming), and **highly colloquial input** where no clear
subject-predicate-object structure exists.

### One limitation of using a graph structure for this kind of memory

A knowledge graph stores facts as static, timeless triples. It cannot natively
represent that "Alice worked at Google 2019–2023 but is now at Anthropic" —
adding the new fact creates a conflicting edge rather than updating the old one.
This makes the graph unreliable for dynamic, evolving information without a
dedicated temporal-versioning layer (timestamps and validity flags on edges).
Real-world memory is inherently sequential; a flat graph treats all facts as
equally current.

---

## Tech Stack

| Layer | Library |
|-------|---------|
| Knowledge Graph | NetworkX 3.x (DiGraph) |
| LLM | Groq API (Llama 3 8B / 70B) |
| NER | spaCy `en_core_web_sm` |
| UI | Streamlit |
| Visualisation | Matplotlib + (optional) pyvis |
| Persistence | JSON (node-link format) |
