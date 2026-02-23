"""
llm_client.py
Wraps the Groq free-tier API (llama3-8b-8192) for:
  1. Extracting knowledge-graph triples from user statements.
  2. Answering questions using retrieved graph context (RAG).
  3. Classifying intent (question vs. information).
"""

from typing import List, Tuple, Dict


class LLMClient:
    DEFAULT_MODEL = "llama-3.1-8b-instant"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self.model = model
        self._api_key = api_key
        self._setup_client()

    def _setup_client(self):
        try:
            from groq import Groq
            self._client = Groq(api_key=self._api_key)
            self._provider = "groq"
        except ImportError:
            raise ImportError("groq package not installed. Run: pip install groq")

    def _call(self, prompt: str, max_tokens: int = 600, temperature: float = 0.1) -> str:
        if self._provider == "groq":
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        raise RuntimeError("No LLM provider configured.")

    # ------------------------------------------------------------------
    # Triple extraction
    # ------------------------------------------------------------------
    def extract_triples_raw(self, text: str) -> str:
        prompt = f"""You are a knowledge-graph builder. Extract factual (subject, relation, object) triples from the text below.

Rules:
- Each triple on its own line in the exact format: (subject, relation, object)
- Use short, lowercase entity names (1-4 words)
- Relations should be snake_case verbs (e.g. works_at, is_friend_of, born_in)
- Only extract clear, factual statements
- Output ONLY the triples, nothing else

Text: {text}

Triples:"""
        return self._call(prompt, max_tokens=500, temperature=0.0)

    # ------------------------------------------------------------------
    # RAG answering
    # ------------------------------------------------------------------
    def answer_with_context(
        self,
        question: str,
        context_triples: List[Tuple[str, str, str]],
        chat_history: List[Dict],
    ) -> str:
        if context_triples:
            context_str = "Knowledge Graph Context:\n" + "\n".join(
                f"  • {s}  --[{r}]-->  {o}" for s, r, o in context_triples
            )
        else:
            context_str = "Knowledge Graph Context: (empty — no relevant facts stored yet)"

        history_str = ""
        for msg in chat_history[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"

        prompt = f"""You are a helpful personal assistant with access to a knowledge graph memory.

{context_str}

Recent conversation:
{history_str}
User: {question}

Instructions:
- Answer based primarily on the knowledge graph context above.
- If the graph contains relevant information, use it explicitly.
- If the graph lacks the answer, say so clearly — do not hallucinate.
- Be concise and conversational.

Answer:"""
        return self._call(prompt, max_tokens=700, temperature=0.3)

    # ------------------------------------------------------------------
    # Intent classification
    # ------------------------------------------------------------------
    def classify_intent(self, text: str) -> str:
        """Returns 'question' or 'statement'."""
        prompt = f"""Is the following text a question/query (Q) or a statement providing information (S)?
Reply with exactly one letter: Q or S.

Text: {text}
Answer:"""
        result = self._call(prompt, max_tokens=5, temperature=0.0)
        return "question" if result.strip().upper().startswith("Q") else "statement"