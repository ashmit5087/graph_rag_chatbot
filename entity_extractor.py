"""
entity_extractor.py
spaCy-based entity recognition plus regex fallback.
The primary triple extraction is delegated to the LLM (in llm_client.py),
but this module provides supplementary NER and triple parsing utilities.
"""

import re
from typing import List, Tuple

# ------------------------------------------------------------------ #
#  spaCy loader with graceful fallback
# ------------------------------------------------------------------ #
try:
    import spacy

    try:
        _nlp = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess, sys
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            check=True,
        )
        _nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    _nlp = None
    SPACY_AVAILABLE = False


# ------------------------------------------------------------------ #
#  Entity extraction
# ------------------------------------------------------------------ #

def extract_entities(text: str) -> List[str]:
    """
    Returns a list of entity strings found in *text*.
    Uses spaCy NER + noun chunks when available, regex fallback otherwise.
    """
    entities = []

    if SPACY_AVAILABLE and _nlp:
        doc = _nlp(text)
        # Named entities
        entities.extend(ent.text for ent in doc.ents)
        # Short noun chunks
        entities.extend(
            chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) <= 4
        )
    else:
        # Regex: capitalised words / phrases
        entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)

    # Deduplicate, lower-case
    seen, result = set(), []
    for e in entities:
        key = e.lower().strip()
        if key not in seen and len(key) > 1:
            seen.add(key)
            result.append(key)
    return result


# ------------------------------------------------------------------ #
#  Triple parser  (parses LLM output)
# ------------------------------------------------------------------ #

_TRIPLE_PATTERNS = [
    # (subject, relation, object)
    re.compile(
        r"\(?['\"]?(?P<s>[^,\n\|\"']+)['\"]?\s*,\s*['\"]?(?P<r>[^,\n\|\"']+)['\"]?\s*,\s*['\"]?(?P<o>[^,\n\|\"'\)]+)['\"]?\)?",
        re.IGNORECASE,
    ),
    # subject -> relation -> object
    re.compile(
        r"(?P<s>[^>\n]+?)\s*-+>\s*(?P<r>[^>\n]+?)\s*-+>\s*(?P<o>[^>\n]+)",
        re.IGNORECASE,
    ),
    # subject | relation | object
    re.compile(
        r"(?P<s>[^\|\n]+?)\s*\|\s*(?P<r>[^\|\n]+?)\s*\|\s*(?P<o>[^\|\n]+)",
        re.IGNORECASE,
    ),
]


def parse_triples(llm_output: str) -> List[Tuple[str, str, str]]:
    """
    Parse (subject, relation, object) triples from raw LLM text.
    Tries several regex patterns; deduplicates results.
    """
    triples = []
    seen = set()

    for line in llm_output.strip().split("\n"):
        line = line.strip(" -•*\t")
        if not line:
            continue

        for pattern in _TRIPLE_PATTERNS:
            m = pattern.search(line)
            if m:
                s = _clean(m.group("s"))
                r = _clean(m.group("r"))
                o = _clean(m.group("o"))

                if s and r and o and len(s) < 80 and len(r) < 60 and len(o) < 80:
                    key = (s, r, o)
                    if key not in seen:
                        seen.add(key)
                        triples.append(key)
                break  # stop trying patterns for this line

    return triples


def _clean(text: str) -> str:
    text = text.strip().strip("\"'()[]").strip()
    # Remove leading numbers / bullets
    text = re.sub(r"^\d+[\.\)]\s*", "", text)
    return text.lower().strip()


# ------------------------------------------------------------------ #
#  Intent classifier  (question vs information)
# ------------------------------------------------------------------ #

_QUESTION_STARTERS = re.compile(
    r"^\s*(what|who|where|when|why|how|is|are|was|were|do|does|did|can|could|"
    r"should|would|tell me|explain|describe|give me|find|show|list|which)\b",
    re.IGNORECASE,
)


def is_question(text: str) -> bool:
    """
    Lightweight heuristic: returns True when text looks like a query
    rather than a statement providing information.
    """
    if text.strip().endswith("?"):
        return True
    if _QUESTION_STARTERS.match(text):
        return True
    # Short sentences with no verb-object clues are treated as questions
    words = text.strip().split()
    if len(words) < 4 and not any(w.lower() in ("is", "am", "are", "has", "have") for w in words):
        return True
    return False
