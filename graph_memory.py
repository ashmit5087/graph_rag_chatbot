"""
graph_memory.py
Per-user knowledge graph memory backed by NetworkX DiGraph.
"""

import os
import json
import networkx as nx
from datetime import datetime
from typing import List, Tuple, Dict, Optional


class GraphMemory:
    """
    A directed knowledge graph stored per user.
    Each edge represents a (subject, relation, object) triple.
    """

    def __init__(self, user_id: str, storage_dir: str = "graph_data"):
        self.user_id = user_id
        self.storage_dir = storage_dir
        self.graph_path = os.path.join(storage_dir, f"{user_id}_graph.json")
        self.G: nx.DiGraph = nx.DiGraph()
        os.makedirs(storage_dir, exist_ok=True)
        self._load()

    # ------------------------------------------------------------------ #
    #  Core write operations
    # ------------------------------------------------------------------ #

    def add_triple(
        self,
        subject: str,
        relation: str,
        obj: str,
        source_text: str = "",
    ) -> bool:
        """
        Add a (subject, relation, object) triple to the graph.
        Returns True if the edge was newly added, False if it already existed.
        """
        s = subject.lower().strip()
        r = relation.lower().strip().replace(" ", "_")
        o = obj.lower().strip()

        if not s or not r or not o:
            return False
        if s == o:
            return False

        now = datetime.now().isoformat()

        # Upsert nodes
        for node in (s, o):
            if not self.G.has_node(node):
                self.G.add_node(node, added=now, mention_count=0)
            self.G.nodes[node]["mention_count"] = (
                self.G.nodes[node].get("mention_count", 0) + 1
            )

        is_new = not self.G.has_edge(s, o)
        self.G.add_edge(s, o, relation=r, source=source_text, added=now)
        self._save()
        return is_new

    def add_triples(self, triples: List[Tuple[str, str, str]], source_text: str = ""):
        added = 0
        for s, r, o in triples:
            if self.add_triple(s, r, o, source_text):
                added += 1
        return added

    # ------------------------------------------------------------------ #
    #  Query operations
    # ------------------------------------------------------------------ #

    def find_entity(self, name: str) -> Optional[str]:
        """Fuzzy-ish lookup: exact → substring → None."""
        name = name.lower().strip()
        if name in self.G:
            return name
        matches = [n for n in self.G.nodes() if name in n or n in name]
        return matches[0] if matches else None

    def get_context_for_query(self, query: str, max_triples: int = 30) -> List[Tuple[str, str, str]]:
        """
        Extract entities from the query string and pull all related
        triples from the graph up to max_triples.
        """
        words = query.lower().split()
        found_entities = set()

        # Try individual words and 2-grams
        tokens = words + [" ".join(words[i:i+2]) for i in range(len(words)-1)]
        for token in tokens:
            entity = self.find_entity(token)
            if entity:
                found_entities.add(entity)

        if not found_entities:
            # Return most recently added triples as fallback
            return self.get_all_triples()[:max_triples]

        triples = set()
        for entity in found_entities:
            triples.update(self._neighborhood(entity, depth=2))

        return list(triples)[:max_triples]

    def _neighborhood(self, entity: str, depth: int = 2) -> List[Tuple[str, str, str]]:
        results = []
        visited = set()
        queue = [(entity, 0)]

        while queue:
            node, d = queue.pop(0)
            if node in visited or d > depth:
                continue
            visited.add(node)

            for nbr in self.G.successors(node):
                rel = self.G[node][nbr].get("relation", "related_to")
                results.append((node, rel, nbr))
                if d + 1 <= depth:
                    queue.append((nbr, d + 1))

            for pred in self.G.predecessors(node):
                rel = self.G[pred][node].get("relation", "related_to")
                results.append((pred, rel, node))
                if d + 1 <= depth:
                    queue.append((pred, d + 1))

        return results

    def get_all_triples(self) -> List[Tuple[str, str, str]]:
        return [
            (u, data.get("relation", "related_to"), v)
            for u, v, data in self.G.edges(data=True)
        ]

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #

    def _save(self):
        data = nx.node_link_data(self.G)
        with open(self.graph_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        if os.path.exists(self.graph_path):
            try:
                with open(self.graph_path, "r") as f:
                    data = json.load(f)
                self.G = nx.node_link_graph(data)
            except Exception:
                self.G = nx.DiGraph()

    # ------------------------------------------------------------------ #
    #  Stats & utilities
    # ------------------------------------------------------------------ #

    def stats(self) -> Dict:
        return {
            "nodes": self.G.number_of_nodes(),
            "edges": self.G.number_of_edges(),
            "entities": list(self.G.nodes()),
        }

    def clear(self):
        self.G = nx.DiGraph()
        self._save()
