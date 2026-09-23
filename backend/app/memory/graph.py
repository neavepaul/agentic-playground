"""Long-term belief graph with confidence-weighted edges.

The agent writes here autonomously during idle consolidation — not coded
rules. Each edge represents something the agent believes to be true, with
a confidence score that reflects how recently and directly it was observed.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class BeliefGraph:
    """In-memory belief graph, optionally persisted to JSON.

    Nodes are entities (people, objects, rooms).
    Edges are relationships (located_in, needs, has, recurring_need) with
    a confidence score and observation count that the agent updates itself.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self.nodes: dict[str, dict] = {}
        self.edges: list[dict] = []

    @classmethod
    def load(cls, path: Path) -> BeliefGraph:
        g = cls(path)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                g.nodes = data.get("nodes", {})
                g.edges = data.get("edges", [])
            except Exception:
                pass
        return g

    def save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps({"nodes": self.nodes, "edges": self.edges}, indent=2),
            encoding="utf-8",
        )

    def upsert_node(self, entity_id: str, entity_type: str) -> None:
        self.nodes.setdefault(entity_id, {"id": entity_id, "type": entity_type})

    def upsert_edge(self, subject: str, relation: str, target: str,
                    confidence: float, reason: str = "") -> None:
        for edge in self.edges:
            if (edge["subject"] == subject
                    and edge["relation"] == relation
                    and edge["target"] == target):
                edge["confidence"] = round(confidence, 3)
                edge["last_observed"] = _now()
                edge["observation_count"] = edge.get("observation_count", 0) + 1
                edge["reason"] = reason
                return
        self.edges.append({
            "subject": subject,
            "relation": relation,
            "target": target,
            "confidence": round(confidence, 3),
            "last_observed": _now(),
            "observation_count": 1,
            "reason": reason,
        })

    def remove_edge(self, subject: str, relation: str, target: str) -> None:
        self.edges = [
            e for e in self.edges
            if not (e["subject"] == subject
                    and e["relation"] == relation
                    and e["target"] == target)
        ]

    def prompt(self) -> dict:
        """Compact representation for LLM context."""
        return {
            "note": "Agent's inferred beliefs — confidence reflects recency and directness of observation.",
            "beliefs": [
                {k: v for k, v in e.items() if k != "reason"}
                for e in self.edges
            ],
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
