"""Long-term belief graph with confidence-weighted edges.

The agent writes here autonomously during idle consolidation — not coded
rules. Each edge represents something the agent believes to be true, with
a confidence score that reflects how recently and directly it was observed.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

# Half-life per relation type. Volatile facts (person locations) decay fast;
# stable patterns (recurring needs) decay slowly.
_HALF_LIFE_SECONDS: dict[str, float] = {
    "located_in":    4  * 3600,        # 4 hours — people and objects move
    "has":           2  * 3600,        # 2 hours — someone may have put it down
    "needs":        12  * 3600,        # 12 hours — need persists but may be met
    "recurring_need": 7 * 24 * 3600,  # 7 days  — stable behavioural pattern
}
_DEFAULT_HALF_LIFE = 6 * 3600


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
        """Compact representation for LLM context.

        Shows both stored_confidence (what was observed) and effective_confidence
        (decayed by time since last observation). The agent reasons from
        effective_confidence — low values flag stale beliefs worth verifying.
        """
        return {
            "note": (
                "Beliefs with effective_confidence below ~0.3 are stale — "
                "consider verifying or lowering them."
            ),
            "beliefs": [
                {
                    "subject": e["subject"],
                    "relation": e["relation"],
                    "target": e["target"],
                    "stored_confidence": e["confidence"],
                    "effective_confidence": _effective(
                        e["confidence"], e["relation"], e.get("last_observed", "")
                    ),
                    "last_observed": e.get("last_observed", ""),
                    "observation_count": e.get("observation_count", 1),
                }
                for e in self.edges
            ],
        }


def _effective(stored: float, relation: str, last_observed: str) -> float:
    """Apply exponential decay to stored confidence based on elapsed time."""
    try:
        observed_at = datetime.fromisoformat(last_observed)
        elapsed = (datetime.now(timezone.utc) - observed_at).total_seconds()
    except (ValueError, TypeError):
        return round(stored, 3)
    half_life = _HALF_LIFE_SECONDS.get(relation, _DEFAULT_HALF_LIFE)
    factor = math.pow(0.5, elapsed / half_life)
    return round(stored * factor, 3)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
