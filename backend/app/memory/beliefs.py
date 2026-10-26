"""Beliefs computed from Kip's persistent tallies. No model calls and nothing stored.

Each belief has a stored confidence (how strong the evidence is) and an effective
confidence (decayed by time since it was last observed), the same shape the belief graph
used, so the Beliefs tab and the idle mind read it unchanged. A belief with no known
observation time is treated as old rather than fresh.
"""
from __future__ import annotations

from datetime import datetime, timezone

HALF_LIFE_SECONDS = {
    "located_in": 4 * 3600,            # people and objects move
    "needs": 12 * 3600,                # a need persists but may be met
    "recurring_need": 7 * 24 * 3600,
}
UNDATED_CEILING = 0.3  # "secondhand or old"
TYPICAL_MIN_SIGHTINGS = 3
TYPICAL_MIN_SHARE = 0.6
LAST_SEEN_CONFIDENCE = 0.7


def effective(stored: float, relation: str, observed_at: str | None, now: datetime | None = None) -> float:
    if not observed_at:
        return round(min(stored, UNDATED_CEILING), 3)
    try:
        elapsed = ((now or datetime.now(timezone.utc)) - datetime.fromisoformat(observed_at)).total_seconds()
    except (TypeError, ValueError):
        return round(min(stored, UNDATED_CEILING), 3)
    return round(stored * 0.5 ** (max(0.0, elapsed) / HALF_LIFE_SECONDS[relation]), 3)


def _typical_confidence(share: float) -> float:
    """60% of sightings in one room -> 0.5, rising to 0.8 at 90% or more."""
    return round(min(0.8, 0.5 + (share - TYPICAL_MIN_SHARE) / 0.3 * 0.3), 3)


def _recurring_confidence(count: int) -> float:
    return 0.9 if count >= 5 else 0.8 if count >= 3 else 0.65


def derive_beliefs(memory, now: datetime | None = None) -> dict:
    found: dict[tuple[str, str, str], dict] = {}

    def add(subject, relation, target, confidence, observed_at, count, reason, pattern=False):
        # A pattern (a habit, a repeated need) is built from counts that only grow, so it
        # does not go stale the way a single sighting does.
        belief = {"subject": subject, "relation": relation, "target": target,
                  "stored_confidence": confidence,
                  "effective_confidence": confidence if pattern else effective(confidence, relation, observed_at, now),
                  "last_observed": observed_at or "", "observation_count": count, "reason": reason}
        key = (subject, relation, target)
        if key not in found or belief["effective_confidence"] > found[key]["effective_confidence"]:
            found[key] = belief

    for pid, person in memory.people.items():
        total = sum(person.location_tally.values())
        if total >= TYPICAL_MIN_SIGHTINGS:
            room, seen = max(person.location_tally.items(), key=lambda item: item[1])
            share = seen / total
            if share >= TYPICAL_MIN_SHARE:
                observed = person.last_seen_at if person.last_seen_room == room else None
                add(pid, "located_in", room, _typical_confidence(share), observed, seen,
                    f"Seen in {room} {seen} of {total} times.", pattern=True)
        if person.last_seen_room:
            add(pid, "located_in", person.last_seen_room, LAST_SEEN_CONFIDENCE, person.last_seen_at,
                person.location_tally.get(person.last_seen_room, 1), f"Last seen in {person.last_seen_room}.")
        for thing, count in person.delivery_counts.items():
            if count >= 2:
                add(pid, "recurring_need", thing, _recurring_confidence(count), None, count,
                    f"Delivered {thing} to {pid} {count} times.", pattern=True)
        for thing in person.known_needs:
            if person.delivery_counts.get(thing, 0) < 2:
                add(pid, "needs", thing, 0.5, None, 1, f"Delivered {thing} to {pid} once.")
    for oid, item in memory.objects.items():
        if item.last_seen_room:
            add(oid, "located_in", item.last_seen_room, LAST_SEEN_CONFIDENCE, item.last_seen_at, 1,
                f"Last seen in {item.last_seen_room}.")
    return {"note": "Computed from what I have observed. Beliefs with effective_confidence below "
                    "~0.3 are stale: worth verifying before acting on them.",
            "beliefs": sorted(found.values(), key=lambda b: (-b["effective_confidence"], b["subject"]))}
