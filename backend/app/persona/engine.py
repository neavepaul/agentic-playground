import json
from typing import Callable

from .appraisal import appraise
from .profile import BASELINE, DRIFT_LIMIT, REACTIONS, TONES, TRAITS, VOICE
from .state import Episode, Followup, PersonaState, Relationship

MOOD_HALF_LIFE = 1800.0       # simulated seconds
RESTING_VALENCE, RESTING_ENERGY = 0.1, 0.6
FOLLOWUP_DELAY = 2 * 3600.0   # check in a couple of simulated hours later
FOLLOWUP_LIFETIME = 24 * 3600.0
TOOL_FAILURE_DIPS_PER_TASK = 5
VIEW_LIMIT = 900              # characters of JSON in any prompt view


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class Persona:
    """Deterministic personality state driven by existing bus events.

    It never calls the model. Agents receive small views of it in calls they
    already make, so it shapes speech and initiative without extra latency.
    """

    def __init__(self, state: PersonaState | None = None, name: str = "Kip",
                 now: Callable[[], float] | None = None, helpfulness_baseline: float = 0.8) -> None:
        self.state = state if state is not None else PersonaState()
        self.name = name
        self._now = now
        self._last_now: float | None = None
        self._helpfulness_baseline = helpfulness_baseline
        self._failure_dips: dict[str, int] = {}
        self._bus = None
        self.dirty = False

    def attach(self, bus) -> None:
        self._bus = bus
        bus.add_listener(self.observe)

    # ── time ────────────────────────────────────────────────────────────────
    def _advance(self) -> float:
        if self._now is not None:
            current = self._now()
            if self._last_now is not None:
                # Clock resets (world reload, restart) never move the persona backwards.
                self.state.age_seconds += max(0.0, current - self._last_now)
            self._last_now = current
        mood = self.state.mood
        elapsed = self.state.age_seconds - mood.updated_at
        if elapsed > 0:
            keep = 0.5 ** (elapsed / MOOD_HALF_LIFE)
            mood.valence = RESTING_VALENCE + (mood.valence - RESTING_VALENCE) * keep
            mood.energy = RESTING_ENERGY + (mood.energy - RESTING_ENERGY) * keep
            mood.updated_at = self.state.age_seconds
        return self.state.age_seconds

    # ── learning from events ────────────────────────────────────────────────
    def observe(self, event: dict) -> None:
        if event["type"].startswith("persona_"):
            return
        appraisal = appraise(event)
        if appraisal is None:
            return
        if event["type"] == "tool_failed":
            task = event.get("data", {}).get("task_id") or ""
            self._failure_dips[task] = self._failure_dips.get(task, 0) + 1
            if self._failure_dips[task] > TOOL_FAILURE_DIPS_PER_TASK:
                return
        now = self._advance()
        state, mood = self.state, self.state.mood
        mood.valence = _clamp(mood.valence + appraisal.valence, -1, 1)
        mood.energy = _clamp(mood.energy + appraisal.energy, 0, 1)
        for trait, amount in appraisal.traits.items():
            self.reinforce(trait, amount)
        feeling = self.mood_label()
        person = appraisal.person
        if person:
            first_meeting = person not in state.relationships
            bond = state.relationships.setdefault(person, Relationship())
            bond.familiarity = _clamp(bond.familiarity + appraisal.familiarity, 0, 1)
            bond.warmth = _clamp(bond.warmth + appraisal.warmth, 0, 1)
            bond.interactions += appraisal.interaction
            bond.times_helped += appraisal.helped
            bond.last_interaction = now
            if appraisal.moment:
                bond.moments = (bond.moments + [appraisal.moment])[-3:]
            if first_meeting:
                self._remember(f"Met {person} for the first time.", feeling, person, now)
            if appraisal.settles_followups:
                state.followups = [f for f in state.followups if f.person != person or f.due_at > now]
            if appraisal.followup:
                state.followups = [f for f in state.followups if f.person != person]
                state.followups.append(Followup(person=person, reason=appraisal.followup,
                                                due_at=now + FOLLOWUP_DELAY,
                                                expires_at=now + FOLLOWUP_LIFETIME))
        if appraisal.episode and (appraisal.outcome or abs(appraisal.valence) >= 0.15):
            self._remember(appraisal.episode, feeling, person, now)
        state.followups = [f for f in state.followups if f.expires_at > now][-6:]
        self.dirty = True
        if self._bus is not None:
            self._bus.emit("persona_updated", self.name, skip_history=True, persona=self.public())

    def reinforce(self, trait: str, amount: float) -> None:
        base = BASELINE[trait]
        value = self.state.traits.get(trait, base) + amount
        self.state.traits[trait] = round(_clamp(value, max(0, base - DRIFT_LIMIT), min(1, base + DRIFT_LIMIT)), 4)

    def _remember(self, summary: str, feeling: str, person: str | None, at: float) -> None:
        self.state.episodes = (self.state.episodes + [Episode(summary=summary, feeling=feeling,
                                                                person=person, at=at)])[-10:]

    # ── derived drives ──────────────────────────────────────────────────────
    def mood_label(self) -> str:
        valence, energy = self.state.mood.valence, self.state.mood.energy
        if energy < 0.3:
            return "tired"
        if valence <= -0.15:
            return "concerned" if energy < 0.7 else "determined"
        return "cheerful" if valence >= 0.35 else "content"

    def trait(self, name: str) -> float:
        return self.state.traits.get(name, BASELINE[name])

    def helpfulness(self) -> float:
        """Replaces the fixed drive_helpfulness: empathy and mood move it."""
        self._advance()
        shift = (self.trait("empathy") - BASELINE["empathy"]) + 0.1 * self.state.mood.valence
        return round(_clamp(self._helpfulness_baseline + shift, 0, 1), 3)

    def intention_threshold(self, base: float) -> float:
        """Empathy lowers the bar for helping; a low mood raises it (caution)."""
        self._advance()
        adjusted = base - 0.1 * (self.trait("empathy") - 0.5) + 0.1 * max(0.0, -self.state.mood.valence)
        return round(_clamp(adjusted, 0.3, 0.95), 3)

    def reaction(self, status: str) -> str | None:
        self._advance()
        return REACTIONS.get((status, self.mood_label()))

    def due_followups(self) -> list[Followup]:
        now = self._advance()
        return [f for f in self.state.followups if f.due_at <= now < f.expires_at]

    # ── views ───────────────────────────────────────────────────────────────
    def _hours_since(self, moment: float | None) -> float | None:
        return None if moment is None else round((self.state.age_seconds - moment) / 3600, 1)

    def _bond(self, person: str, detail: bool) -> dict:
        bond = self.state.relationships[person]
        view = {"familiarity": round(bond.familiarity, 2), "warmth": round(bond.warmth, 2),
                "times_helped": bond.times_helped}
        if detail:
            view["moments"] = bond.moments[-2:]
            view["hours_since_last_interaction"] = self._hours_since(bond.last_interaction)
        return view

    @staticmethod
    def _bounded(view: dict, trim: list[str]) -> dict:
        for key in trim:
            if len(json.dumps(view, ensure_ascii=False)) <= VIEW_LIMIT:
                break
            view.pop(key, None)
        return view

    def voice_view(self, people: list[str] | None = None) -> dict:
        """For speech and written answers: who the robot is and how it feels now."""
        self._advance()
        label = self.mood_label()
        view = {"name": self.name, "voice": VOICE, "mood": label, "tone": TONES[label]}
        known = [p for p in (people or []) if p in self.state.relationships]
        if known:
            view["relationships"] = {p: self._bond(p, detail=True) for p in known[:3]}
        return self._bounded(view, ["relationships"])

    def mind_view(self) -> dict:
        """For idle initiative: traits, mood, closest people and due check-ins."""
        due = self.due_followups()
        bonds = self.state.relationships
        closest = sorted(bonds, key=lambda p: -(bonds[p].warmth + bonds[p].familiarity))
        view = {"name": self.name, "mood": self.mood_label(),
                "traits": {t: round(self.trait(t), 2) for t in TRAITS},
                "relationships": {p: self._bond(p, detail=True) for p in closest[:4]},
                "due_followups": [{"person": f.person, "reason": f.reason} for f in due[:3]],
                "recent_moments": [e.summary for e in self.state.episodes[-3:]]}
        return self._bounded(view, ["recent_moments", "relationships"])

    def salient_moments(self) -> list[dict]:
        return [{"summary": e.summary, "feeling": e.feeling, "person": e.person}
                for e in self.state.episodes[-5:]]

    def public(self) -> dict:
        label = self.mood_label()
        bonds = sorted(self.state.relationships.items(), key=lambda item: -(item[1].warmth + item[1].familiarity))
        return {"name": self.name, "mood": label, "tone": TONES[label],
                "valence": round(self.state.mood.valence, 2), "energy": round(self.state.mood.energy, 2),
                "traits": {t: round(self.trait(t), 3) for t in TRAITS}, "baseline": BASELINE,
                "relationships": {p: {**self._bond(p, detail=True), "interactions": b.interactions}
                                  for p, b in bonds[:8]},
                "moments": [e.model_dump() for e in self.state.episodes[-5:]],
                "followups": [f.model_dump() for f in self.state.followups]}
