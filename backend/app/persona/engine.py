import json
from typing import Callable

from .appraisal import appraise
from .cues import read_cues
from .profile import BASELINE, DRIFT, IDENTITY, TONES, VOICE
from .state import Episode, Followup, ObjectNote, PersonaState, Relationship, SkillRecord

MOOD_HALF_LIFE = 1800.0          # simulated seconds
CONCERN_HALF_LIFE = 6 * 3600.0
RESTING = {"valence": 0.1, "energy": 0.6, "confidence": 0.6, "frustration": 0.0, "intrigue": 0.2}
FOLLOWUP_DELAY = 2 * 3600.0      # check in a couple of simulated hours later
FOLLOWUP_LIFETIME = 24 * 3600.0
QUIET_SECONDS = 8 * 3600.0       # how long "leave me alone" is honoured
TOOL_FAILURE_DIPS_PER_TASK = 5
VIEW_LIMIT = 1600                # characters of JSON in any prompt view
CONCERNED, WELL_KNOWN, STRANGER = 0.35, 0.3, 0.1
INITIATIVE_TRAITS = ("initiative", "curiosity", "caution", "empathy", "loyalty")


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _goal_key(goal: str) -> str:
    return " ".join(goal.casefold().split())


class Persona:
    """Deterministic personality state driven by existing bus events.

    It never calls the model. Agents receive small views of it in calls they
    already make, so it shapes judgement, speech and initiative without extra latency.
    Personality never grants permissions: consent, safety and claim checks live elsewhere.
    """

    def __init__(self, state: PersonaState | None = None, name: str = "Kip",
                 now: Callable[[], float] | None = None, helpfulness_baseline: float = 0.8) -> None:
        self.state = state if state is not None else PersonaState()
        self.name = name
        self._now = now
        self._last_now: float | None = None
        self._helpfulness_baseline = helpfulness_baseline
        self._failure_dips: dict[str, int] = {}
        self._self_initiated: set[str] = set()
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
            for field, rest in RESTING.items():
                setattr(mood, field, rest + (getattr(mood, field) - rest) * keep)
            fade = 0.5 ** (elapsed / CONCERN_HALF_LIFE)
            for bond in self.state.relationships.values():
                bond.concern = round(bond.concern * fade, 4) if bond.concern > 0.01 else 0.0
            mood.updated_at = self.state.age_seconds
        return self.state.age_seconds

    # ── learning from events ────────────────────────────────────────────────
    def observe(self, event: dict) -> None:
        kind, data = event["type"], event.get("data", {})
        if kind.startswith("persona_"):
            return
        if kind == "agent_intention":
            self._self_initiated.add(_goal_key(data.get("summary", "")))
            return
        if kind == "chat_message" and (data.get("turn") or {}).get("role") != "user":
            self._settle_after_reply(data.get("turn") or {})
            return
        self_initiated = False
        if kind in {"task_completed", "task_failed"}:
            key = _goal_key((data.get("task") or {}).get("goal", ""))
            self_initiated = key in self._self_initiated
            self._self_initiated.discard(key)
        if kind == "tool_failed":
            task = data.get("task_id") or ""
            self._failure_dips[task] = self._failure_dips.get(task, 0) + 1
            if self._failure_dips[task] > TOOL_FAILURE_DIPS_PER_TASK:
                return
        now = self._advance()
        appraisal = appraise(event, self.state, self_initiated)
        if appraisal is None:
            return
        self._apply(appraisal, now)
        self.dirty = True
        if self._bus is not None:
            self._bus.emit("persona_updated", self.name, skip_history=True, persona=self.public())

    def _apply(self, appraisal, now: float) -> None:
        state, mood = self.state, self.state.mood
        mood.valence = _clamp(mood.valence + appraisal.valence, -1, 1)
        mood.energy = _clamp(mood.energy + appraisal.energy, 0, 1)
        mood.confidence = _clamp(mood.confidence + appraisal.confidence, 0, 1)
        mood.frustration = _clamp(mood.frustration * (1 - appraisal.relief) + appraisal.frustration, 0, 1)
        mood.intrigue = _clamp(mood.intrigue + appraisal.intrigue, 0, 1)
        for trait, amount in appraisal.traits.items():
            self.reinforce(trait, amount)
        if appraisal.skill:
            record = state.skills.setdefault(appraisal.skill, SkillRecord())
            record.attempts += 1
            record.successes += appraisal.succeeded
            record.streak = (max(record.streak, 0) + 1) if appraisal.succeeded else (min(record.streak, 0) - 1)
        self._note_sightings(appraisal.seen_objects, now)
        feeling = self.mood_label()
        person = appraisal.person
        if person:
            self._update_bond(person, appraisal, feeling, now)
        if appraisal.episode:
            self._remember(appraisal.episode, feeling, person, now)
        state.followups = [f for f in state.followups if f.expires_at > now][-6:]

    def _update_bond(self, person: str, appraisal, feeling: str, now: float) -> None:
        state = self.state
        first_meeting = person not in state.relationships
        bond = state.relationships.setdefault(person, Relationship())
        bond.familiarity = _clamp(bond.familiarity + appraisal.familiarity, 0, 1)
        bond.warmth = _clamp(bond.warmth + appraisal.warmth, 0, 1)
        bond.interactions += appraisal.interaction
        bond.times_helped += appraisal.helped
        bond.chats += appraisal.chat
        bond.humour_reception = _clamp(bond.humour_reception + appraisal.humour_reception, 0, 1)
        if appraisal.concern:
            bond.concern = _clamp(bond.concern + appraisal.concern, 0, 1)
            bond.concern_reason, bond.concern_at = appraisal.concern_reason, now
        bond.last_interaction = now
        if appraisal.moment:
            bond.moments = (bond.moments + [appraisal.moment])[-3:]
        if first_meeting:
            self._remember(f"Met {person} for the first time.", feeling, person, now)
        if appraisal.wants_space:
            bond.quiet_until = now + QUIET_SECONDS
            state.followups = [f for f in state.followups if f.person != person]
        if appraisal.settles_followups:
            state.followups = [f for f in state.followups if f.person != person or f.due_at > now]
        if appraisal.followup and not self.wants_space(person, now):
            state.followups = [f for f in state.followups if f.person != person]
            state.followups.append(Followup(person=person, reason=appraisal.followup,
                                            due_at=now + FOLLOWUP_DELAY, expires_at=now + FOLLOWUP_LIFETIME))

    def _settle_after_reply(self, turn: dict) -> None:
        """Kip has spoken to this person: a due check-in has had its chance."""
        person = turn.get("to")
        if not person or turn.get("source"):
            return
        now = self._advance()
        before = len(self.state.followups)
        self.state.followups = [f for f in self.state.followups if f.person != person or f.due_at > now]
        self.dirty |= len(self.state.followups) != before

    def _note_sightings(self, seen: dict[str, str], now: float) -> None:
        for key, room in seen.items():
            note = self.state.seen_objects.get(key)
            if note is None:
                self.state.seen_objects[key] = ObjectNote(first_room=room, last_room=room, rooms=[room])
                continue
            if note.last_room != room:
                note.moves += 1
                note.last_room = room
                if room not in note.rooms:
                    note.rooms = (note.rooms + [room])[-5:]
                if note.moves == 2:
                    self._remember(f"Noticed the {key.replace('_', ' ')} keeps turning up in different "
                                   f"places ({', '.join(r.replace('_', ' ') for r in note.rooms)}).",
                                   self.mood_label(), None, now)

    def reinforce(self, trait: str, amount: float) -> None:
        base, limit = BASELINE[trait], DRIFT[trait]
        value = self.state.traits.get(trait, base) + amount
        self.state.traits[trait] = round(_clamp(value, max(0, base - limit), min(1, base + limit)), 4)

    def _remember(self, summary: str, feeling: str, person: str | None, at: float) -> None:
        self.state.episodes = (self.state.episodes + [Episode(summary=summary, feeling=feeling,
                                                                person=person, at=at)])[-12:]

    # ── derived drives ──────────────────────────────────────────────────────
    def _recent_concern(self) -> float:
        now = self.state.age_seconds
        return max((b.concern for b in self.state.relationships.values()
                    if b.last_interaction is not None and now - b.last_interaction < 2 * 3600), default=0.0)

    def mood_label(self) -> str:
        mood = self.state.mood
        if mood.energy < 0.3:
            return "tired"
        if self._recent_concern() >= CONCERNED or (mood.valence <= -0.15 and mood.frustration < 0.35):
            return "concerned"
        if mood.frustration >= 0.35:
            return "determined"
        if mood.intrigue >= 0.5:
            return "curious"
        return "pleased" if mood.valence >= 0.35 else "content"

    def trait(self, name: str) -> float:
        return self.state.traits.get(name, BASELINE[name])

    def wants_space(self, person: str | None, now: float | None = None) -> bool:
        bond = self.state.relationships.get(person) if person else None
        now = self._advance() if now is None else now
        return bool(bond and bond.quiet_until and bond.quiet_until > now)

    def helpfulness(self) -> float:
        """Replaces the fixed drive_helpfulness: empathy and mood move it."""
        self._advance()
        shift = (self.trait("empathy") - BASELINE["empathy"]) + 0.1 * self.state.mood.valence
        return round(_clamp(self._helpfulness_baseline + shift, 0, 1), 3)

    def intention_threshold(self, base: float) -> float:
        """Initiative and empathy lower the bar for acting unasked; a low mood or recent
        setbacks raise it. Only the bar moves: permission to act is never granted here."""
        self._advance()
        mood = self.state.mood
        adjusted = (base - 0.15 * (self.trait("initiative") - 0.5) - 0.1 * (self.trait("empathy") - 0.5)
                    + 0.1 * max(0.0, -mood.valence) + 0.1 * mood.frustration)
        return round(_clamp(adjusted, 0.3, 0.95), 3)

    def due_followups(self) -> list[Followup]:
        now = self._advance()
        return [f for f in self.state.followups
                if f.due_at <= now < f.expires_at and not self.wants_space(f.person, now)]

    def track_record(self, limit: int = 5) -> dict[str, str]:
        skills = sorted(self.state.skills.items(), key=lambda item: -item[1].attempts)
        return {name: f"{r.successes} of {r.attempts} succeeded" for name, r in skills[:limit] if r.attempts}

    def suspicions(self, limit: int = 3) -> list[str]:
        """Patterns Kip has noticed himself. Suspicions, not facts."""
        moving = sorted(((k, n) for k, n in self.state.seen_objects.items() if n.moves >= 2),
                        key=lambda item: -item[1].moves)
        return [f"the {k.replace('_', ' ')} moves around (seen in {len(n.rooms)} places)" for k, n in moving[:limit]]

    # ── how to pitch a reply ────────────────────────────────────────────────
    def stance(self, speaker: str | None = None, message: str = "") -> str:
        """One compact, deterministic line on how to pitch this reply to this person.

        It is derived from their own words and Kip's history with them, and never
        changes facts, consent or what Kip will do.
        """
        now = self._advance()
        cues = read_cues(message)
        bond = self.state.relationships.get(speaker) if speaker else None
        parts = []
        if speaker is None:
            parts.append("You don't know who is speaking: don't assume, and don't attach what they say to anyone.")
        elif bond is None or bond.familiarity < STRANGER:
            parts.append("You barely know them yet: friendly, a little reserved.")
        elif bond.familiarity >= WELL_KNOWN:
            parts.append("You know them well: relaxed, the shorthand of shared history.")
        label = self.mood_label()
        if cues.serious:
            parts.append("Serious subject: no humour; calm, clear and practical.")
        elif cues.distress:
            parts.append(f"They said '{cues.evidence}': acknowledge it briefly and listen. No jokes, and no "
                         "plans or errands unless they ask.")
        elif cues.boundary or self.wants_space(speaker, now):
            parts.append("They want space: a short, warm acknowledgement; ask nothing, offer nothing.")
        elif bond and bond.concern >= CONCERNED:
            parts.append(f"Earlier they said '{bond.concern_reason}': be gentle; humour only if they start it.")
        elif cues.humour_unwanted:
            parts.append("They don't want jokes right now: plain and sincere.")
        else:
            # Humour is earned: until Kip knows someone, he holds back whatever they laughed at once.
            known = bool(bond and bond.familiarity >= STRANGER)
            reception = 0.5 + (bond.humour_reception if known else 0.3)
            mood_factor = {"tired": 0.6, "concerned": 0.4, "determined": 0.7}.get(label, 1.0)
            score = self.trait("dry_humour") * reception * mood_factor
            tease = bool(bond and bond.familiarity >= WELL_KNOWN and bond.humour_reception >= 0.55
                         and self.trait("mischief") >= 0.5)
            if score >= 0.8:
                parts.append("A dry remark is welcome if the moment genuinely offers one"
                             + (", and gentle teasing suits them." if tease else "."))
            elif score >= 0.5:
                parts.append("At most one understated aside, if any.")
            else:
                parts.append("Keep it plain; humour only if they start it.")
        if cues.criticism:
            parts.append("They are critical of you: take it on board without defensiveness; admit what is fair.")
        elif cues.praise and not cues.sensitive:
            parts.append("Take thanks lightly, without gushing.")
        due = [f for f in self.due_followups() if f.person == speaker]
        if due and not cues.sensitive:
            parts.append(f"Background: {due[0].reason}. One brief check-in is natural, if it fits.")
        return " ".join(parts)

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
        if self.wants_space(person):
            view["do_not_disturb"] = True
        return view

    @staticmethod
    def _bounded(view: dict, trim: list[str]) -> dict:
        for key in trim:
            if len(json.dumps(view, ensure_ascii=False)) <= VIEW_LIMIT:
                break
            view.pop(key, None)
        return view

    def voice_view(self, people: list[str] | None = None, speaker: str | None = None,
                   message: str | None = None) -> dict:
        """For speech and written answers: who Kip is, how he feels, and (in conversation)
        how to pitch this reply to the person speaking."""
        self._advance()
        label = self.mood_label()
        view = {"name": self.name, "identity": IDENTITY, "voice": VOICE, "mood": label, "tone": TONES[label]}
        if message is not None:
            view["stance"] = self.stance(speaker, message)
            record = self.track_record()
            if record:
                view["track_record"] = record
        known = [p for p in (people or []) if p in self.state.relationships]
        if known:
            view["relationships"] = {p: self._bond(p, detail=True) for p in known[:3]}
        return self._bounded(view, ["relationships", "track_record", "identity"])

    def mind_view(self) -> dict:
        """For idle initiative: traits, mood, closest people, due check-ins, open questions."""
        due = self.due_followups()
        bonds = self.state.relationships
        closest = sorted(bonds, key=lambda p: -(bonds[p].warmth + bonds[p].familiarity))
        view = {"name": self.name, "mood": self.mood_label(),
                "traits": {t: round(self.trait(t), 2) for t in INITIATIVE_TRAITS},
                "relationships": {p: self._bond(p, detail=True) for p in closest[:4]},
                "due_followups": [{"person": f.person, "reason": f.reason} for f in due[:3]],
                "noticed": self.suspicions(),
                "recent_moments": [e.summary for e in self.state.episodes[-3:]]}
        return self._bounded(view, ["recent_moments", "noticed", "relationships"])

    def public(self) -> dict:
        label, mood = self.mood_label(), self.state.mood
        bonds = sorted(self.state.relationships.items(), key=lambda item: -(item[1].warmth + item[1].familiarity))
        return {"name": self.name, "mood": label, "tone": TONES[label],
                "valence": round(mood.valence, 2), "energy": round(mood.energy, 2),
                "affect": {key: round(getattr(mood, key), 2) for key in ("confidence", "frustration", "intrigue")},
                "traits": {t: round(self.trait(t), 3) for t in BASELINE}, "baseline": BASELINE,
                "relationships": {p: {**self._bond(p, detail=True), "interactions": b.interactions, "chats": b.chats,
                                      "humour_reception": round(b.humour_reception, 2),
                                      "concern": round(b.concern, 2)}
                                  for p, b in bonds[:8]},
                "track_record": self.track_record(limit=10),
                "noticed": self.suspicions(),
                "moments": [e.model_dump() for e in self.state.episodes[-5:]],
                "followups": [f.model_dump() for f in self.state.followups]}
