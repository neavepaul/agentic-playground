"""Pure rules: how real events change Kip's state.

An outcome is judged against what Kip expected from his own record, whether it
mattered to someone, and how it fits recent history, so the same event can move
him differently on different days. Nothing here produces dialogue.
"""
from dataclasses import dataclass, field

from .cues import read_cues
from .profile import TOOL_SKILLS
from .state import PersonaState

@dataclass
class Appraisal:
    valence: float = 0.0
    energy: float = 0.0
    confidence: float = 0.0
    frustration: float = 0.0
    intrigue: float = 0.0
    relief: float = 0.0  # fraction of frustration released
    traits: dict[str, float] = field(default_factory=dict)
    person: str | None = None
    familiarity: float = 0.0
    warmth: float = 0.0
    interaction: bool = False
    helped: bool = False
    chat: bool = False
    humour_reception: float = 0.0
    concern: float = 0.0
    concern_reason: str = ""
    wants_space: bool = False
    episode: str | None = None
    moment: str | None = None
    followup: str | None = None
    settles_followups: bool = False
    skill: str | None = None
    succeeded: bool | None = None


def skill_of(task: dict) -> str:
    conditions = task.get("conditions") or []
    kinds = {c.get("kind") for c in conditions}
    tools = {tool for c in conditions for tool in c.get("required_tools") or []}
    for tool, skill in TOOL_SKILLS.items():
        if tool in tools:
            return skill
    return "answering questions" if "answer" in kinds else "errands"


def _outcome(task: dict, succeeded: bool, state: PersonaState) -> Appraisal:
    skill = skill_of(task)
    record = state.skills.get(skill)
    expected = record.expected if record else 0.5
    surprise = (1 - expected) if succeeded else expected
    goal = task.get("goal", "a task")[:120]
    first_time = record is None or record.attempts == 0
    if succeeded:
        streak_break = bool(record and record.streak <= -2)
        return Appraisal(
            valence=(0.05 + 0.2 * surprise), energy=0.05, confidence=0.12 * surprise + 0.02,
            relief=0.6 if streak_break else 0.3, skill=skill, succeeded=True,
            traits={"resilience": 0.004} if streak_break else {},
            # Ordinary successes are history in the memory store; only notable ones are moments.
            episode=(f"Finished: {goal}" + (" (after a run of failures)" if streak_break else ""))
            if first_time or streak_break or surprise > 0.5 else None)
    failures_in_a_row = -min(record.streak, 0) + 1 if record else 1
    resilience = state.traits.get("resilience", 0.9)
    return Appraisal(
        valence=-(0.05 + 0.2 * surprise) * (1.3 - resilience),
        energy=-0.05, confidence=-0.12 * surprise - 0.02,
        frustration=min(0.4, 0.12 + 0.06 * failures_in_a_row), skill=skill, succeeded=False,
        # Repeated failure makes him more careful and methodical, not more upset.
        traits={"caution": 0.01} if failures_in_a_row >= 2 else {},
        episode=f"Could not finish: {goal}")  # failures are never quietly dropped


def _chat(turn: dict, state: PersonaState) -> Appraisal | None:
    person = turn.get("speaker")
    if turn.get("role") != "user" or not person:
        return None  # an unknown speaker never borrows someone else's relationship
    cues = read_cues(turn.get("text", ""))
    bond = state.relationships.get(person)
    # A new conversation after a gap counts; every line of one conversation barely does.
    new_visit = bond is None or bond.last_interaction is None or \
        state.age_seconds - bond.last_interaction > 1800
    appraisal = Appraisal(person=person, chat=True, interaction=new_visit,
                          familiarity=0.03 if new_visit else 0.004)
    if cues.laughter:
        appraisal.humour_reception, appraisal.warmth, appraisal.valence = 0.08, 0.01, 0.03
    if cues.humour_unwanted:
        appraisal.humour_reception = -0.25
    if cues.sensitive:
        appraisal.concern = 0.5 if cues.serious else 0.35
        appraisal.concern_reason = cues.evidence
    if cues.boundary:
        appraisal.wants_space = True
    if cues.praise:
        appraisal.warmth += 0.02
        appraisal.valence += 0.04
    if cues.criticism:
        # Taken on board, not taken personally.
        appraisal.valence -= 0.03
        appraisal.confidence = -0.02
    return appraisal


def appraise(event: dict, state: PersonaState) -> Appraisal | None:
    kind, data = event["type"], event.get("data", {})
    arguments = data.get("arguments") or {}
    if kind == "chat_message":
        return _chat(data.get("turn") or {}, state)
    if kind in {"task_completed", "task_failed"}:
        return _outcome(data.get("task") or {}, kind == "task_completed", state)
    if kind == "tool_failed":
        return Appraisal(valence=-0.02, frustration=0.02)
    if kind == "capability_observed" and data.get("tool") != "finish_answer":
        return Appraisal(energy=0.02, intrigue=0.05, traits={"curiosity": 0.001})
    return None
