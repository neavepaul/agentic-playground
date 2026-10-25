"""Pure rules: how existing world and task events change the persona."""
from dataclasses import dataclass, field


@dataclass
class Appraisal:
    valence: float = 0.0
    energy: float = 0.0
    traits: dict[str, float] = field(default_factory=dict)
    person: str | None = None
    familiarity: float = 0.0
    warmth: float = 0.0
    interaction: bool = False
    helped: bool = False
    episode: str | None = None
    moment: str | None = None
    followup: str | None = None
    settles_followups: bool = False
    outcome: bool = False


def appraise(event: dict) -> Appraisal | None:
    kind, data = event["type"], event.get("data", {})
    arguments = data.get("arguments") or {}
    if kind == "object_given":
        thing, person = arguments.get("object", "something"), arguments.get("person")
        return Appraisal(valence=0.2, energy=0.05, traits={"empathy": 0.005, "loyalty": 0.005},
                         person=person, familiarity=0.03, warmth=0.06, interaction=True, helped=True,
                         moment=f"brought them the {thing}",
                         followup=f"I brought {person} the {thing}; check they are all right")
    if kind == "person_spoken_to":
        return Appraisal(valence=0.03, person=arguments.get("person"), familiarity=0.02,
                         warmth=0.01, interaction=True, settles_followups=True)
    if kind == "task_completed":
        return Appraisal(valence=0.15, energy=0.1, traits={"enthusiasm": 0.005}, outcome=True,
                         episode="Finished: " + data.get("task", {}).get("goal", "a task")[:120])
    if kind == "task_failed":
        return Appraisal(valence=-0.25, energy=-0.1, traits={"caution": 0.01}, outcome=True,
                         episode="Could not finish: " + data.get("task", {}).get("goal", "a task")[:120])
    if kind == "tool_failed":
        return Appraisal(valence=-0.03)
    if kind == "capability_observed" and data.get("tool") != "finish_answer":
        return Appraisal(energy=0.03, traits={"curiosity": 0.003})
    return None
