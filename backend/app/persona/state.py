from pydantic import BaseModel, Field, model_validator

from .profile import BASELINE, DRIFT, LEGACY_BASELINE, LEGACY_RENAMES

CHARACTER_VERSION = 2


class Mood(BaseModel):
    """Functional affect: it selects behaviour and tone, it is never announced as numbers."""
    valence: float = Field(default=0.2, ge=-1, le=1)
    energy: float = Field(default=0.6, ge=0, le=1)
    confidence: float = Field(default=0.6, ge=0, le=1)   # in his own competence, right now
    frustration: float = Field(default=0.0, ge=0, le=1)  # from setbacks; makes him methodical
    intrigue: float = Field(default=0.2, ge=0, le=1)     # aroused curiosity from novelty
    updated_at: float = 0.0  # persona-life seconds


class Relationship(BaseModel):
    familiarity: float = Field(default=0.0, ge=0, le=1)
    warmth: float = Field(default=0.5, ge=0, le=1)
    interactions: int = 0
    times_helped: int = 0
    last_interaction: float | None = None
    moments: list[str] = Field(default_factory=list)
    # Learned from how this person actually responds, never assumed.
    chats: int = 0
    humour_reception: float = Field(default=0.5, ge=0, le=1)
    concern: float = Field(default=0.0, ge=0, le=1)
    concern_reason: str = ""          # their own words that raised it
    concern_at: float | None = None
    quiet_until: float | None = None  # they asked for space: no check-ins or initiative toward them


class Episode(BaseModel):
    summary: str
    feeling: str
    person: str | None = None
    at: float


class Followup(BaseModel):
    person: str
    reason: str
    due_at: float
    expires_at: float


class SkillRecord(BaseModel):
    """Kip's record of one kind of work: what he can honestly say about his reliability."""
    attempts: int = 0
    successes: int = 0
    streak: int = 0  # >0 consecutive successes, <0 consecutive failures

    @property
    def expected(self) -> float:
        return (self.successes + 1) / (self.attempts + 2)


class ObjectNote(BaseModel):
    """Where Kip has personally seen a thing. Raw sightings; patterns are only suspicions."""
    first_room: str
    last_room: str
    rooms: list[str] = Field(default_factory=list)
    moves: int = 0


class PersonaState(BaseModel):
    """Everything about the persona that evolves. Persisted with PersistentMemory."""
    character_version: int = CHARACTER_VERSION
    traits: dict[str, float] = Field(default_factory=lambda: dict(BASELINE))
    # Trait values from an earlier character definition, kept as history.
    legacy_traits: dict[str, float] = Field(default_factory=dict)
    mood: Mood = Field(default_factory=Mood)
    relationships: dict[str, Relationship] = Field(default_factory=dict)
    episodes: list[Episode] = Field(default_factory=list)
    followups: list[Followup] = Field(default_factory=list)
    skills: dict[str, SkillRecord] = Field(default_factory=dict)
    seen_objects: dict[str, ObjectNote] = Field(default_factory=dict)
    # Simulated seconds lived across sessions. World clocks restart with the
    # backend, so time is accumulated rather than read as an absolute value.
    age_seconds: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def _carry_over_earlier_character(cls, data):
        """A state saved by the earlier character keeps its experience: drift each trait
        earned is applied to the new baseline (within the new limits), and the old
        values are kept verbatim in legacy_traits."""
        if not isinstance(data, dict) or "character_version" in data or "traits" not in data:
            return data
        old = dict(data.get("traits") or {})
        traits = dict(BASELINE)
        for name, value in old.items():
            if name in LEGACY_BASELINE:
                new = LEGACY_RENAMES.get(name, name)
                drift = value - LEGACY_BASELINE[name]
                limit = DRIFT[new]
                traits[new] = round(BASELINE[new] + max(-limit, min(limit, drift)), 4)
        return {**data, "traits": traits, "legacy_traits": old, "character_version": CHARACTER_VERSION}
