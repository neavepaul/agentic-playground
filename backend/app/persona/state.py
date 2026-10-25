from pydantic import BaseModel, Field

from .profile import BASELINE


class Mood(BaseModel):
    valence: float = Field(default=0.2, ge=-1, le=1)
    energy: float = Field(default=0.6, ge=0, le=1)
    updated_at: float = 0.0  # persona-life seconds


class Relationship(BaseModel):
    familiarity: float = Field(default=0.0, ge=0, le=1)
    warmth: float = Field(default=0.5, ge=0, le=1)
    interactions: int = 0
    times_helped: int = 0
    last_interaction: float | None = None
    moments: list[str] = Field(default_factory=list)


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


class PersonaState(BaseModel):
    """Everything about the persona that evolves. Persisted with PersistentMemory."""
    traits: dict[str, float] = Field(default_factory=lambda: dict(BASELINE))
    mood: Mood = Field(default_factory=Mood)
    relationships: dict[str, Relationship] = Field(default_factory=dict)
    episodes: list[Episode] = Field(default_factory=list)
    followups: list[Followup] = Field(default_factory=list)
    # Simulated seconds lived across sessions. World clocks restart with the
    # backend, so time is accumulated rather than read as an absolute value.
    age_seconds: float = 0.0
