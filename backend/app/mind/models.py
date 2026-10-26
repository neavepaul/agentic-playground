from pydantic import BaseModel, Field

from app.agents.schemas import GoalCondition


class Intention(BaseModel):
    goal: str = Field(max_length=200,
                      description="A specific goal the robot should pursue right now.")
    reason: str = Field(max_length=300,
                        description="Why this is worth doing — grounded in memory or recent events.")
    priority: float = Field(ge=0.0, le=1.0,
                             description="How urgent this is. 0=idle curiosity, 0.6=worth acting on, "
                                         "0.9=genuinely needed. Be conservative.")
    conditions: list[GoalCondition] = Field(
        default_factory=list,
        description="Machine-checkable success conditions. May be empty; Coordinator will derive them.")


class IntentionOrIdle(BaseModel):
    """Output schema for the idle intention generator.

    Return an intention only when it has genuine grounding from memory or events.
    Prefer returning null and staying idle over inventing busywork.
    """
    intention: Intention | None = Field(
        default=None,
        description="A grounded intention to pursue, or null to remain idle.")
