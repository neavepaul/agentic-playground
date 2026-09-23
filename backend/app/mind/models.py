from pydantic import BaseModel, Field

from app.agents.schemas import GoalCondition


class EdgeUpsert(BaseModel):
    """A belief the agent wants to add or strengthen."""
    subject: str = Field(description="Entity id (person or object).")
    relation: str = Field(description="One of: located_in | needs | has | recurring_need.")
    target: str = Field(description="Target entity id (room, object, or person).")
    confidence: float = Field(ge=0.0, le=1.0,
                              description="0.9=just observed; 0.7=recent; 0.5=inferred; 0.3=stale.")
    reason: str = Field(default="", max_length=200,
                        description="The specific observation that grounds this belief.")


class EdgeRemove(BaseModel):
    """A belief the agent wants to retract because it was directly contradicted."""
    subject: str
    relation: str
    target: str


class GraphConsolidation(BaseModel):
    """Memory consolidation output — what the agent learned from recent experience."""
    upsert_edges: list[EdgeUpsert] = Field(
        default_factory=list,
        description="Beliefs to add or update. Return empty when nothing meaningful changed.")
    remove_edges: list[EdgeRemove] = Field(
        default_factory=list,
        description="Beliefs contradicted by direct observation and should be retracted.")


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
