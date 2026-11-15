from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ShortText = Annotated[str, Field(min_length=1, max_length=300)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class CoordinatorDecision(StrictModel):
    action: Literal["delegate", "consult_critic", "complete", "fail"]
    summary: ShortText
    task: str = Field(default="", max_length=600)
    plan: str = Field(default="", max_length=600)
    evidence_ids: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def required_fields(self):
        if self.action == "delegate" and not self.task.strip():
            raise ValueError("Delegation requires a task.")
        if self.action == "consult_critic" and not self.plan.strip():
            raise ValueError("Review requires a plan.")
        if self.action == "complete" and not self.evidence_ids:
            raise ValueError("Completion requires observation evidence IDs.")
        return self


class PythonProgram(StrictModel):
    """The program for a request that only asks for Python: the one thing the model writes."""
    code: str = Field(min_length=1, max_length=4000)


class ExplorerDecision(StrictModel):
    action: Literal["tool", "report"]
    summary: ShortText
    tool: str = Field(default="", max_length=40)
    arguments: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def required_fields(self):
        if self.action == "tool" and not self.tool:
            raise ValueError("Tool action requires a tool name.")
        if self.action == "report" and (self.tool or self.arguments):
            raise ValueError("A report cannot also execute a tool.")
        return self


class CriticReview(StrictModel):
    approved: bool
    summary: ShortText
    suggestion: str = Field(default="", max_length=400)


class GoalCondition(StrictModel):
    """A verifiable outcome: an answer, backed by the tools it needs."""
    kind: Literal["answer"] = "answer"
    required_tools: list[Literal["web_search", "read_webpage", "run_python"]] = Field(default_factory=list, max_length=3)


class InitialCapability(StrictModel):
    tool: Literal["web_search", "read_webpage", "run_python"]
    arguments: dict = Field(default_factory=dict)


class GoalPlan(StrictModel):
    summary: ShortText
    conditions: list[GoalCondition] = Field(min_length=1, max_length=8)
    initial_actions: list[InitialCapability] = Field(default_factory=list, max_length=4)


class LookupPlan(GoalPlan):
    """A plain web lookup; for a sport with a results feed it also carries the router's reading
    (domain, entities, focus, time, place), so the feed can answer before any search."""
    feed: dict | None = None
