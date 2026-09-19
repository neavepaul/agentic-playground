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
