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


class CommandChoice(StrictModel):
    command_id: str
    message: str = Field(default="", max_length=500)
    summary: ShortText


class GoalCondition(StrictModel):
    kind: Literal["find_object", "find_person", "identify_recipient", "hold_object", "deliver", "notify",
                  "notify_everyone", "place_object", "visit_room"]
    object: str = Field(default="", max_length=40)
    person: str = Field(default="", max_length=40)
    room: str = Field(default="", max_length=40)
    message: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def required_targets(self):
        if self.kind in {"find_object", "identify_recipient", "hold_object", "deliver", "place_object"} and not self.object:
            raise ValueError("This condition requires an object ID.")
        if self.kind in {"find_person", "notify"} and not self.person:
            raise ValueError("This condition requires a person ID.")
        if self.kind in {"place_object", "visit_room"} and not self.room:
            raise ValueError("This condition requires a room ID.")
        if self.kind in {"notify", "notify_everyone"} and not self.message:
            raise ValueError("A notification requires its actual message.")
        return self


class GoalPlan(StrictModel):
    summary: ShortText
    conditions: list[GoalCondition] = Field(min_length=1, max_length=8)
