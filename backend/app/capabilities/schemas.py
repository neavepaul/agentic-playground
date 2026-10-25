from typing import Annotated, Literal

from pydantic import Field

from app.agents.schemas import StrictModel


class SearchAction(StrictModel):
    tool: Literal["web_search"]
    query: str = Field(min_length=1, max_length=400, pattern=r"\S")


class ReadAction(StrictModel):
    tool: Literal["read_webpage"]
    url: str = Field(min_length=1, max_length=2048)
    start_char: int = Field(default=0, ge=0, le=200000)


class PythonAction(StrictModel):
    tool: Literal["run_python"]
    code: str = Field(min_length=1, max_length=8000)
    data: dict = Field(default_factory=dict, max_length=100)


class AnswerAction(StrictModel):
    tool: Literal["finish_answer"]
    answer: str = Field(min_length=1, max_length=12000, pattern=r"\S")
    evidence_ids: list[str] = Field(min_length=1, max_length=20)


DigitalAction = Annotated[SearchAction | ReadAction | PythonAction | AnswerAction,
                          Field(discriminator="tool")]


class CapabilityChoice(StrictModel):
    summary: str = Field(min_length=1, max_length=300)
    action: DigitalAction


ACTION_SCHEMAS = {"web_search": SearchAction, "read_webpage": ReadAction,
                  "run_python": PythonAction, "finish_answer": AnswerAction}
