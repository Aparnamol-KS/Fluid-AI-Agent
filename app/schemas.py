from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PlannedTask(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)

    id: str
    description: str
    action: Literal[
        "draft_content",
        "review_content",
        "create_document",
    ]


class DocumentPlan(BaseModel):
    title: str
    document_type: str
    assumptions: list[str] = Field(default_factory=list)
    sections: list[str] = Field(min_length=2, max_length=12)
    tasks: list[PlannedTask] = Field(min_length=3, max_length=6)

    @model_validator(mode="after")
    def validate_tasks(self):
        actions = [task.action for task in self.tasks]

        required = [
            "draft_content",
            "review_content",
            "create_document",
        ]

        if len(actions) != len(set(actions)):
            raise ValueError("Duplicate task actions are not allowed.")

        if actions != required:
            raise ValueError(
                "Tasks must contain draft_content, review_content, "
                "and create_document in that order."
            )

        return self


class AgentRequest(BaseModel):
    request: str = Field(
        min_length=5,
        max_length=5000,
        description="Natural-language request for a business document",
    )


class Task(BaseModel):
    id: str
    description: str
    status: str = "pending"


class AgentResponse(BaseModel):
    message: str
    title: str
    plan: list[Task]
    assumptions: list[str]
    execution_log: list[str]
    document_url: str
