from pydantic import BaseModel, Field


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
