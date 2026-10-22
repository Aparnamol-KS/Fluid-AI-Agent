import json
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, ConfigDict, Field, ValidationError

load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class PlannedTask(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)

    id: str
    description: str
    action: Literal[
        "draft_content",
        "review_content",
        "create_document",
    ]


class Plan(BaseModel):
    title: str
    document_type: str
    assumptions: list[str] = Field(default_factory=list)
    tasks: list[PlannedTask] = Field(min_length=1, max_length=6)
    sections: list[str] = Field(min_length=2, max_length=12)


def ask_llm(system_prompt: str, user_prompt: str) -> str:
    """Call Groq and return the generated text."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_completion_tokens=3000,
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("The model returned an empty response.")

    return content


def generate_plan(user_request: str) -> Plan:
    """Generate and validate an execution plan."""
    raw = ask_llm(
        system_prompt=(
            "You are a business-document planning agent. "
            "Return only valid JSON. Do not use Markdown fences. "
            "Create a concise, actionable plan for the request. "
            "Use reasonable assumptions for missing nonessential details. "
            "Use only these task actions: draft_content, review_content, "
            "create_document. Include all three actions in a sensible order."
        ),
        user_prompt=f"""
            Create a plan for this request:

            {user_request}

            Return a JSON object with:
            - title: a suitable document title
            - document_type: e.g. proposal, project plan, report, SOP
            - assumptions: list of reasonable assumptions
            - sections: ordered list of document section headings
            - tasks: list of objects containing id, description, action

            The task actions must be draft_content, review_content, or create_document.
        """,
    )

    try:
        plan = Plan.model_validate_json(raw)
    except (ValidationError, ValueError) as exc:
        raise ValueError("The model returned an invalid execution plan.") from exc

    # Ensure the execution pipeline is complete and correctly ordered.
    required_actions = [
        "draft_content",
        "review_content",
        "create_document",
    ]

    planned_actions = [task.action for task in plan.tasks]

    for action in required_actions:
        if action not in planned_actions:
            plan.tasks.append(
                PlannedTask(
                    id=action,
                    description=f"Execute the {action.replace('_', ' ')} step.",
                    action=action,
                )
            )

    # Normalize order so every prerequisite runs before its dependent step.
    order = {action: index for index, action in enumerate(required_actions)}
    plan.tasks.sort(key=lambda task: order[task.action])

    return plan


def execute_plan(user_request: str, plan: Plan) -> dict:
    """Execute the planned actions and track their results."""
    task_results = []
    execution_log = []
    content = ""
    output_path = None

    try:
        for task in plan.tasks:
            task_results.append(
                {
                    "id": task.id,
                    "description": task.description,
                    "action": task.action,
                    "status": "running",
                }
            )
            current = task_results[-1]

            try:
                if task.action == "draft_content":
                    content = ask_llm(
                        system_prompt=(
                            "You are a professional business writer. "
                            "Create a complete, practical document using "
                            "the requested sections. Use clear headings, "
                            "specific recommendations, and realistic detail. "
                            "Do not invent real-world facts or claim that "
                            "unverified actions have occurred. Label examples "
                            "and assumptions clearly."
                        ),
                        user_prompt=f"""
                            User request:
                            {user_request}

                            Document title: {plan.title}
                            Document type: {plan.document_type}
                            Assumptions: {json.dumps(plan.assumptions)}
                            Required sections: {json.dumps(plan.sections)}

                            Write the document body with all the required sections.
                        """,
                    )
                    execution_log.append("Drafted document content using Groq.")

                elif task.action == "review_content":
                    if not content:
                        raise ValueError("No draft is available to review.")

                    content = ask_llm(
                        system_prompt=(
                            "You are a meticulous business-document editor. "
                            "Correct inconsistencies, improve clarity, remove "
                            "repetition, and preserve useful details. Return "
                            "only the revised document, without commentary."
                        ),
                        user_prompt=f"""
                            Review this document against the original request.

                            Original request: {user_request}
                            Document title: {plan.title}
                            Required sections: {json.dumps(plan.sections)}

                            Draft:
                            {content}
                        """,
                    )
                    execution_log.append(
                        "Reviewed the draft for clarity and consistency."
                    )

                elif task.action == "create_document":
                    if not content:
                        raise ValueError("No reviewed content is available.")

                    from app.document import create_docx

                    output_path = create_docx(
                        title=plan.title,
                        document_type=plan.document_type,
                        content=content,
                        assumptions=plan.assumptions,
                    )
                    execution_log.append("Generated the Word document.")

                current["status"] = "completed"

            except Exception:
                current["status"] = "failed"
                raise

    except Exception as exc:
        execution_log.append(
            f"Execution stopped because a step failed: {type(exc).__name__}: {exc}"
        )
        raise AgentExecutionError(
            "The agent could not complete the execution plan.",
            task_results,
            execution_log,
        ) from exc

    return {
        "plan": task_results,
        "assumptions": plan.assumptions,
        "execution_log": execution_log,
        "document_path": str(Path(output_path).resolve()),
        "title": plan.title,
    }


class AgentExecutionError(Exception):
    def __init__(
        self,
        message: str,
        plan: list[dict],
        execution_log: list[str],
    ):
        super().__init__(message)
        self.plan = plan
        self.execution_log = execution_log
