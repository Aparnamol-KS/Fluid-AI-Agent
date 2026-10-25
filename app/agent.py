import json
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from pydantic import ValidationError
from app.schemas import DocumentPlan

load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def print_stage(title: str, content: str = "", limit: int = 2500):
    print(f"\n{'=' * 20} {title} {'=' * 20}", flush=True)

    if content:
        if len(content) > limit:
            print(content[:limit] + "\n... [output truncated in terminal]", flush=True)
        else:
            print(content, flush=True)


def ask_llm(system_prompt: str, user_prompt: str) -> str:
    """Call Groq and report useful details if no text is returned."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_completion_tokens=6000,
    )

    if not response.choices:
        raise ValueError("Groq returned no choices.")

    choice = response.choices[0]
    content = choice.message.content

    if choice.finish_reason == "length":
        raise ValueError(
            "Groq reached the completion token limit. "
            "The response may be incomplete."
        )

    if not content or not content.strip():
        raise ValueError(
            "Groq returned an empty message. "
            f"Finish reason: {choice.finish_reason!r}; "
            f"refusal: {getattr(choice.message, 'refusal', None)!r}"
        )

    return content.strip()


def generate_plan(user_request: str) -> DocumentPlan:
    """Generate and validate a plan, retrying once if validation fails."""

    system_prompt = """
You are a planning agent that creates execution plans for business documents.

Return exactly one valid JSON object without Markdown fences or explanations.

The object must contain all five keys:
- title: string
- document_type: string
- assumptions: array of strings
- sections: array of 2 to 12 strings
- tasks: array containing exactly three task objects

Each task must contain:
- id: string
- description: string
- action: string

Include these actions exactly once and in this order:
1. draft_content
2. review_content
3. create_document

Adapt the title, assumptions, section headings, and task descriptions
to the user's request. Do not present unknown information as confirmed facts.
Return every required field, even when assumptions is an empty array.
"""

    user_prompt = f"""
Create an execution plan for this request:

{user_request}

Return JSON following this structure, adapted to the request:

{{
  "title": "Document title",
  "document_type": "Project Plan",
  "assumptions": [],
  "sections": [
    "Executive Summary",
    "Recommendations"
  ],
  "tasks": [
    {{
      "id": "T1",
      "description": "Draft the requested document",
      "action": "draft_content"
    }},
    {{
      "id": "T2",
      "description": "Review the draft",
      "action": "review_content"
    }},
    {{
      "id": "T3",
      "description": "Generate the Word document",
      "action": "create_document"
    }}
  ]
}}
"""

    last_error = None
    previous_raw = None

    for attempt in range(2):
        prompt = user_prompt

        if attempt == 1:
            prompt += f"""

Your previous response failed Pydantic validation.

Validation error:
{last_error}

Previous response:
{previous_raw}

Correct the response. Include all five required keys:
title, document_type, assumptions, sections, and tasks.

Include exactly three tasks with these actions in this order:
draft_content, review_content, create_document.

Return only a complete, valid JSON object.
"""

        raw = ask_llm(system_prompt, prompt)
        previous_raw = raw

        try:
            return DocumentPlan.model_validate_json(raw)
        except ValidationError as exc:
            last_error = exc

    raise ValueError(
        f"The model returned an invalid plan after two attempts: {last_error}"
    ) from last_error


def execute_plan(user_request: str, plan: DocumentPlan) -> dict:
    """Execute the planned actions and track their results."""
    task_results = []
    execution_log = []
    content = ""
    output_path = None

    try:
        print_stage(
            "STAGE 1: GENERATED PLAN",
            plan.model_dump_json(indent=2),
        )
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
                            "You are a careful, general-purpose document author. "
                            "Write a complete document that fulfills the user's request, "
                            "required sections, assumptions, and intended audience.\n\n"
                            "Adapt the structure, tone, and detail to the document's purpose. "
                            "Cover every required section, using examples, tables, or lists "
                            "when useful. Be concise and avoid repetition.\n\n"
                            "Respect the user's constraints. Do not invent facts, decisions, "
                            "completed work, test results, or performance claims. Label "
                            "assumptions and proposed decisions clearly, and distinguish "
                            "them from established facts. Do not silently resolve important "
                            "ambiguities by guessing.\n\n"
                            "Keep terminology and details consistent throughout. Use valid "
                            "Markdown, with correctly formatted tables and code blocks. "
                            "Complete all required sections and avoid unfinished content.\n\n"
                            "Return only the document body in Markdown. The application "
                            "inserts the title and assumptions separately, so do not repeat "
                            "the title or add a separate Assumptions section. "
                            "Do not claim that work was executed, verified, or tested "
                            "without supporting evidence."
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
                    print_stage("STAGE 2: INITIAL DRAFT", content)
                    execution_log.append("Drafted document content using Groq.")

                elif task.action == "review_content":
                    if not content:
                        raise ValueError("No draft is available to review.")

                    reviewed_content = ask_llm(
                        system_prompt=(
                            "You are a general-purpose document reviewer and editor. "
                            "Improve the supplied draft against the user's request, "
                            "required sections, assumptions, and intended audience.\n\n"
                            "Check for missing requirements, unsupported claims, invented "
                            "details, contradictions, incomplete content, unclear explanations, "
                            "formatting errors, irrelevant material, and repetition.\n\n"
                            "Respect the original scope and constraints. Distinguish facts "
                            "from assumptions, estimates, examples, and proposals. Correct "
                            "clear errors without guessing when information is uncertain. "
                            "Do not introduce unjustified commitments or claim that work "
                            "was completed, verified, or tested without evidence.\n\n"
                            "Preserve accurate content, useful detail, and the document's "
                            "intended tone. Make targeted corrections rather than rewriting "
                            "or expanding sound material unnecessarily. Ensure all required "
                            "sections are addressed and the document is complete and "
                            "internally consistent.\n\n"
                            "The application inserts the title and assumptions separately. "
                            "Do not repeat the title or add a separate Assumptions section.\n\n"
                            "Return only the complete revised document in Markdown."
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
                    if not reviewed_content.strip():
                        raise ValueError("The reviewer returned empty content.")
                    print_stage("STAGE 3: REVIEWED DOCUMENT", reviewed_content)
                    content = reviewed_content
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
                    print_stage(
                        "STAGE 4: WORD DOCUMENT GENERATED",
                        f"Output path: {output_path}\n"
                        f"File exists: {Path(output_path).is_file()}",
                    )
                    execution_log.append("Generated the Word document.")
                else:
                    raise ValueError(f"Unsupported task action: {task.action}")
                
                current["status"] = "completed"
                print(f"[SUCCESS] Task completed: {task.action}", flush=True)

            except Exception:
                current["status"] = "failed"
                
                import traceback
                traceback.print_exc()
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
    if output_path is None or not Path(output_path).is_file():
        raise RuntimeError(
            "Document generation did not produce an output file."
        )
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
