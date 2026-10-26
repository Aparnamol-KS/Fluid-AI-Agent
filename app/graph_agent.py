import json
import logging
import os
from pathlib import Path
from pprint import pformat
from typing import Any, TypedDict

from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import END, START, StateGraph

from app.document import create_docx
from app.schemas import DocumentPlan

load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MAX_REVISIONS = 2
MAX_AGENT_TURNS = 8


logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    request: str
    plan: dict[str, Any]
    content: str
    review_feedback: str
    review_passed: bool
    revisions: int
    agent_turns: int
    messages: list[dict[str, Any]]
    task_results: list[dict[str, str]]
    execution_log: list[str]
    output_path: str
    error: str
    finished: bool


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "draft_content",
            "description": (
                "Draft the requested document using the validated plan. "
                "Call this before reviewing the document."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "review_content",
            "description": (
                "Review the current draft for missing sections, unsupported "
                "claims, contradictions, and repetition. Call after drafting "
                "or after a revision."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "revise_content",
            "description": (
                "Revise the current draft using the review feedback. "
                "Only call after a review identifies problems."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_document",
            "description": (
                "Generate the final Word document after review passes "
                "or the revision limit is reached."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
]


# def log_data(stage: str, label: str, data) -> None:
#     """Print intermediate agent data for debugging and demonstration."""
#     logger.info("\n%s\n%s\n%s", "=" * 25, f"[{stage}] {label}", "=" * 25)
#     logger.info("%s", pformat(data, width=100, sort_dicts=False))


def log_data(stage: str, label: str, data) -> None:
    print(f"\n{'=' * 20} [{stage}] {label} {'=' * 20}", flush=True)
    print(pformat(data, width=100, sort_dicts=False), flush=True)


def log_text_preview(
    stage: str,
    label: str,
    content: str,
    limit: int = 600,
) -> None:
    preview = content[:limit]

    if len(content) > limit:
        preview += "\n... [truncated]"

    log_data(
        stage,
        label,
        {
            "total_characters": len(content),
            "preview": preview,
        },
    )


def _call_llm(system: str, user: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_completion_tokens=4000,
    )

    if not response.choices:
        raise ValueError("Groq returned no choices.")

    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("Groq returned an empty response.")

    return content.strip()


def _update_task(state: AgentState, action: str, status: str) -> list[dict[str, str]]:
    tasks = [dict(task) for task in state.get("task_results", [])]
    existing = next((task for task in tasks if task["action"] == action), None)

    if existing:
        existing["status"] = status
    else:
        tasks.append(
            {
                "id": f"T{len(tasks) + 1}",
                "description": action.replace("_", " ").title(),
                "action": action,
                "status": status,
            }
        )

    return tasks


def agent_node(state: AgentState) -> dict:
    turns = state.get("agent_turns", 0) + 1

    if turns > MAX_AGENT_TURNS:
        return {
            "agent_turns": turns,
            "error": "Agent stopped after reaching the maximum number of turns.",
            "finished": True,
        }

    plan = DocumentPlan.model_validate(state["plan"])

    system = f"""
You are a document-generation agent.

You have four tools: draft_content, review_content, revise_content,
and create_document.

Use the tools to complete this workflow:
1. Draft the document.
2. Review the draft.
3. If review finds important problems, revise it and review again.
4. Generate the Word document after review passes, or after the revision
   limit has been reached.

Call only one tool at a time. Never claim a tool succeeded before its result
is returned. Do not call a tool that has already completed successfully,
except revise_content when review feedback requires it.

The application enforces tool prerequisites and retry limits.
If a tool returns an error, explain the failure rather than claiming success.

Validated plan:
{plan.model_dump_json(indent=2)}
"""

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": f"Create a document for this request:\n{state['request']}",
        },
        *state.get("messages", []),
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOL_DEFINITIONS,
        tool_choice="auto",
        temperature=0.1,
        max_completion_tokens=1200,
    )

    if not response.choices:
        return {
            "agent_turns": turns,
            "error": "Groq returned no choices.",
            "finished": True,
            "execution_log": [
                *state.get("execution_log", []),
                "Agent stopped because Groq returned no choices.",
            ],
        }

    choice = response.choices[0]
    assistant_message = choice.message.model_dump(exclude_none=True)

    updated_messages = [
        *state.get("messages", []),
        assistant_message,
    ]

    tool_calls = choice.message.tool_calls or []

    if len(tool_calls) > 1:
        return {
            "messages": updated_messages,
            "agent_turns": turns,
            "error": "Agent requested multiple tools in one turn.",
            "finished": True,
            "execution_log": [
                *state.get("execution_log", []),
                "Agent stopped because it requested multiple tools.",
            ],
        }

    if not choice.message.tool_calls:
        return {
            "messages": updated_messages,
            "agent_turns": turns,
            "error": "The model stopped without completing the required tool workflow.",
            "finished": True,
            "execution_log": [
                *state.get("execution_log", []),
                "Agent stopped without requesting another tool.",
            ],
        }

    return {
        "messages": updated_messages,
        "agent_turns": turns,
        "execution_log": [
            *state.get("execution_log", []),
            f"Agent requested {len(choice.message.tool_calls)} tool call(s).",
        ],
    }


def tool_node(state: AgentState) -> dict:
    messages = list(state.get("messages", []))
    last_message = messages[-1]

    # Keep state changes local until each tool has passed its prerequisites.
    content = state.get("content", "")
    feedback = state.get("review_feedback", "")
    passed = state.get("review_passed", False)
    revisions = state.get("revisions", 0)
    output_path = state.get("output_path", "")
    tasks = [dict(task) for task in state.get("task_results", [])]
    logs = list(state.get("execution_log", []))
    tool_messages = []
    error = ""

    plan = DocumentPlan.model_validate(state["plan"])

    for call in last_message.get("tool_calls", []):
        function = call["function"]
        name = function["name"]

        try:
            args = json.loads(function.get("arguments") or "{}")
            if args != {}:
                raise ValueError("This tool accepts no arguments.")

            if name == "draft_content":
                if content:
                    raise ValueError("A draft already exists.")

                content = _call_llm(
                    "Write a complete Markdown document. Follow the user's "
                    "request and every required section. Do not invent facts. "
                    "Label assumptions clearly. Return only the document body.",
                    f"Request: {state['request']}\n"
                    f"Title: {plan.title}\n"
                    f"Sections: {json.dumps(plan.sections)}\n"
                    f"Assumptions: {json.dumps(plan.assumptions)}",
                )
                log_text_preview(
                    "DRAFT",
                    "Generated Draft Preview",
                    content,
                    limit=600,
                )
                tasks = _update_task(
                    {**state, "task_results": tasks}, name, "completed"
                )
                logs.append("Draft content generated.")

            elif name == "review_content":
                if not content:
                    raise ValueError("Draft the document before reviewing it.")
                log_text_preview(
                    "REVIEW",
                    "Input Draft Preview",
                    content,
                    limit=300,
                )
                review = _call_llm(
                    "Review the draft against the request and required sections. "
                    "Check for missing requirements, contradictions, unsupported "
                    "claims, and repetition. Return valid JSON with exactly two keys: "
                    "'passed' (boolean) and 'feedback' (string). Do not invent facts.",
                    f"Request: {state['request']}\n"
                    f"Required sections: {json.dumps(plan.sections)}\n"
                    f"Draft:\n{content}",
                )

                # Reject malformed review output rather than guessing its meaning.
                result = json.loads(review)
                if not isinstance(result, dict):
                    raise ValueError("Review response must be a JSON object.")

                if not isinstance(result.get("passed"), bool):
                    raise ValueError("Review 'passed' must be a boolean.")

                if not isinstance(result.get("feedback"), str):
                    raise ValueError("Review 'feedback' must be a string.")

                if set(result.keys()) != {"passed", "feedback"}:
                    raise ValueError(
                        "Review response contains unexpected or missing keys."
                    )

                passed = result["passed"]
                feedback = result["feedback"]

                
                log_data(
                    "REVIEW",
                    "Review Result",
                    {
                        "passed": passed,
                        "feedback": feedback,
                    },
                )
                if passed:
                    tasks = _update_task(
                        {**state, "task_results": tasks}, name, "completed"
                    )
                    logs.append("Review passed.")
                else:
                    tasks = _update_task(
                        {**state, "task_results": tasks}, name, "completed"
                    )
                    logs.append(f"Review identified issues: {feedback}")

            elif name == "revise_content":
                if not content or not feedback or passed:
                    raise ValueError("Revision requires a draft and review feedback.")
                if revisions >= MAX_REVISIONS:
                    raise ValueError("Maximum revision count reached.")

                content = _call_llm(
                    "Revise the document using the review feedback. "
                    "Preserve accurate information and all required sections. "
                    "Do not invent facts. Return only the complete Markdown body.",
                    f"Request: {state['request']}\n"
                    f"Feedback: {feedback}\n"
                    f"Draft:\n{content}",
                )
                log_text_preview(
                    "REVISION",
                    "Revised Draft Preview",
                    content,
                    limit=600,
                )
                revisions += 1
                passed = False
                tasks = _update_task(
                    {**state, "task_results": tasks}, name, "completed"
                )
                logs.append(f"Draft revised ({revisions}/{MAX_REVISIONS}).")

            elif name == "create_document":
                if not content:
                    raise ValueError("No document content exists.")
                if not passed and revisions < MAX_REVISIONS:
                    raise ValueError(
                        "Review must pass or the revision limit must be reached."
                    )
                log_data(
                    "DOCUMENT",
                    "Final Document Input",
                    {
                        "title": plan.title,
                        "document_type": plan.document_type,
                        "assumptions": plan.assumptions,
                        "content": content,
                    },
                )
                output_path = str(
                    create_docx(
                        title=plan.title,
                        document_type=plan.document_type,
                        content=content,
                        assumptions=plan.assumptions,
                    )
                )
                if not Path(output_path).is_file():
                    raise RuntimeError("Word document was not created.")

                tasks = _update_task(
                    {**state, "task_results": tasks}, name, "completed"
                )
                logs.append(f"Word document generated: {output_path}")

            else:
                raise ValueError(f"Unknown tool: {name}")

            result_text = json.dumps(
                {
                    "success": True,
                    "tool": name,
                    "message": f"{name} completed successfully.",
                }
            )

        except Exception as exc:
            error = f"{name} failed: {type(exc).__name__}: {exc}"
            tasks = _update_task({**state, "task_results": tasks}, name, "failed")
            logs.append(error)
            result_text = json.dumps({"success": False, "error": error})

        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": result_text,
            }
        )

        if error:
            break

    # All tool calls in a single model response receive a result.
    # If a call failed, later calls from the same response are not executed.
    if error:
        for call in last_message.get("tool_calls", [])[len(tool_messages) :]:
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(
                        {
                            "success": False,
                            "error": "Not executed because an earlier tool call failed.",
                        }
                    ),
                }
            )

    return {
        "messages": [*messages, *tool_messages],
        "content": content,
        "review_feedback": feedback,
        "review_passed": passed,
        "revisions": revisions,
        "output_path": output_path,
        "task_results": tasks,
        "execution_log": logs,
        "error": error,
        "finished": bool(error) or bool(output_path),
    }


def route_after_agent(state: AgentState) -> str:
    if state.get("finished") or state.get("error"):
        return "end"

    last_message = state["messages"][-1]
    if last_message.get("tool_calls"):
        return "tools"

    return "end"


def route_after_tools(state: AgentState) -> str:
    if state.get("finished") or state.get("error"):
        return "end"
    return "agent"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "end": END},
    )
    graph.add_conditional_edges(
        "tools",
        route_after_tools,
        {"agent": "agent", "end": END},
    )

    return graph.compile()


def execute_graph(request: str, plan: DocumentPlan) -> dict:
    graph = build_graph()

    initial_state: AgentState = {
        "request": request,
        "plan": plan.model_dump(),
        "content": "",
        "review_feedback": "",
        "review_passed": False,
        "revisions": 0,
        "agent_turns": 0,
        "messages": [],
        "task_results": [],
        "execution_log": ["LangGraph execution started."],
        "output_path": "",
        "error": "",
        "finished": False,
    }

    result = graph.invoke(initial_state)

    if result.get("error"):
        raise RuntimeError(result["error"])

    output_path = result.get("output_path", "")
    if not output_path or not Path(output_path).is_file():
        raise RuntimeError("The graph finished without generating a Word document.")

    return {
        "plan": result.get("task_results", []),
        "assumptions": plan.assumptions,
        "execution_log": result.get("execution_log", []),
        "document_path": str(Path(output_path).resolve()),
        "title": plan.title,
    }
