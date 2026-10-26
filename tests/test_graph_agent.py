import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import graph_agent as agent
from app import main


def make_state():
    """Create a minimal state for testing individual graph tools."""
    return {
        "request": "Write a short technical report about AI.",
        "plan": {},
        "content": "",
        "review_feedback": "",
        "review_passed": False,
        "revisions": 0,
        "agent_turns": 0,
        "messages": [],
        "task_results": [],
        "execution_log": [],
        "output_path": "",
        "error": "",
        "finished": False,
    }


def make_tool_state(name, state=None):
    """Attach one simulated Groq tool call to a state."""
    state = state or make_state()

    state["messages"] = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_test_1",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": "{}",
                    },
                }
            ],
        }
    ]
    return state


@pytest.fixture
def fake_plan(monkeypatch):
    """Provide a predictable plan with the methods used by the agent."""

    plan = SimpleNamespace(
        title="AI Technical Report",
        document_type="technical_report",
        sections=["Introduction", "Conclusion"],
        assumptions=[],
        model_dump_json=lambda indent=2: json.dumps(
            {
                "title": "AI Technical Report",
                "document_type": "technical_report",
                "sections": ["Introduction", "Conclusion"],
                "assumptions": [],
            },
            indent=indent,
        ),
    )

    monkeypatch.setattr(
        agent.DocumentPlan,
        "model_validate",
        lambda value: plan,
    )

    return plan


def test_route_after_agent_sends_tool_call_to_tools():
    state = make_state()
    state["messages"] = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "call_1"}],
        }
    ]

    assert agent.route_after_agent(state) == "tools"


def test_route_after_agent_stops_when_finished():
    state = make_state()
    state["finished"] = True

    assert agent.route_after_agent(state) == "end"


def test_route_after_tools_returns_to_agent():
    state = make_state()

    assert agent.route_after_tools(state) == "agent"


def test_draft_content_tool(monkeypatch, fake_plan):
    state = make_tool_state("draft_content")
    monkeypatch.setattr(
        agent,
        "_call_llm",
        lambda system, user: "# Introduction\nAI is useful.\n\n# Conclusion\nDone.",
    )

    result = agent.tool_node(state)

    assert "AI is useful" in result["content"]
    assert result["error"] == ""
    assert result["finished"] is False
    assert result["task_results"][0]["status"] == "completed"


def test_review_content_passes(monkeypatch, fake_plan):
    state = make_tool_state("review_content")
    state["content"] = "A complete draft."

    monkeypatch.setattr(
        agent,
        "_call_llm",
        lambda system, user: json.dumps(
            {"passed": True, "feedback": "Looks complete."}
        ),
    )

    result = agent.tool_node(state)

    assert result["review_passed"] is True
    assert result["review_feedback"] == "Looks complete."
    assert result["error"] == ""


def test_review_rejects_invalid_json(monkeypatch, fake_plan):
    state = make_tool_state("review_content")
    state["content"] = "A draft."

    monkeypatch.setattr(
        agent,
        "_call_llm",
        lambda system, user: "This is not JSON",
    )

    result = agent.tool_node(state)

    assert result["error"]
    assert result["finished"] is True


def test_revision_updates_content_and_counter(monkeypatch, fake_plan):
    state = make_tool_state("revise_content")
    state["content"] = "An incomplete draft."
    state["review_feedback"] = "Add a conclusion."
    state["review_passed"] = False

    monkeypatch.setattr(
        agent,
        "_call_llm",
        lambda system, user: "A revised draft with a conclusion.",
    )

    result = agent.tool_node(state)

    assert result["content"] == "A revised draft with a conclusion."
    assert result["revisions"] == 1
    assert result["review_passed"] is False
    assert result["error"] == ""


def test_revision_limit_is_enforced(fake_plan):
    state = make_tool_state("revise_content")
    state["content"] = "Draft."
    state["review_feedback"] = "Needs improvement."
    state["revisions"] = agent.MAX_REVISIONS

    result = agent.tool_node(state)

    assert result["error"]
    assert "Maximum revision count reached" in result["error"]
    assert result["finished"] is True


def test_document_creation_requires_review_or_revision_limit(monkeypatch, fake_plan):
    state = make_tool_state("create_document")
    state["content"] = "A draft."
    state["review_passed"] = False
    state["revisions"] = 0

    result = agent.tool_node(state)

    assert result["error"]
    assert result["finished"] is True


def test_document_creation_succeeds_after_review(monkeypatch, fake_plan, tmp_path):
    output_file = tmp_path / "report.docx"
    output_file.write_bytes(b"test document")

    state = make_tool_state("create_document")
    state["content"] = "A reviewed draft."
    state["review_passed"] = True

    monkeypatch.setattr(
        agent,
        "create_docx",
        lambda **kwargs: output_file,
    )

    result = agent.tool_node(state)

    assert Path(result["output_path"]).is_file()
    assert result["error"] == ""
    assert result["finished"] is True


def test_complete_graph_workflow_without_groq(monkeypatch, fake_plan, tmp_path):
    """Verify that LangGraph runs the agent/tool loop to completion."""

    # Simulate the model requesting one tool per turn.
    tool_sequence = [
        "draft_content",
        "review_content",
        "create_document",
    ]
    requested_tools = []

    def fake_chat_completion_create(**kwargs):
        # Capture the model's requested tool.
        # messages = kwargs["messages"]

        # Each mocked response requests the next tool.
        tool_name = tool_sequence[len(requested_tools)]
        requested_tools.append(tool_name)

        message = SimpleNamespace(
            role="assistant",
            content=None,
            tool_calls=[
                SimpleNamespace(
                    id=f"call_{len(requested_tools)}",
                    type="function",
                    function=SimpleNamespace(
                        name=tool_name,
                        arguments="{}",
                    ),
                )
            ],
            model_dump=lambda exclude_none=True: {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{len(requested_tools)}",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": "{}",
                        },
                    }
                ],
            },
        )

        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    # Mock Groq chat completions.
    monkeypatch.setattr(
        agent.client.chat.completions,
        "create",
        fake_chat_completion_create,
    )

    # Mock the LLM used by draft and review tools.
    def fake_llm(system, user):
        if "Return valid JSON" in system:
            return json.dumps(
                {
                    "passed": True,
                    "feedback": "The document meets the requirements.",
                }
            )

        return "# Introduction\nAI helps solve problems.\n\n# Conclusion\nAI has limitations."

    monkeypatch.setattr(agent, "_call_llm", fake_llm)

    # Use a temporary DOCX path instead of writing a real document.
    output_file = tmp_path / "test_report.docx"
    output_file.write_bytes(b"mock DOCX content")

    monkeypatch.setattr(
        agent,
        "create_docx",
        lambda **kwargs: output_file,
    )

    # Run the actual LangGraph graph.
    graph = agent.build_graph()
    result = graph.invoke(
        {
            "request": "Write a technical report about AI.",
            "plan": {
                "title": fake_plan.title,
                "document_type": fake_plan.document_type,
                "sections": fake_plan.sections,
                "assumptions": fake_plan.assumptions,
                "tasks": [
                    {"action": "draft_content"},
                    {"action": "review_content"},
                    {"action": "create_document"},
                ],
            },
            "content": "",
            "review_feedback": "",
            "review_passed": False,
            "revisions": 0,
            "agent_turns": 0,
            "messages": [],
            "task_results": [],
            "execution_log": [],
            "output_path": "",
            "error": "",
            "finished": False,
        }
    )

    assert requested_tools == tool_sequence
    assert result["error"] == ""
    assert result["finished"] is True
    assert result["review_passed"] is True
    assert Path(result["output_path"]).is_file()


@pytest.fixture
def api_client():
    return TestClient(main.app)


def test_agent_endpoint_returns_document_response(monkeypatch, api_client, tmp_path):
    """Verify the endpoint response without calling Groq."""

    fake_plan = SimpleNamespace(
        title="AI Technical Report",
        assumptions=["Examples are illustrative."],
    )

    output_file = tmp_path / "report.docx"
    output_file.write_bytes(b"mock DOCX content")

    monkeypatch.setattr(
        main,
        "generate_plan",
        lambda request: fake_plan,
    )

    monkeypatch.setattr(
        main,
        "execute_graph",
        lambda request, plan: {
            "plan": [
                {
                    "id": "T1",
                    "description": "Draft Content",
                    "action": "draft_content",
                    "status": "completed",
                },
                {
                    "id": "T2",
                    "description": "Review Content",
                    "action": "review_content",
                    "status": "completed",
                },
                {
                    "id": "T3",
                    "description": "Create Document",
                    "action": "create_document",
                    "status": "completed",
                },
            ],
            "assumptions": ["Examples are illustrative."],
            "execution_log": ["LangGraph execution started."],
            "document_path": str(output_file),
            "title": "AI Technical Report",
        },
    )

    response = api_client.post(
        "/agent",
        json={"request": "Write a technical report about AI."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "AI Technical Report"
    assert body["document_url"] == "/documents/report.docx"
    assert len(body["plan"]) == 3


def test_agent_endpoint_handles_graph_failure(monkeypatch, api_client):
    """Verify graph failures become controlled HTTP responses."""

    fake_plan = SimpleNamespace(title="AI Report", assumptions=[])

    monkeypatch.setattr(
        main,
        "generate_plan",
        lambda request: fake_plan,
    )

    def fail_graph(request, plan):
        raise RuntimeError("Simulated graph failure")

    monkeypatch.setattr(main, "execute_graph", fail_graph)

    response = api_client.post(
        "/agent",
        json={"request": "Write a technical report about AI."},
    )

    assert response.status_code == 502
    assert "Agent execution failed" in response.json()["detail"]


def test_health_endpoint(api_client):
    response = api_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_complete_graph_workflow_with_revision(monkeypatch, fake_plan, tmp_path):
    """Verify the graph revises a rejected draft before creating the DOCX."""

    tool_sequence = [
        "draft_content",
        "review_content",
        "revise_content",
        "review_content",
        "create_document",
    ]
    requested_tools = []
    review_count = 0

    def fake_chat_completion_create(**kwargs):
        tool_name = tool_sequence[len(requested_tools)]
        requested_tools.append(tool_name)

        call_id = f"call_{len(requested_tools)}"

        message = SimpleNamespace(
            role="assistant",
            content=None,
            tool_calls=[
                SimpleNamespace(
                    id=call_id,
                    type="function",
                    function=SimpleNamespace(
                        name=tool_name,
                        arguments="{}",
                    ),
                )
            ],
            model_dump=lambda exclude_none=True: {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": "{}",
                        },
                    }
                ],
            },
        )

        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(
        agent.client.chat.completions,
        "create",
        fake_chat_completion_create,
    )

    def fake_llm(system, user):
        nonlocal review_count

        if "Return valid JSON" in system:
            review_count += 1

            if review_count == 1:
                return json.dumps(
                    {
                        "passed": False,
                        "feedback": "Add a conclusion section.",
                    }
                )

            return json.dumps(
                {
                    "passed": True,
                    "feedback": "The revised document meets requirements.",
                }
            )

        if "Revise the document" in system:
            return (
                "# Introduction\nAI is useful.\n\n"
                "# Conclusion\nAI has benefits and limitations."
            )

        return "# Introduction\nAI is useful."

    monkeypatch.setattr(agent, "_call_llm", fake_llm)

    output_file = tmp_path / "revised_report.docx"
    output_file.write_bytes(b"mock DOCX content")

    monkeypatch.setattr(
        agent,
        "create_docx",
        lambda **kwargs: output_file,
    )

    graph = agent.build_graph()
    result = graph.invoke(
        {
            "request": "Write a technical report about AI.",
            "plan": {
                "title": fake_plan.title,
                "document_type": fake_plan.document_type,
                "sections": fake_plan.sections,
                "assumptions": fake_plan.assumptions,
            },
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
    )

    assert requested_tools == tool_sequence
    assert review_count == 2
    assert result["error"] == ""
    assert result["finished"] is True
    assert result["review_passed"] is True
    assert result["revisions"] == 1
    assert "Conclusion" in result["content"]
    assert Path(result["output_path"]).is_file()

    tasks = {task["action"]: task["status"] for task in result["task_results"]}
    assert tasks["draft_content"] == "completed"
    assert tasks["review_content"] == "completed"
    assert tasks["revise_content"] == "completed"
    assert tasks["create_document"] == "completed"

    assert any("Draft revised (1/2)." in entry for entry in result["execution_log"])
    assert any("Review passed." in entry for entry in result["execution_log"])
