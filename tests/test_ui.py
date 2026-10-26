from pathlib import Path

from app import ui


def test_gradio_ui_displays_revision_workflow(monkeypatch):
    """Test the Gradio callback without calling FastAPI or Groq."""

    fake_result = {
        "message": "The agent completed the workflow and generated a Word document.",
        "title": "AI Technical Report",
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
                "description": "Revise Content",
                "action": "revise_content",
                "status": "completed",
            },
            {
                "id": "T4",
                "description": "Create Document",
                "action": "create_document",
                "status": "completed",
            },
        ],
        "execution_log": [
            "LangGraph execution started.",
            "Agent requested 1 tool call(s).",
            "Draft content generated.",
            "Review identified issues: Add a conclusion section.",
            "Draft revised (1/2).",
            "Review passed.",
            "Word document generated: report.docx",
        ],
        "document_url": "/documents/report.docx",
    }

    class FakeResponse:
        def __init__(self, status_code=200, data=None, content=b""):
            self.status_code = status_code
            self._data = data
            self.content = content
            self.ok = 200 <= status_code < 300

        def json(self):
            return self._data

        def raise_for_status(self):
            if not self.ok:
                raise ui.requests.HTTPError(f"HTTP {self.status_code}")

    # Simulate the FastAPI response.
    monkeypatch.setattr(
        ui.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(data=fake_result),
    )

    # Simulate downloading the generated DOCX.
    monkeypatch.setattr(
        ui.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(content=b"Mock Word document"),
    )

    status, plan_html, log_html, document_path = ui.generate_document(
        "Write a technical report about AI."
    )

    try:
        assert "AI Technical Report" in status
        assert "completed the workflow" in status

        # All four tasks should be displayed.
        assert "Draft Content" in plan_html
        assert "Review Content" in plan_html
        assert "Revise Content" in plan_html
        assert "Create Document" in plan_html

        # The revision and subsequent review should be visible.
        assert "Draft revised (1/2)." in log_html
        assert "Review identified issues" in log_html
        assert "Review passed." in log_html

        # Gradio should receive a downloadable file.
        assert document_path is not None
        assert Path(document_path).is_file()
        assert Path(document_path).read_bytes() == b"Mock Word document"

    finally:
        # Clean up the temporary file created by the UI callback.
        if document_path and Path(document_path).exists():
            Path(document_path).unlink()
