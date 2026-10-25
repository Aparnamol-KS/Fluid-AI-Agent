import html
import os
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import gradio as gr
import requests

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
).rstrip("/")


def render_plan(plan):
    """Render execution tasks as readable status cards."""
    if not plan:
        return """
        <div style="color: var(--body-text-color-subdued);">
            No execution plan available.
        </div>
        """

    cards = []

    for index, task in enumerate(plan, start=1):
        description = html.escape(str(task.get("description", "Untitled task")))
        status = str(task.get("status", "pending")).lower()

        if status == "completed":
            icon = "✓"
            color = "#16a34a"
            label = "Completed"
        elif status == "failed":
            icon = "!"
            color = "#dc2626"
            label = "Failed"
        else:
            icon = "○"
            color = "#64748b"
            label = html.escape(status.title())

        cards.append(f"""
        <div style="
            display: flex;
            align-items: flex-start;
            gap: 14px;
            padding: 16px;
            margin-bottom: 10px;
            border: 1px solid var(--border-color-primary);
            border-radius: 12px;
            background: var(--background-fill-secondary);
        ">
            <div style="
                display: flex;
                align-items: center;
                justify-content: center;
                min-width: 30px;
                height: 30px;
                border-radius: 50%;
                background: {color}18;
                color: {color};
                font-weight: 700;
            ">{icon}</div>

            <div style="flex: 1;">
                <div style="font-weight: 600; margin-bottom: 5px;">
                    Step {index}
                </div>
                <div>{description}</div>
                <div style="
                    color: {color};
                    font-size: 13px;
                    margin-top: 7px;
                    font-weight: 600;
                ">{label}</div>
            </div>
        </div>
        """)

    return "\n".join(cards)


def generate_document(request_text: str):
    """Call FastAPI and return readable execution results."""

    if not request_text or not request_text.strip():
        return (
            "Please enter a document request.",
            render_plan([]),
            None,
        )

    try:
        response = requests.post(
            f"{API_BASE_URL}/agent",
            json={"request": request_text.strip()},
            timeout=(10, 300),
        )

        if not response.ok:
            try:
                error_data = response.json()
                detail = error_data.get("detail", error_data)

                # Failed execution plans are nested inside detail.
                if not isinstance(detail, dict):
                    detail = {}

                plan = detail.get("plan", [])
                message = detail.get("message", "Document generation failed.")

            except ValueError:
                plan = []
                message = response.text[:1000]

            return (
                f"**Generation failed:** {message}",
                render_plan(plan),
                None,
            )

        result = response.json()
        plan = result.get("plan", [])
        document_url = result.get("document_url")

        if not document_url:
            return (
                "**Generation failed:** The API returned no document URL.",
                render_plan(plan),
                None,
            )

        parsed_url = urlparse(document_url)
        filename = Path(unquote(parsed_url.path)).name

        if (
            parsed_url.scheme
            or parsed_url.netloc
            or parsed_url.query
            or parsed_url.fragment
            or not filename.endswith(".docx")
            or filename in ("", ".", "..")
        ):
            return (
                "**Generation failed:** Invalid document URL.",
                render_plan(plan),
                None,
            )

        file_response = requests.get(
            f"{API_BASE_URL}/documents/{filename}",
            timeout=(10, 60),
        )
        file_response.raise_for_status()

        with tempfile.NamedTemporaryFile(
            suffix=".docx",
            prefix="fluid_ai_",
            delete=False,
        ) as temp_file:
            temp_file.write(file_response.content)
            document_path = temp_file.name

        return (
            result.get("message", "Document generated successfully."),
            render_plan(plan),
            document_path,
        )

    except requests.RequestException as exc:
        return (
            f"**Request error:** {html.escape(str(exc))}",
            render_plan([]),
            None,
        )

    except (ValueError, OSError) as exc:
        return (
            f"**Response error:** {html.escape(str(exc))}",
            render_plan([]),
            None,
        )


with gr.Blocks(title="Fluid AI Document Agent") as demo:
    gr.Markdown(
        """
        # Fluid AI — Document Agent
        Turn a written request into a structured, downloadable Word document.

        Enter your requirements below. The agent will plan, draft, review,
        and generate the document using the existing FastAPI backend.
        """
    )

    request_input = gr.Textbox(
        label="Document request",
        placeholder=(
            "Create a technical project proposal with objectives, "
            "architecture, schedule, testing, and risks..."
        ),
        lines=7,
    )

    generate_button = gr.Button(
        "Generate Word Document",
        variant="primary",
    )

    status_output = gr.Markdown(label="Result")

    with gr.Accordion("Execution plan", open=True):
        plan_output = gr.HTML()

    document_output = gr.File(
        label="Download generated Word document",
        interactive=False,
    )

    generate_button.click(
        fn=generate_document,
        inputs=[request_input],
        outputs=[
            status_output,
            plan_output,
            document_output,
        ],
        concurrency_limit=1,
    )


if __name__ == "__main__":
    demo.queue().launch()
