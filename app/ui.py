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
    """Render LangGraph task results as status cards."""
    if not plan:
        return (
            '<div style="color: var(--body-text-color-subdued);">'
            "No execution tasks available."
            "</div>"
        )

    cards = []

    for index, task in enumerate(plan, start=1):
        description = html.escape(str(task.get("description", "Untitled task")))
        action = html.escape(str(task.get("action", "")))
        status = str(task.get("status", "pending")).lower()

        if status == "completed":
            icon, color, label = "✓", "#16a34a", "Completed"
        elif status == "failed":
            icon, color, label = "!", "#dc2626", "Failed"
        else:
            icon, color, label = "○", "#64748b", status.title()

        cards.append(
            f"""
            <div style="
                display:flex;
                align-items:flex-start;
                gap:14px;
                padding:14px;
                margin-bottom:10px;
                border:1px solid var(--border-color-primary);
                border-radius:12px;
                background:var(--background-fill-secondary);
            ">
                <div style="
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    min-width:30px;
                    height:30px;
                    border-radius:50%;
                    background:{color}18;
                    color:{color};
                    font-weight:700;
                ">{icon}</div>

                <div style="flex:1; min-width:0;">
                    <div style="font-weight:600;">
                        Step {index}: {description}
                    </div>
                    <div style="
                        color:var(--body-text-color-subdued);
                        font-size:12px;
                        margin-top:4px;
                        overflow-wrap:anywhere;
                    ">{action}</div>
                    <div style="
                        color:{color};
                        font-size:13px;
                        margin-top:7px;
                        font-weight:600;
                    ">{html.escape(label)}</div>
                </div>
            </div>
            """
        )

    return "\n".join(cards)


def render_execution_log(log):
    """Render the execution trace, including revisions."""
    if not log:
        return (
            '<div style="color: var(--body-text-color-subdued);">'
            "No execution log available."
            "</div>"
        )

    items = []

    for index, entry in enumerate(log, start=1):
        text = html.escape(str(entry))

        if "failed" in str(entry).lower() or "error" in str(entry).lower():
            color = "#dc2626"
            marker = "!"
        elif "revised" in str(entry).lower():
            color = "#d97706"
            marker = "↻"
        elif "passed" in str(entry).lower() or "generated" in str(entry).lower():
            color = "#16a34a"
            marker = "✓"
        else:
            color = "#64748b"
            marker = "•"

        items.append(
            f"""
            <div style="
                display:flex;
                align-items:flex-start;
                gap:10px;
                padding:8px 0;
                border-bottom:1px solid var(--border-color-primary);
            ">
                <span style="color:{color}; font-weight:700;">
                    {marker}
                </span>
                <span style="overflow-wrap:anywhere;">{text}</span>
            </div>
            """
        )

    return "\n".join(items)


def extract_error(response):
    """Extract useful error information from FastAPI responses."""
    try:
        data = response.json()
    except ValueError:
        return response.text[:1000], []

    detail = data.get("detail", data.get("message", data))

    # Support structured errors if the API adds them in the future.
    if isinstance(detail, dict):
        message = detail.get(
            "message",
            detail.get("error", str(detail)),
        )
        plan = detail.get("plan", [])
    else:
        message = str(detail)
        plan = data.get("plan", [])

    return message, plan if isinstance(plan, list) else []


def generate_document(request_text: str):
    """Call FastAPI and return status, tasks, log, and DOCX file."""
    empty_plan = render_plan([])
    empty_log = render_execution_log([])

    if not request_text or not request_text.strip():
        return (
            "Please enter a document request.",
            empty_plan,
            empty_log,
            None,
        )

    try:
        response = requests.post(
            f"{API_BASE_URL}/agent",
            json={"request": request_text.strip()},
            timeout=(10, 300),
        )

        if not response.ok:
            message, plan = extract_error(response)

            return (
                f"**Generation failed ({response.status_code}):** "
                f"{html.escape(message)}",
                render_plan(plan),
                empty_log,
                None,
            )

        result = response.json()
        plan = result.get("plan", [])
        execution_log = result.get("execution_log", [])
        document_url = result.get("document_url")

        if not document_url:
            return (
                "**Generation failed:** The API returned no document URL.",
                render_plan(plan),
                render_execution_log(execution_log),
                None,
            )

        # Only accept a relative document path, not an arbitrary URL.
        parsed_url = urlparse(document_url)
        filename = Path(unquote(parsed_url.path)).name

        if (
            parsed_url.scheme
            or parsed_url.netloc
            or parsed_url.query
            or parsed_url.fragment
            or not filename.lower().endswith(".docx")
            or filename in ("", ".", "..")
        ):
            return (
                "**Generation failed:** Invalid document URL.",
                render_plan(plan),
                render_execution_log(execution_log),
                None,
            )

        file_response = requests.get(
            f"{API_BASE_URL}/documents/{filename}",
            timeout=(10, 60),
        )
        file_response.raise_for_status()

        # Save the API response so Gradio can serve it as a download.
        with tempfile.NamedTemporaryFile(
            suffix=".docx",
            prefix="fluid_ai_",
            delete=False,
        ) as temp_file:
            temp_file.write(file_response.content)
            document_path = temp_file.name

        title = html.escape(str(result.get("title", "Generated document")))
        message = html.escape(
            str(result.get("message", "Document generated successfully."))
        )

        status = f"### {title}\n\n{message}"

        return (
            status,
            render_plan(plan),
            render_execution_log(execution_log),
            document_path,
        )

    except requests.HTTPError as exc:
        return (
            f"**Download/API error:** {html.escape(str(exc))}",
            empty_plan,
            empty_log,
            None,
        )

    except requests.RequestException as exc:
        return (
            f"**Connection error:** {html.escape(str(exc))}\n\n"
            "Check that the FastAPI backend is running and API_BASE_URL "
            "points to the correct address.",
            empty_plan,
            empty_log,
            None,
        )

    except (ValueError, OSError) as exc:
        return (
            f"**Response/file error:** {html.escape(str(exc))}",
            empty_plan,
            empty_log,
            None,
        )


with gr.Blocks(title="Fluid AI Document Agent") as demo:
    gr.Markdown(
        """
        # Fluid AI — Document Agent

        Turn a written request into a structured Word document.

        The LangGraph agent selects tools to draft, review, revise when
        needed, and generate the final document.
        """
    )

    request_input = gr.Textbox(
        label="Document request",
        placeholder=(
            "Create a technical project proposal with objectives, "
            "architecture, schedule, testing, and risks..."
        ),
        lines=7,
        max_lines=14,
    )

    generate_button = gr.Button(
        "Generate Word Document",
        variant="primary",
    )

    status_output = gr.Markdown(label="Result")

    with gr.Accordion("Execution plan", open=True):
        plan_output = gr.HTML()

    with gr.Accordion("Agent execution log", open=True):
        log_output = gr.HTML()

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
            log_output,
            document_output,
        ],
        concurrency_limit=1,
    )


if __name__ == "__main__":
    demo.queue().launch()
