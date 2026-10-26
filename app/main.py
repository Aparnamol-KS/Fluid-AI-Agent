import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from app.agent import generate_plan
from app.graph_agent import execute_graph, log_data
from app.schemas import AgentRequest, AgentResponse

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fluid AI Autonomous Agent",
    description=(
        "An LLM-powered agent that plans tasks, executes them, "
        "and generates business documents."
    ),
    version="1.0.0",
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/agent", response_model=AgentResponse)
async def run_agent(payload: AgentRequest):
    # Step 1: Generate and validate the document plan.
    try:
        plan = await run_in_threadpool(
            generate_plan,
            payload.request,
        )

        log_data("PLANNING", "Generated Document Plan", plan)
    except Exception as exc:
        logger.exception("Agent planning failed")
        raise HTTPException(
            status_code=502,
            detail=(
                f"Planning failed: {type(exc).__name__}. "
                "Check the server terminal for details."
            ),
        ) from exc

    # Step 2: Execute the plan using the LangGraph agent.
    try:
        result = await run_in_threadpool(
            execute_graph,
            payload.request,
            plan,
        )
    except Exception as exc:
        logger.exception("LangGraph execution failed")
        raise HTTPException(
            status_code=502,
            detail=(
                f"Agent execution failed: {type(exc).__name__}. "
                "Check the server terminal for details."
            ),
        ) from exc

    # Step 3: Return the existing API response format.
    filename = Path(result["document_path"]).name

    return AgentResponse(
        message="The agent completed the workflow and generated a Word document.",
        title=result["title"],
        plan=result["plan"],
        assumptions=result["assumptions"],
        execution_log=result["execution_log"],
        document_url=f"/documents/{filename}",
    )


@app.get("/documents/{filename}")
def download_document(filename: str):
    # Only serve DOCX files from the designated output directory.
    if Path(filename).name != filename or not filename.endswith(".docx"):
        raise HTTPException(
            status_code=400,
            detail="Invalid filename.",
        )

    file_path = (OUTPUT_DIR / filename).resolve()

    if file_path.parent != OUTPUT_DIR.resolve() or not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return FileResponse(
        path=file_path,
        media_type=(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ),
        filename=filename,
    )
