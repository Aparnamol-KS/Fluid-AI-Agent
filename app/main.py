from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
import logging


from app.agent import (
    AgentExecutionError,
    execute_plan,
    generate_plan,
)
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
    try:
        plan = await run_in_threadpool(generate_plan, payload.request)
    except Exception as exc:
        logger.exception("Agent planning failed")

        raise HTTPException(
            status_code=502,
            detail=(
                f"Planning failed: {type(exc).__name__}. "
                "Check the server terminal for details."
            ),
        ) from exc

    try:
        result = await run_in_threadpool(
            execute_plan,
            payload.request,
            plan,
        )
    except AgentExecutionError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "message": str(exc),
                "plan": exc.plan,
                "execution_log": exc.execution_log,
            },
        ) from exc

    filename = Path(result["document_path"]).name

    return AgentResponse(
        message="The agent completed the plan and generated a Word document.",
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
        raise HTTPException(status_code=400, detail="Invalid filename.")

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
