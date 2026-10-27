# AI Document Generation Agent

An AI-powered document generation system that converts natural-language requests into structured Word documents using model-directed tool calling and a review-and-revision workflow.

## Features

- **AI Planning:** *Generates a document plan with a title, sections, assumptions, and tasks.*
- **Agent Workflow:** *Uses Groq tool calling and LangGraph to select and execute tools.*
- **Content Review:** *Reviews generated content and supports revisions when necessary.*
- **Word Generation:** *Creates downloadable `.docx` documents.*
- **Execution Tracking:** *Displays task statuses, execution logs, and document download links.*
- **Testing:** *Includes automated tests for API endpoints and agent workflows.*

## Architecture

The application follows this workflow:

<img width="1622" height="815" alt="AI Document Generation Agent Architecture" src="https://github.com/user-attachments/assets/e701de8d-8c68-4d2e-8d07-3d0ad09b06cf" />

The agent can draft content, review it, revise it if required, and create the final document. Revision and agent-turn limits help prevent endless execution loops.

## Tech Stack

- **Language:** Python
- **LLM:** Groq API
- **Agent Orchestration:** LangGraph
- **Backend:** FastAPI
- **Frontend:** Gradio
- **Document Generation:** Python-docx
- **Testing:** Pytest
- **Environment Management:** uv

## Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- A Groq API key

## Setup

1. Clone the repository and navigate to the project directory.

   ```bash
   https://github.com/Aparnamol-KS/Fluid-AI-Agent.git
   cd Fluid-AI-Agent
   ```

2. Install dependencies.

   ```bash
   uv sync
   ```

3. Create a `.env` file in the project root:

   ```env
   GROQ_API_KEY=your_groq_api_key
   ```

## Run the Application

Start the FastAPI backend:

```bash
uv run uvicorn app.main:app --reload
```

In a separate terminal, start the Gradio interface:

```bash
uv run python app/ui.py
```

Open the local Gradio URL displayed in the terminal to submit document-generation requests.

## Run Tests

```bash
uv run pytest
```

The tests cover the API and agent workflow, including content revision scenarios using mocked responses.

## Example Request

> Create a project proposal for a task management web application.

The agent plans the document, generates and reviews the content, and produces a downloadable Word document.

## Document Generation Approaches

The project evolved from a predefined workflow to a model-directed agent workflow.

- **`app/agent.py` — Planning and Original Workflow:** Generates a structured document plan using the LLM. The original `execute_plan()` function follows predefined steps to execute the plan.
- **`app/graph_agent.py` — Agent Workflow:** Uses Groq tool calling and LangGraph to manage drafting, reviewing, conditional revision, and Word document creation. The model selects tools based on the current workflow state and review results.

The current application retains `generate_plan()` from `agent.py` and uses `graph_agent.py` for execution, enabling more flexible, model-directed workflows.
