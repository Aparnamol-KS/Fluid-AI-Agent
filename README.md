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

This project explores two approaches to AI-powered document generation.

### 1. `app/agent.py` — Plan-Based Document Generation

This is the initial implementation, designed around a structured document-generation plan.

When a user submits a request, the system uses an LLM to identify the document type, title, assumptions, sections, and tasks needed to create the document. The resulting plan provides a structured representation of what the document should contain.


### 2. `app/graph_agent.py` — Model-Directed Agent Workflow

The second implementation extends the system with a more flexible, model-directed workflow using Groq tool calling and LangGraph.

Instead of relying only on a predefined sequence of operations, the model can select the next tool based on the current state and review results. The workflow supports:

- Drafting content based on the document plan.
- Reviewing the generated content.
- Revising the draft when the review identifies issues.
- Creating the final Word document when the content is ready.

The workflow also tracks execution state and task results, with limits on revisions and agent turns to prevent uncontrolled execution.

The initial implementation focuses on **structured planning**, while the second introduces **model-directed execution and conditional workflows**.


