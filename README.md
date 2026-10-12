# Task-Automating AI Agent

A framework-free educational reference implementation of an autonomous AI agent built in Python using Google Gemini's native function calling via the official `google-genai` SDK.

The agent takes natural language goals (e.g., *"Search the latest Python release, summarize it, and save it to notes.md"*), dynamically selects and sequences tools, executes multi-step workflows through an LLM-driven ReAct loop, and returns grounded final answers.

---

## Key Highlights

- **Pure Python Mechanics**: No LangChain, AutoGen, or external agent frameworks. Every mechanical step—from schema generation to state management—is clear and visible.
- **Native Gemini Function Calling**: Uses the official `google-genai` SDK. Internally maintains a `role="tool"` history and converts function responses to SDK-compliant `user` Content messages with `Part.from_function_response`. Multiple tool results from a single turn are merged into a single user message.
- **Model Turn Preservation**: Preserves the original model `Content` object to retain thought signatures on Gemini 3.x models.
- **Pydantic Validation & Self-Correction**: LLM-provided arguments are validated against Pydantic models. Argument errors are returned to the model as observations so it can self-correct dynamically.
- **Sandboxed File Operations**: All file reading and writing is restricted to `./workspace` using `Path.is_relative_to()`. Path traversal (`../`), sibling directories (`workspace_evil`), and absolute path escapes are blocked.
- **Calculator with DoS Protection**: AST-based math evaluation blocking `eval()`, with caps on operand magnitude, exponent size, and expression length.
- **Rule-Based Summarizer**: A lightweight heuristic sentence extraction tool that ranks and extracts key takeaway sentences by position and keyword frequency.
- **Human-in-the-Loop Safety**: Risky tools (`file_write`) can be gated with interactive terminal approval using `--approve-risky`.
- **Rate-Limit Resilience**: Targeted retries with exponential backoff and jitter for HTTP 429 (`RESOURCE_EXHAUSTED`), 500, and 503 errors, while failing immediately on non-retryable 4xx client errors.
- **Prompt Tool Timeout**: ThreadPoolExecutor tool execution with non-blocking shutdown so hanging tools return promptly.
- **Observability**: Step-by-step event logging to console in `--verbose` mode and to structured JSONL logs (`logs/run_<timestamp>.jsonl`).
- **Hermetic Offline Testing**: 36 pytest unit and integration tests using a mocked LLM client with zero network or API key dependencies.

---

## Directory Structure

```text
task_automating_ai/
│
├── agent/                      # Core agent framework
│   ├── __init__.py
│   ├── agent.py                # ReAct execution loop & error handling
│   ├── config.py               # Pydantic Settings & environment variables
│   ├── llm_client.py           # Provider-agnostic client (Gemini & Mock)
│   ├── logger.py               # JSONL structured logger and console tracer
│   └── tools/                  # Tool system
│       ├── __init__.py
│       ├── base.py             # Tool base class, schema generation & timeouts
│       ├── registry.py         # Tool registry with security allowlist
│       ├── calculator.py       # Safe AST-based math evaluator with DoS limits
│       ├── web_search.py       # DuckDuckGo search (via ddgs)
│       ├── file_ops.py         # Sandboxed file I/O (read, write, list)
│       ├── datetime_tool.py    # Temporal anchoring tool
│       └── summarizer.py       # Rule-based sentence extraction tool
│
├── main.py                     # CLI entry point
├── scripts/
│   └── list_models.py          # Lists Gemini models available to your API key
│
├── tests/                      # Automated test suite (Mocked LLM, offline)
│   ├── conftest.py             # Pytest fixtures & mock environment
│   ├── test_agent_loop.py      # Multi-step, loop guard & approval tests
│   ├── test_llm_client.py      # Message conversion & retry tests
│   ├── test_registry.py        # Allowlist & schema generation tests
│   ├── test_tools.py           # Sandboxing, timeouts & tool unit tests
│   └── test_live.py            # Optional single live API test (-m live)
│
├── docs/                       # Architectural & interview documentation
│   ├── EXPLAINER.md            # Architecture walkthrough & traced run
│   └── INTERVIEW_PREP.md       # Resume bullets, 2-min pitch & 20 Q&As
│
├── workspace/                  # Sandboxed directory for agent file I/O
├── logs/                       # JSONL run traces
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Pytest configuration
├── .env.example                # Sample environment file
└── .gitignore                  # Git exclusions for secrets & temp files
```

---

## Quickstart

### 1. Prerequisites
- Python 3.11+
- Virtual environment (`venv`)
- (Optional for live runs) Free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### 2. Installation
Clone or navigate to the project directory, then activate the virtual environment and install dependencies:

```powershell
# Windows
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to `.env`:
```powershell
copy .env.example .env
```
Open `.env` and set your API key:
```ini
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
```

*(Note: Automated tests run completely offline and do not require an API key!)*

---

## Usage

### 1. Run a Live Agent Goal
```powershell
.\.venv\Scripts\python main.py "Search the latest Python release, summarize it, and save it to notes.md"
```

### 2. Verbose Mode (Live Step Tracing)
Watch the agent think, pick tools, and observe intermediate outputs in real time:
```powershell
.\.venv\Scripts\python main.py "Calculate sqrt(256) * 12 + 45 and write to calc.txt" --verbose
```

### 3. Human-in-the-Loop Approval (Gating Risky Actions)
Require interactive terminal confirmation before executing file writes:
```powershell
.\.venv\Scripts\python main.py "Write a report to summary.txt" --approve-risky
```

### 4. Restrict Available Tools (Allowlist)
```powershell
.\.venv\Scripts\python main.py "Calculate 15 * 8" --tools "calculator"
```

### 5. Check Models Available to Your Key
```powershell
.\.venv\Scripts\python scripts/list_models.py
```

---

## Testing

Run the offline test suite (completely offline with zero API calls):

```powershell
.\.venv\Scripts\pytest -q
```

All 36 offline tests cover:
- Multi-step reasoning loops
- Pydantic validation error self-correction
- Loop guard intervention on repeated identical calls
- Maximum iteration safety limits
- Prompt return on per-tool execution timeouts
- File sandboxing & directory traversal prevention (parent directory, siblings, symlinks)
- DoS limits on calculator expressions, operands, and exponents
- Merging multiple tool responses into a single user Content in SDK conversion
- Preserving raw model Content objects
- Retrying 429, 500, and 503 errors while failing immediately on 400 client errors

To run the optional live API test (requires valid `GEMINI_API_KEY` in `.env`):
```powershell
.\.venv\Scripts\pytest -m live
```

---

## Documentation

For technical explanations and interview preparation, consult the `docs/` folder:
- **[docs/EXPLAINER.md](docs/EXPLAINER.md)**: Conceptual walkthrough, Mermaid sequence diagram, code explanation, traced message exchanges, error controls, and design trade-offs.
- **[docs/INTERVIEW_PREP.md](docs/INTERVIEW_PREP.md)**: 3 resume bullet points, a 2-minute elevator pitch, and 20 interview questions with practical answers.
