# CivicHacks 2026 Demo — Comprehensive User Guide

**Workshop:** *Why Open Source AI Changes Everything — And How to Use It This Weekend*
**Event:** CivicHacks 2026 · Boston University · February 21-22, 2026
**Presenters:** Jan Mark Holzer & Lucas Yoon, Red Hat

---

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Prerequisites & Installation](#prerequisites--installation)
4. [Project Structure](#project-structure)
5. [Step 1 — Local AI with Ollama](#step-1--local-ai-with-ollama)
6. [Step 2 — RAG with Civic Data](#step-2--rag-with-civic-data)
7. [Step 3 — Gradio Web Application](#step-3--gradio-web-application)
8. [Step 4 — Bring Your Own Data](#step-4--bring-your-own-data)
9. [Step 5 — BYOD Web Application](#step-5--byod-web-application)
10. [Civic Datasets Reference](#civic-datasets-reference)
11. [Live Demo Presenter Guide](#live-demo-presenter-guide)
12. [Customization & Adaptation](#customization--adaptation)
13. [Deployment Options](#deployment-options)
14. [Troubleshooting](#troubleshooting)
15. [Resources & Further Reading](#resources--further-reading)
16. [License](#license)

---

## Introduction

This project is a live-demo toolkit that builds a **complete civic AI application** on stage in three progressive steps. It proves that open source AI is free, powerful, and accessible to anyone with a laptop. The demo is designed for the opening workshop of CivicHacks 2026 and can be adapted for any hackathon, classroom, or conference setting.

**What the audience sees:**

| Step | Duration | Takeaway |
|------|----------|----------|
| Step 1 | ~60 sec | "AI runs on a laptop for free" |
| Step 2 | ~90 sec | "It can analyze our city's data" |
| Step 3 | ~60 sec | "That's a real product — built in minutes" |
| Step 4 | ~3-5 min | "Now plug in YOUR data and start asking questions" |
| Step 5 | ~2 min | "Now your BYOD tool is a shareable web app too" |

The demo uses **synthetic but realistic Boston and Massachusetts civic datasets** covering four hackathon tracks: EcoHack (environment), CityHack (311 services), EduHack (public schools), and JusticeHack (criminal justice). The audience votes on which track to demo live, creating engagement and ownership.

---

## Architecture Overview

The application stack is composed of four open source layers, all running locally with zero cost:

```
┌─────────────────────────────────────────────┐
│              Gradio Web UI (Step 3)         │  ← Browser-based chat interface
├─────────────────────────────────────────────┤
│          LlamaIndex RAG Pipeline (Step 2)   │  ← Retrieval Augmented Generation
│   ┌──────────────┐    ┌──────────────────┐  │
│   │ Vector Index  │    │ HuggingFace      │  │
│   │ (in-memory)   │    │ Embeddings       │  │
│   │               │    │ (all-MiniLM-L6)  │  │
│   └──────────────┘    └──────────────────┘  │
├─────────────────────────────────────────────┤
│            Ollama + Llama 3.1 (Step 1)      │  ← Local LLM inference
├─────────────────────────────────────────────┤
│            Civic Data Files (data/)         │  ← .txt datasets per track
└─────────────────────────────────────────────┘
```

**Data flow:**

1. User asks a question (via terminal or Gradio UI)
2. LlamaIndex uses the HuggingFace embedding model to convert the question into a vector
3. The vector index retrieves the most relevant chunks from the civic dataset
4. The retrieved context + question are sent to Llama 3.1 via Ollama
5. The LLM generates a grounded answer citing real data
6. The response streams back to the user

---

## Prerequisites & Installation

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| GPU VRAM | Not required (CPU works) | 8+ GB (much faster) |
| Storage | 10 GB free | 20 GB free |
| CPU | Any modern 4-core | Apple Silicon or recent Intel/AMD |
| Python | 3.10+ | 3.12+ |

**Apple Silicon Macs** (M1/M2/M3/M4) are ideal — unified memory handles Llama 3.1 8B at 15-25 tokens/second. CPU-only machines still work at ~3-5 tokens/second.

### Step-by-Step Installation

#### 1. Install Ollama

Ollama is the local LLM runtime that hosts Llama 3.1 on your machine.

**macOS (Homebrew):**
```bash
brew install ollama
```

**macOS (direct download):**
Download from https://ollama.com

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download the installer from https://ollama.com

#### 2. Pull the Llama 3.1 Model

```bash
# Downloads ~4.7 GB — use reliable wifi
ollama pull llama3.1
```

Verify it works:
```bash
ollama run llama3.1 "Say hello in 10 words or less"
```

If Ollama isn't running as a background service, start it first:
```bash
ollama serve
```

#### 3. Clone the Repository & Set Up Python

```bash
git clone https://github.com/YOUR_USERNAME/civichacks-demo.git
cd civichacks-demo

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Understand the Dependencies

The `requirements.txt` installs five packages:

| Package | Purpose |
|---------|---------|
| `llama-index>=0.12.0` | Core RAG framework for connecting AI to data |
| `llama-index-llms-ollama>=0.5.0` | Ollama integration for local LLM inference |
| `llama-index-embeddings-huggingface>=0.5.0` | Local embedding model (no API key needed) |
| `llama-index-readers-file>=0.4.0` | File readers for PDF, DOCX, and other formats |
| `gradio>=5.0.0` | Web UI framework for building the chat interface |

#### 5. Pre-warm Everything (Critical for Live Demos)

First runs are slower because models need to load into memory and the embedding model (~80 MB) downloads on first use. Run each step once before presenting:

```bash
# Pre-warm Step 1
python scripts/demo_step1_ollama.py

# Pre-warm Step 2 (each track)
python scripts/demo_step2_rag.py eco
python scripts/demo_step2_rag.py city
python scripts/demo_step2_rag.py edu
python scripts/demo_step2_rag.py justice

# Pre-warm Step 3 (start it, verify it loads, then Ctrl+C)
python scripts/demo_step3_app.py
```

---

## Project Structure

```
civichacks-demo/
├── README.md                             # Project overview and demo flow
├── USER_GUIDE.md                         # This comprehensive guide
├── CivicHacks_Demo_Guide.pdf            # Printable presenter reference
├── requirements.txt                      # Python dependencies (4 packages)
├── data/                                 # Civic datasets (one per track)
│   ├── ecohack_boston_environment.txt     # Boston environmental quality data
│   ├── cityhack_boston_311.txt            # Boston 311 service request data
│   ├── eduhack_boston_schools.txt         # Boston public schools equity data
│   └── justicehack_ma_justice.txt        # MA criminal justice reform data
├── userdata/                             # Drop your own files here for Step 4
└── scripts/                              # Demo scripts (run in order)
    ├── cost_estimator.py                 # Shared: local vs. cloud cost comparison
    ├── demo_step1_ollama.py              # Step 1: Basic local AI inference
    ├── demo_step2_rag.py                 # Step 2: RAG with civic data
    ├── demo_step3_app.py                 # Step 3: Full Gradio web app
    ├── demo_step4_byod.py               # Step 4: Bring Your Own Data (interactive)
    └── demo_step5_byod_app.py           # Step 5: BYOD Web Application (Gradio)
```

---

## Step 1 — Local AI with Ollama

**File:** `scripts/demo_step1_ollama.py`
**Purpose:** Prove that a GPT-4-class model runs locally, for free, with no API key
**Duration:** ~60 seconds on stage

### What It Does

The script sends a civic-themed prompt to the local Ollama instance and **streams** the response token by token so the audience watches the AI generate in real time. When done, it prints the elapsed time, tokens per second, and a **live cost comparison** — the actual electricity cost versus what the same query would cost on cloud APIs like GPT-4o.

### The Prompt

The hardcoded prompt asks the model to act as a civic technology advisor and provide three bullet points on why open source AI matters for hackathon participants building community tools.

### How to Run

```bash
python scripts/demo_step1_ollama.py          # Run the demo
python scripts/demo_step1_ollama.py --help   # Show usage info
```

### Expected Output

```
🏛️  CivicHacks 2026 — Open Source AI, Running Locally

📡 Model: llama3.1 (8B) — running on YOUR-HOSTNAME
🕐 Time: February 21, 2026 at 10:15:23 AM
🔒 Data: never leaves YOUR-HOSTNAME

────────────────────────────────────────────────────────────

💬 Prompt: You are a civic technology advisor...

────────────────────────────────────────────────────────────

🤖 Response:

[Streamed AI response appears here token by token]

────────────────────────────────────────────────────────────
⏱️  12.3s · 142 tokens · 11 tok/s
⚡ Local: $0.000009 (0.051 Wh @ 15W) · GPT-4o: $0.0017 (189x more)
────────────────────────────────────────────────────────────

✅ That's it. Local AI. Private. And virtually free.
```

### How It Works Internally

1. Imports the `ollama` Python client library and the shared `cost_estimator` module
2. Displays the live hostname (`platform.node()`) and current timestamp
3. Calls `ollama.chat()` with `stream=True` targeting the `llama3.1` model
4. Iterates over chunks, printing each token fragment immediately (`flush=True`)
5. Extracts token counts from Ollama's final streaming chunk
6. Calls `format_cost_comparison()` to show local electricity cost vs. cloud API pricing

### Key Code (simplified)

```python
from cost_estimator import format_cost_comparison

stream = ollama.chat(
    model="llama3.1",
    messages=[{"role": "user", "content": PROMPT}],
    stream=True,
)
for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)

# After streaming completes:
cost_line = format_cost_comparison(elapsed, input_tokens, output_tokens)
print(cost_line)
```

### Cost Estimator Module

All three scripts share `scripts/cost_estimator.py`, which provides:

- **`detect_power_watts()`** — Auto-detects hardware wattage (Apple Silicon base/Pro/Max, x86 laptop, desktop GPU, etc.)
- **`estimate_local_cost(duration, watts)`** — Calculates actual electricity cost: `watts × seconds / 3600 = Wh`, then `Wh × $/kWh`
- **`estimate_cloud_cost(input_tokens, output_tokens)`** — Looks up published per-token pricing for GPT-4o, Claude 3.5 Sonnet, Gemini 2.5 Flash, and others
- **`format_cost_comparison()`** — Full one-line format for terminal output
- **`format_cost_short()`** — Compact format for Gradio chat metadata

---

## Step 2 — RAG with Civic Data

**File:** `scripts/demo_step2_rag.py`
**Purpose:** Connect the local AI to real civic data using Retrieval Augmented Generation
**Duration:** ~90 seconds on stage

### What It Does

1. Loads a track-specific civic dataset from the `data/` directory
2. Builds an in-memory vector index using HuggingFace embeddings
3. Queries the index with a pre-written question relevant to the track
4. The AI generates a response grounded in the actual data, citing specific statistics

### Available Tracks

| Track Key | Name | Data File | Focus |
|-----------|------|-----------|-------|
| `eco` | EcoHack | `ecohack_boston_environment.txt` | Air quality, heat islands, climate resilience |
| `city` | CityHack | `cityhack_boston_311.txt` | 311 service requests, equity gaps |
| `edu` | EduHack | `eduhack_boston_schools.txt` | Achievement gaps, absenteeism, tech access |
| `justice` | JusticeHack | `justicehack_ma_justice.txt` | Incarceration disparities, policing data |

### How to Run

```bash
# Random question from the city track (default)
python scripts/demo_step2_rag.py city

# Specific question number (1-3)
python scripts/demo_step2_rag.py city 2

# All three queries for the track
python scripts/demo_step2_rag.py eco --all

# Defaults to "city" if no argument given
python scripts/demo_step2_rag.py
```

Each query displays a cost comparison showing local electricity cost versus cloud API pricing, just like Step 1.

### Sample Questions per Track

**EcoHack:**
- Which Boston neighborhoods have the worst air quality and why?
- What are the biggest environmental justice concerns in this data?
- How is climate change specifically threatening Boston's coastline?

**CityHack:**
- Which neighborhoods have the longest 311 response times and what are the equity implications?
- What are the biggest service gaps for non-English speaking residents?
- What patterns suggest systemic inequity in city service delivery?

**EduHack:**
- What are the most significant achievement gaps in Boston public schools?
- How does transportation affect student attendance and outcomes?
- What technology access barriers exist for students and teachers?

**JusticeHack:**
- What racial disparities exist in pretrial detention in Massachusetts?
- How effective are reentry programs at reducing recidivism?
- What does the data reveal about policing patterns in Boston?

### How It Works Internally

1. **Configure LLM:** Sets `Settings.llm` to Ollama with `llama3.1` and a 120-second timeout
2. **Configure Embeddings:** Sets `Settings.embed_model` to HuggingFace's `all-MiniLM-L6-v2` (runs on CPU, ~80 MB)
3. **Load Documents:** `SimpleDirectoryReader` reads the specified `.txt` file
4. **Build Index:** `VectorStoreIndex.from_documents()` chunks the text, computes embeddings, and builds a searchable vector index in memory
5. **Query:** `index.as_query_engine(streaming=True, similarity_top_k=3)` retrieves the 3 most relevant chunks, then sends them + the question to the LLM
6. **Stream Response:** The answer streams to the terminal in real time

### Key Code (simplified)

```python
Settings.llm = Ollama(model="llama3.1", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

documents = SimpleDirectoryReader(input_files=[str(data_file)]).load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(streaming=True, similarity_top_k=3)

response = query_engine.query("Which neighborhoods have the longest 311 response times?")
response.print_response_stream()
```

---

## Step 3 — Gradio Web Application

**File:** `scripts/demo_step3_app.py`
**Purpose:** Wrap the RAG pipeline in a shareable web application
**Duration:** ~60 seconds to launch on stage

### What It Does

Launches a polished Gradio-based chat interface in the browser with:
- A **dynamic header** that updates its title and description when you switch tracks
- A track selector dropdown (switch between all four datasets)
- A chat interface with message history
- **Dynamic example questions** that change per track when the selector changes
- **Per-query cost comparison** (local electricity vs. cloud API) appended to each response
- Live hostname and timestamp showing the app is running locally
- A footer showing the full open source stack and privacy note
- Red Hat-themed styling (red primary color, soft theme)

### How to Run

```bash
python scripts/demo_step3_app.py
```

The app starts on `http://localhost:7860` and should open automatically in your default browser.

### UI Components

| Component | Description |
|-----------|-------------|
| **Header** | Dynamic — shows "CivicHacks AI Assistant", track name, track description, hostname, and start time. Updates when track changes |
| **Track Selector** | Dropdown to switch between EcoHack, CityHack, EduHack, JusticeHack — triggers header and example updates |
| **Chatbot** | Message-style chat with user/assistant bubbles and Red Hat avatar. Each response includes timing and cost comparison |
| **Question Input** | Text field + "Ask" button (also supports Enter key) |
| **Example Questions** | Clickable examples that auto-populate the input field — **update dynamically** when the track changes |
| **Footer** | Stack info, model info, hostname, privacy note, and per-query cost estimate note |

### How It Works Internally

1. **Global Index Cache:** Built indices are cached in a `dict` so switching tracks after the first load is instant
2. **`build_index(track_name)`:** Checks the cache, and if the track hasn't been loaded yet, reads the data file, builds a `VectorStoreIndex`, and caches it
3. **`query_civic_data(question, track_name, history)`:** Appends the user message to chat history, queries the cached index, appends the AI response with timing and cost comparison metadata (via `format_cost_short()`), and returns the updated history
4. **`build_header_html(track_name)`:** Generates dynamic HTML for the header section with track name, description, hostname, and start time
5. **`on_track_change(track_name)`:** Called when the track selector changes — returns updated header HTML and a new `gr.Dataset` with track-specific example questions
6. **Gradio Blocks:** The UI is built using `gr.Blocks` with a `Soft` theme and custom CSS passed to `launch()`. The `track_selector.change()` event wires to `on_track_change()` to dynamically update the header and example questions

### Key Code (simplified)

```python
THEME = gr.themes.Soft(primary_hue="red", secondary_hue="slate")
CSS = ".header { text-align: center; } .header h1 { color: #CC0000; }"

with gr.Blocks(title="CivicHacks AI Assistant") as app:
    header = gr.HTML(build_header_html(default_track))
    track_selector = gr.Dropdown(choices=list(TRACKS.keys()), ...)
    chatbot = gr.Chatbot(height=420, ...)
    question_input = gr.Textbox(placeholder="Ask anything...", ...)
    submit_btn = gr.Button("Ask", variant="primary")
    examples = gr.Examples(examples=..., inputs=[question_input])

    submit_btn.click(fn=query_civic_data, ...)

    # Dynamic track switching — updates header and example questions
    track_selector.change(
        fn=on_track_change,
        inputs=[track_selector],
        outputs=[header, examples.dataset],
    )

app.launch(server_name="0.0.0.0", server_port=7860, theme=THEME, css=CSS)
```

> **Note:** Gradio 6.x moved `theme` and `css` parameters from `gr.Blocks()` to `launch()`, and the `Chatbot` component uses messages format by default (the `type` parameter was removed).

### Command-Line Options

```bash
python scripts/demo_step3_app.py              # Launch on default port 7860
python scripts/demo_step3_app.py --port 8080  # Launch on custom port
python scripts/demo_step3_app.py --share      # Get a public URL
python scripts/demo_step3_app.py --help       # Show all options
```

| Option | Default | Description |
|--------|---------|-------------|
| `--port` | `7860` | HTTP port for the web UI |
| `--share` | off | Create a temporary public URL via Gradio's tunneling service |

---

## Step 4 — Bring Your Own Data

**File:** `scripts/demo_step4_byod.py`
**Purpose:** Let attendees plug in their own data file and interactively ask questions about it
**Duration:** ~3-5 minutes (interactive hands-on segment)

### What It Does

An interactive terminal-based script that:
1. **Auto-discovers** files in the `userdata/` directory, or accepts a file path as a CLI argument
2. Supports loading a **single file** or **all files at once** (`--all`) for cross-file exploration
3. Analyzes each file (type, size, word count, content preview)
4. Builds a vector index and generates an AI summary of the contents
5. Enters an interactive Q&A loop where the user types questions and gets AI answers grounded in their data
6. Shows cost comparison (local electricity vs. cloud API) on every query

### How to Run

```bash
# Auto-discover files in userdata/ (prompts to select if multiple found)
python scripts/demo_step4_byod.py

# Load ALL files in userdata/ into one index (cross-file exploration)
python scripts/demo_step4_byod.py --all

# With a specific file path
python scripts/demo_step4_byod.py path/to/your/file.txt

# With a PDF
python scripts/demo_step4_byod.py ~/Downloads/report.pdf

# Use a different model
python scripts/demo_step4_byod.py myfile.txt --model phi3:mini

# Show usage info
python scripts/demo_step4_byod.py --help
```

### userdata/ Auto-Discovery

Drop files into the `userdata/` directory before running the script. The auto-discovery logic works as follows:

| Files in userdata/ | Behavior |
|---------------------|----------|
| 0 files | Prompts for a file path |
| 1 file | Automatically uses that file |
| 2+ files | Shows a numbered list to pick from (or type `a` to load all) |
| `--all` flag | Loads every file into a single combined index |

The `--all` mode builds one unified vector index across all files, enabling cross-document questions like "Compare the findings across these reports" or "What themes are common across all the data?"

### Supported File Types

| Extension | Type | Notes |
|-----------|------|-------|
| `.txt` | Plain text | Simplest option, works everywhere |
| `.pdf` | PDF document | Requires `llama-index-readers-file` (already in requirements) |
| `.csv` | CSV spreadsheet | Read as text content |
| `.docx` | Word document | Requires `llama-index-readers-file` |

### Expected Output

```
════════════════════════════════════════════════════════════
  CIVICHACKS 2026 — Bring Your Own Data
════════════════════════════════════════════════════════════

  Configuring local AI stack...
   Host: jholzer-mac
   Time: February 21, 2026 at 02:15:30 PM
   Model: llama3.1 (via Ollama — running on jholzer-mac)
   Embeddings: all-MiniLM-L6-v2 (runs on CPU)

────────────────────────────────────────────────────────────
  File Analysis
────────────────────────────────────────────────────────────

   File:      boston_budget_2026.pdf
   Path:      /Users/attendee/Downloads/boston_budget_2026.pdf
   Type:      PDF document
   Size:      2.4 MB
   Modified:  February 18, 2026

   Content:   3 document(s), 45,230 characters, ~8,120 words

   Preview:
   "CITY OF BOSTON FISCAL YEAR 2026 OPERATING BUDGET..."

────────────────────────────────────────────────────────────

  Building vector index (this is the 'RAG' magic)...
   Index built in 2.3s

────────────────────────────────────────────────────────────
  AI Summary of: boston_budget_2026.pdf
────────────────────────────────────────────────────────────

  [Streamed AI response — topic summary, key data points,
   and suggested questions about the data]

  8.4s · ~185 tokens
  Local: $0.000010 (0.035 Wh @ 15W) · GPT-4o: $0.0023 (230x more)

════════════════════════════════════════════════════════════
  Interactive Q&A — Ask anything about your data
  Type 'quit' to end | 'help' for commands
════════════════════════════════════════════════════════════

  [You] >> What are the biggest budget increases this year?

  [Streamed AI answer grounded in the document data...]

  [You] >> quit

════════════════════════════════════════════════════════════
  Session complete — 1 question answered
  All processing done locally on jholzer-mac.
  Zero data sent to the cloud.
════════════════════════════════════════════════════════════
```

### Interactive Commands

| Command | Action |
|---------|--------|
| *(any question)* | Query the AI about your data |
| `summary` | Re-generate the AI summary |
| `help` | Show available commands |
| `quit` / `exit` / `q` | End the session |

### How It Works Internally

1. **`find_userdata_files()`** scans the `userdata/` directory for supported file types
2. **`validate_file()`** resolves the path, checks extension and file size, handles drag-and-drop quote stripping
3. **`analyze_file()`** loads a single file via `SimpleDirectoryReader`, prints metadata and a content preview
4. **`analyze_all_files()`** loads multiple files, prints per-file status, skips unreadable files gracefully, and combines all documents
5. **`VectorStoreIndex.from_documents()`** builds the in-memory vector index (same as Steps 2 & 3)
6. **`generate_summary()`** queries the index with a summary prompt (single-file or multi-file variant) and streams the AI response
7. **`interactive_loop()`** runs the Q&A loop — each question is an independent RAG query with cost comparison
8. Ctrl+C is caught cleanly (no Python traceback during live demos)

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `file` | *(auto-discover)* | Path to data file (positional, optional) |
| `--all` | off | Load ALL files in userdata/ into a single index for cross-file exploration |
| `--model` | `llama3.1` | Ollama model to use (lets attendees try different models) |

---

## Step 5 — BYOD Web Application

**File:** `scripts/demo_step5_byod_app.py`
**Purpose:** Wrap the BYOD experience in a Gradio web UI with drag-and-drop file upload
**Duration:** ~2 minutes on stage (just run it, browser opens)

### What It Does

A Gradio web application that provides the same BYOD functionality as Step 4, but in a polished browser interface:
1. **Drag-and-drop file upload** or select files from `userdata/` directory
2. Supports loading a **single file** or **multiple files at once** for cross-file exploration
3. Displays file analysis metadata (name, type, size, word count) in the UI
4. Generates an **AI summary** shown as the first chat message
5. **Interactive Q&A chat** — ask questions and get AI answers grounded in your data
6. Shows cost comparison (local electricity vs. cloud API) on every response

### How to Run

```bash
# Launch the web app (opens at http://localhost:8861)
python scripts/demo_step5_byod_app.py

# Custom port
python scripts/demo_step5_byod_app.py --port 8080

# Use a different model
python scripts/demo_step5_byod_app.py --model phi3:mini

# Create a public URL for sharing
python scripts/demo_step5_byod_app.py --share

# Show usage info
python scripts/demo_step5_byod_app.py --help
```

### UI Components

| Component | Description |
|-----------|-------------|
| **Header** | Dynamic — shows "CivicHacks BYOD AI Assistant", loaded file names, hostname, and start time. Updates when files are loaded |
| **File Upload Tab** | Drag-and-drop area for uploading `.txt`, `.pdf`, `.csv`, `.docx` files. Supports multiple files. "Load & Analyze" button |
| **userdata/ Tab** | Checkbox list of files in `userdata/` directory. "Load Selected", "Load ALL", and "Refresh" buttons |
| **File Analysis** | Markdown display showing metadata for each loaded file (type, size, modified date, word count) |
| **Chatbot** | Message-style chat. First message is the AI summary. Subsequent messages show Q&A with cost comparison |
| **Question Input** | Text field + "Ask" button (also supports Enter key) |
| **Footer** | Stack info, model info, hostname, privacy note |

### How It Works Internally

1. **`find_userdata_files()`** scans the `userdata/` directory for supported file types
2. **`validate_uploaded_file()`** validates file path, extension, and size (returns error tuple instead of exiting)
3. **`analyze_file_metadata()`** returns markdown string with file metadata for display
4. **`load_and_index_files()`** loads documents via `SimpleDirectoryReader`, builds the vector index, generates AI summary — with `gr.Progress()` for visual feedback
5. **`query_byod_data()`** queries the index and appends the response with cost metadata to chat history
6. **`gr.State`** stores the per-session index and loaded file list, supporting multiple concurrent browser sessions

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--port` | `8861` | HTTP port for the web UI (different from Step 3's 7860) |
| `--model` | `llama3.1` | Ollama model to use |
| `--share` | off | Create a temporary public URL via Gradio's tunneling service |

---

## Civic Datasets Reference

All four datasets are **synthetic but realistic** — fabricated for demonstration purposes using real-world patterns. They live in the `data/` directory as plain `.txt` files.

### EcoHack: Boston Environment (`ecohack_boston_environment.txt`)

**Scope:** City of Boston Environmental Quality Report, Q3 2025

**Sections covered:**
- Air Quality Summary (AQI averages, PM2.5 by neighborhood, NO2 near airport)
- Urban Heat Island Effect (tree canopy vs. surface temperature mapping)
- Water Quality (harbor standards, combined sewer overflows, E. coli)
- Waste & Recycling (contamination rates, composting pilot)
- Climate Resilience (sea level rise projections, coastal projects)
- Environmental Justice Metrics (air quality days, asthma, green space, contaminated sites)

### CityHack: Boston 311 Services (`cityhack_boston_311.txt`)

**Scope:** City of Boston 311 Service Request Analysis, 2025

**Sections covered:**
- Overview (487K+ requests, resolution times, satisfaction ratings, channels)
- Top Request Categories (street cleaning, potholes, code enforcement, street lights, trees)
- Geographic Disparities (resolution times and satisfaction by neighborhood vs. income)
- Staffing & Budget (agent count, wait times, abandonment rates)
- Language Access (request volume by language, translation gaps, resolution disparities)
- Open Issues (overdue request backlog by category)

### EduHack: Boston Public Schools (`eduhack_boston_schools.txt`)

**Scope:** Boston Public Schools Educational Equity Report, 2024-2025

**Sections covered:**
- Enrollment & Demographics (49K+ students, 125 schools, 78 home languages)
- Achievement Gaps (MCAS proficiency by race, income, ELL status, disability)
- Chronic Absenteeism (rates by demographic, transportation as predictor)
- Technology Access (device ratio, home internet gaps, digital literacy)
- Staffing (vacancies by subject, demographic mismatch, turnover)
- College & Career Readiness (graduation rates, AP enrollment/pass rates)

### JusticeHack: MA Criminal Justice (`justicehack_ma_justice.txt`)

**Scope:** Massachusetts Criminal Justice Reform Progress Report, 2025

**Sections covered:**
- Incarceration Overview (population trends, racial disparity ratios)
- Pretrial Detention (daily population, bail amounts, racial breakdown, cost)
- Recidivism (3-year rates by offense, impact of education and employment)
- Policing Data (FIO stops by race, use of force, body cameras, complaints)
- Juvenile Justice (DYS commitments, community alternatives, mental health)
- Immigration Enforcement Intersection (detainers, legal representation gaps)
- Access to Legal Representation (public defender caseloads, wait times)

---

## Live Demo Presenter Guide

### Timeline

| Time | Step | Action |
|------|------|--------|
| ~Minute 3 | Step 1 | Run `demo_step1_ollama.py` — prove local AI works |
| ~Minute 15 | Audience Vote | Audience picks a track (EcoHack, CityHack, EduHack, JusticeHack) |
| ~Minute 20 | Step 2 | Run `demo_step2_rag.py <track>` — show RAG with real data |
| ~Minute 32 | Step 3 | Run `demo_step3_app.py` — reveal the web app |
| ~Minute 40 | Step 4 | Run `demo_step4_byod.py` — attendee brings their own data |
| ~Minute 45 | Step 5 | Run `demo_step5_byod_app.py` — BYOD as a web app |

### Before Going on Stage

1. Ensure Ollama is running (`ollama list` should show `llama3.1`)
2. Run all pre-warm commands (see [Prerequisites](#prerequisites--installation))
3. Set terminal font size large enough for the back row
4. Close unnecessary applications to free memory
5. Have a backup screen recording ready in case of hardware failure

### Talking Points by Step

**Step 1 — While the model generates:**
> "This is the same architecture behind GPT-4. It's generating on this laptop's CPU/GPU. No internet required. No data leaving this machine."

**Step 1 — After it finishes:**
> "Look at the cost line — fractions of a cent in electricity, versus what you'd pay on GPT-4o. That's the power of local inference."

**Step 2 — While the index builds:**
> "It's reading the city's data and building a search index. This is the same technique behind every enterprise AI chatbot — except we're doing it locally, for free."

**Step 2 — After the answer:**
> "It pulled specific numbers from the city's own data. That's RAG — the model retrieves relevant chunks of real data before generating. This is how you build civic tech that's trustworthy."

**Step 3 — Walk through the UI:**
> "Watch what happens when I switch tracks — the header, description, and example questions all update. Each answer shows you the real cost comparison. This is a production app, built in minutes."

**Step 3 — The kicker:**
> "Ollama — free. Llama 3.1 — free. LlamaIndex — free. Gradio — free. Every query costs fractions of a cent in electricity. That's what open source AI makes possible."

**Step 4 — Bring Your Own Data:**
> "You've seen what our civic data can do. But what about YOUR data? Got a PDF, a spreadsheet, a text file? Drop it in and start asking questions — no code changes, no configuration. That's the whole point of open source AI — you're not limited to what we prepared."

**Step 5 — BYOD Web App:**
> "Same BYOD capability, now as a shareable web app. Drag and drop files in the browser, get a summary, ask questions — and you can share it with your whole team."

---

## Customization & Adaptation

### Swap in Your Own Data

Replace or add files in the `data/` directory. The RAG pipeline supports:

| Format | Notes |
|--------|-------|
| `.txt` | Used in this demo — simplest option |
| `.pdf` | Add `llama-index-readers-file` to requirements |
| `.csv` | Supported out of the box |
| `.docx` | Supported out of the box |
| Web pages | Use `SimpleWebPageReader` from LlamaIndex |

After adding new files, update the `TRACKS` dictionaries in `demo_step2_rag.py` and `demo_step3_app.py` to reference them.

### Change the Model

```bash
# Smaller / faster (for limited hardware)
ollama pull phi3:mini          # 3.8B parameters
ollama pull llama3.2:3b        # 3B parameters, very fast

# Larger / better (if you have the RAM)
ollama pull llama3.1:70b       # ~40 GB RAM needed

# Strong reasoning
ollama pull deepseek-r1:7b     # MIT license
```

Then update the model name in the scripts:

```python
Settings.llm = Ollama(model="phi3:mini")  # Change in demo_step2_rag.py and demo_step3_app.py
```

For Step 1, update the `ollama.chat()` call:

```python
stream = ollama.chat(model="phi3:mini", ...)  # Change in demo_step1_ollama.py
```

### Change the Prompt (Step 1)

Edit the `PROMPT` variable in `demo_step1_ollama.py` to match your event theme.

### Change Example Questions (Step 3)

Edit the `EXAMPLE_QUESTIONS` dictionary in `demo_step3_app.py`.

### Change the Theme

The Gradio theme is set in `demo_step3_app.py`:

```python
theme=gr.themes.Soft(primary_hue="red", secondary_hue="slate")
```

Change `primary_hue` to any Gradio color (e.g., `"blue"`, `"green"`, `"purple"`).

---

## Deployment Options

### Local Only (Default)

The app runs entirely on `localhost:7860`. No data ever leaves the machine.

### Temporary Public URL (Gradio Share)

```bash
python scripts/demo_step3_app.py --share
```

Gradio provides a temporary public URL (valid ~72 hours) via tunneling. Useful for letting judges or remote participants try the app.

### Hugging Face Spaces (Free Hosting)

1. Create an account at https://huggingface.co
2. Create a new Space (select "Gradio" as SDK)
3. Push your code
4. **Note:** You'll need to swap Ollama for the HF Inference API or a hosted model endpoint, since Ollama runs locally

### Streamlit Cloud (Free Alternative)

Convert `demo_step3_app.py` to use Streamlit instead of Gradio. Streamlit Cloud offers free hosting with GitHub integration.

---

## Troubleshooting

### Ollama Issues

| Problem | Solution |
|---------|----------|
| "Ollama isn't responding" | Run `ollama serve` to start the daemon, then `ollama list` to verify |
| Model not found | Run `ollama pull llama3.1` to download it |
| Connection refused | Ollama defaults to `localhost:11434` — ensure no firewall is blocking it |

### Performance Issues

| Problem | Solution |
|---------|----------|
| Very slow generation | CPU-only: expect 3-8 tok/sec. Close other apps to free RAM |
| Index building is slow | First run downloads the embedding model (~80 MB). Subsequent runs use cache |
| App is unresponsive | Ensure at least 8 GB RAM is available. Llama 3.1 8B uses ~4-5 GB |

### Gradio Issues

| Problem | Solution |
|---------|----------|
| Browser doesn't open | Navigate manually to `http://localhost:7860` |
| Port already in use | Kill the existing process or change `server_port` in the script |
| Specific browser needed | Set env var: `BROWSER=chrome python scripts/demo_step3_app.py` |

### Embedding Model Issues

| Problem | Solution |
|---------|----------|
| Download hangs | Check internet connection. The model is ~80 MB from HuggingFace |
| "No module named..." | Ensure you installed all requirements: `pip install -r requirements.txt` |
| Cached model location | Default: `~/.cache/huggingface/hub/` |

### Backup Plan

If hardware or wifi fails during a live demo, have a pre-recorded screen capture of the full demo flow. Record it at the venue so the environment looks authentic.

---

## Resources & Further Reading

| Resource | URL | Description |
|----------|-----|-------------|
| Ollama | https://ollama.com | Run LLMs locally |
| LlamaIndex | https://docs.llamaindex.ai | RAG framework documentation |
| Gradio | https://gradio.app | ML web UI framework |
| Hugging Face | https://huggingface.co | Models, datasets, free deployment |
| AI Templates | https://aitemplates.io | Production-ready AI app templates (Red Hat) |
| Artificial Analysis | https://artificialanalysis.ai | Independent model benchmarks |
| LangChain | https://langchain.com | Alternative LLM application framework |
| CrewAI | https://crewai.com | Multi-agent framework |

### Real Civic Data Sources

| Source | URL | Description |
|--------|-----|-------------|
| City of Boston | https://data.boston.gov | Boston open data portal |
| Massachusetts | https://mass.gov/open-data | State open data |
| Federal | https://data.gov | US federal open data |

> **Full Resource Cheat Sheet:** For a comprehensive list of tools, frameworks, models, vector databases, agent frameworks, deployment platforms, and hackathon tips, see **[RESOURCES.md](RESOURCES.md)**.

---

## License

This demo code is released under the **Apache 2.0 License**. Use it, fork it, teach with it, build on it.

The civic datasets included are **synthetic/illustrative** — they are realistic but fabricated for demonstration purposes. For real civic data, use the sources listed above.

---

*Built for CivicHacks 2026 at Boston University. Go build something that matters.*
