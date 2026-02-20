"""
╔══════════════════════════════════════════════════════════════╗
║  CIVICHACKS 2026 — LIVE DEMO STEP 5                          ║
║  "Bring Your Own Data — Web App Edition"                     ║
║                                                              ║
║  Proves: Your BYOD AI tool works as a shareable web app      ║
║  Time on stage: ~2 minutes (just run it, browser opens)      ║
╚══════════════════════════════════════════════════════════════╝

Run this after Step 4 to show the same BYOD capability
wrapped in a polished Gradio web interface — upload files
via drag-and-drop and chat with your data in the browser.

PREREQUISITES:
  $ ollama pull llama3.1
  $ pip install llama-index llama-index-llms-ollama llama-index-embeddings-huggingface llama-index-readers-file gradio
"""

import argparse
import os
import platform
import time
from datetime import datetime
from pathlib import Path

# Suppress harmless warnings and noisy progress bars (keeps demo output clean)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

import logging
import warnings
logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*unauthenticated.*HF Hub.*")

import gradio as gr
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from cost_estimator import format_cost_short

# ── Constants ─────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".csv", ".docx"}

FILE_TYPE_NAMES = {
    ".txt": "Plain text",
    ".pdf": "PDF document",
    ".csv": "CSV spreadsheet",
    ".docx": "Word document",
}

USERDATA_DIR = Path(__file__).parent.parent / "userdata"

SUMMARY_PROMPT = (
    "You are analyzing a document that was just loaded. "
    "Provide a concise summary covering: "
    "1) What this document is about (topic and scope), "
    "2) Key data points or findings (cite specific numbers if present), "
    "3) Three questions someone might want to ask about this data. "
    "Keep it under 200 words."
)

MULTI_SUMMARY_PROMPT = (
    "You are analyzing a collection of documents that were just loaded together. "
    "Provide a concise summary covering: "
    "1) What these documents are about — identify common themes or the overall scope, "
    "2) Key data points or findings across the documents (cite specific numbers if present), "
    "3) Three questions someone might want to ask that span multiple documents. "
    "Keep it under 250 words."
)


# ── Utility functions ─────────────────────────────────────────────
def find_userdata_files():
    """Scan the userdata/ directory for supported files."""
    if not USERDATA_DIR.is_dir():
        return []
    files = [
        f for f in sorted(USERDATA_DIR.iterdir())
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return files


def format_file_size(size_bytes):
    """Convert bytes to a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def validate_uploaded_file(filepath):
    """Validate a file path for Gradio context. Returns (ok, message)."""
    filepath = Path(filepath)
    if not filepath.exists():
        return False, f"File not found: {filepath.name}"
    if not filepath.is_file():
        return False, f"'{filepath.name}' is a directory, not a file."
    ext = filepath.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        return False, f"Unsupported file type: {ext}. Supported: {supported}"
    if filepath.stat().st_size == 0:
        return False, f"File is empty: {filepath.name}"
    return True, ""


def analyze_file_metadata(filepath):
    """Return a markdown string describing file metadata."""
    filepath = Path(filepath)
    ext = filepath.suffix.lower()
    file_type = FILE_TYPE_NAMES.get(ext, ext)
    size = format_file_size(filepath.stat().st_size)
    modified = datetime.fromtimestamp(filepath.stat().st_mtime).strftime("%B %d, %Y")
    return (
        f"**{filepath.name}**\n"
        f"- Type: {file_type}\n"
        f"- Size: {size}\n"
        f"- Modified: {modified}\n"
    )


def load_and_index_files(filepaths, progress=gr.Progress()):
    """
    Load files, build vector index, generate AI summary.

    Returns:
        (index, analysis_md, summary_text) or (None, analysis_md, error_text)
    """
    progress(0.1, desc="Loading files...")

    all_documents = []
    analysis_lines = []

    for i, fp in enumerate(filepaths):
        fp = Path(fp)
        frac = 0.1 + 0.3 * (i / len(filepaths))
        progress(frac, desc=f"Loading {fp.name}...")

        meta = analyze_file_metadata(fp)
        try:
            docs = SimpleDirectoryReader(input_files=[str(fp)]).load_data()
            if docs and any(len(d.text.strip()) > 0 for d in docs):
                all_documents.extend(docs)
                total_words = sum(len(d.text.split()) for d in docs)
                meta += f"- Content: {len(docs)} chunk(s), ~{total_words:,} words\n"
            else:
                meta += "- **Warning:** No text extracted\n"
        except Exception as e:
            meta += f"- **Error:** Could not read: {e}\n"

        analysis_lines.append(meta)

    analysis_md = "\n".join(analysis_lines)

    if not all_documents:
        return None, analysis_md, "Could not extract text from any file."

    # Build index
    progress(0.5, desc="Building vector index...")
    index = VectorStoreIndex.from_documents(all_documents)

    # Generate summary
    progress(0.7, desc="Generating AI summary...")
    prompt = MULTI_SUMMARY_PROMPT if len(filepaths) > 1 else SUMMARY_PROMPT
    query_engine = index.as_query_engine(similarity_top_k=3)

    start = time.time()
    response = query_engine.query(prompt)
    elapsed = time.time() - start

    summary_text = str(response)
    est_output_tokens = int(len(summary_text.split()) * 1.3)
    est_input_tokens = int(len(prompt.split()) * 1.3) + 200
    cost_info = format_cost_short(elapsed, est_input_tokens, est_output_tokens)

    summary_with_meta = (
        f"{summary_text}\n\n---\n"
        f"*Summary generated in {elapsed:.1f}s on {HOSTNAME} — {cost_info}*"
    )

    progress(1.0, desc="Ready!")
    return index, analysis_md, summary_with_meta


# ── Gradio event handlers ─────────────────────────────────────────
def build_header_html(loaded_files=None):
    """Build the header HTML with loaded file info."""
    if loaded_files:
        n = len(loaded_files)
        file_label = f"{n} file{'s' if n > 1 else ''} loaded: {', '.join(loaded_files)}"
    else:
        file_label = "No files loaded yet — upload above or select from userdata/"

    return f"""
    <div class="header">
        <h1>🏛️ CivicHacks BYOD AI Assistant</h1>
        <h3>Bring Your Own Data</h3>
        <p>{file_label}<br>
        Powered by <strong>open source AI</strong> running locally on <strong>{HOSTNAME}</strong><br>
        <em>Started: {STARTED_AT}</em> — no cloud, no cost, no data leaving this machine.</p>
    </div>
    """


def refresh_userdata_list():
    """Rescan userdata/ and update the CheckboxGroup choices."""
    found = find_userdata_files()
    choices = [f.name for f in found]
    return gr.update(choices=choices, value=[])


def on_upload_files(files, index_state, loaded_files_state, progress=gr.Progress()):
    """Handle file upload — validate, index, and generate summary."""
    if not files:
        return (
            "*Upload files above or select from userdata/ to get started.*",
            [],
            None,
            [],
            build_header_html(),
        )

    filepaths = []
    errors = []
    for f in files:
        fp = Path(f)
        ok, msg = validate_uploaded_file(fp)
        if ok:
            filepaths.append(fp)
        else:
            errors.append(msg)

    if not filepaths:
        error_md = "\n".join(f"- {e}" for e in errors)
        return f"**Errors:**\n{error_md}", [], None, [], build_header_html()

    index, analysis_md, summary = load_and_index_files(filepaths, progress)
    file_names = [fp.name for fp in filepaths]

    if errors:
        analysis_md = "**Warnings:**\n" + "\n".join(f"- {e}" for e in errors) + "\n\n" + analysis_md

    # Put summary as first assistant message in chatbot
    history = [{"role": "assistant", "content": f"📋 **AI Summary**\n\n{summary}"}]
    header = build_header_html(file_names)

    return analysis_md, history, index, file_names, header


def on_load_userdata(selected_files, index_state, loaded_files_state, progress=gr.Progress()):
    """Load selected files from the userdata/ directory."""
    if not selected_files:
        return (
            "No files selected. Check the boxes next to the files you want to load.",
            [],
            None,
            [],
            build_header_html(),
        )

    filepaths = [USERDATA_DIR / name for name in selected_files]

    valid = []
    errors = []
    for fp in filepaths:
        ok, msg = validate_uploaded_file(fp)
        if ok:
            valid.append(fp)
        else:
            errors.append(msg)

    if not valid:
        error_md = "\n".join(f"- {e}" for e in errors)
        return f"**Errors:**\n{error_md}", [], None, [], build_header_html()

    index, analysis_md, summary = load_and_index_files(valid, progress)
    file_names = [fp.name for fp in valid]

    if errors:
        analysis_md = "**Warnings:**\n" + "\n".join(f"- {e}" for e in errors) + "\n\n" + analysis_md

    history = [{"role": "assistant", "content": f"📋 **AI Summary**\n\n{summary}"}]
    header = build_header_html(file_names)

    return analysis_md, history, index, file_names, header


def on_load_all_userdata(index_state, loaded_files_state, progress=gr.Progress()):
    """Load every supported file from userdata/."""
    found = find_userdata_files()
    if not found:
        return (
            "No supported files found in userdata/. Drop `.txt`, `.pdf`, `.csv`, or `.docx` files there first.",
            [],
            None,
            [],
            build_header_html(),
        )
    names = [f.name for f in found]
    return on_load_userdata(names, index_state, loaded_files_state, progress)


def query_byod_data(question, history, index_state, loaded_files_state):
    """Query the BYOD index and append response to chat history."""
    if not question.strip():
        return history, ""

    if index_state is None:
        history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "Please load a file first — upload above or select from the userdata/ tab."},
        ]
        return history, ""

    history = history + [{"role": "user", "content": question}]

    query_engine = index_state.as_query_engine(similarity_top_k=3)

    start = time.time()
    response = query_engine.query(question)
    elapsed = time.time() - start

    answer = str(response)

    est_output_tokens = int(len(answer.split()) * 1.3)
    est_input_tokens = int(len(question.split()) * 1.3) + 200
    cost_info = format_cost_short(elapsed, est_input_tokens, est_output_tokens)

    answer += f"\n\n---\n*⏱️ {elapsed:.1f}s · 🤖 {MODEL_NAME} on {HOSTNAME} · 💰 {cost_info}*"

    history = history + [{"role": "assistant", "content": answer}]
    return history, ""


# ── Parse arguments ───────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="CivicHacks 2026 — Step 5: BYOD Web Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
What this script does:
  Launches a Gradio web application for Bring Your Own Data (BYOD).
  Upload files or select from userdata/, get an AI summary, and ask
  questions in an interactive chat — all running locally via Ollama.

Supported file types:
  .txt    Plain text files
  .pdf    PDF documents (via llama-index-readers-file)
  .csv    Comma-separated values
  .docx   Microsoft Word documents

Prerequisites:
  1. Install Ollama        https://ollama.com
  2. Pull the model        ollama pull llama3.1
  3. Install dependencies  pip install -r requirements.txt

Examples:
  python scripts/demo_step5_byod_app.py                    # Launch on port 7861
  python scripts/demo_step5_byod_app.py --port 8080        # Custom port
  python scripts/demo_step5_byod_app.py --model phi3:mini  # Different model
  python scripts/demo_step5_byod_app.py --share            # Public URL
        """,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7861,
        help="Port to run the web server on (default: 7861)",
    )
    parser.add_argument(
        "--model",
        default="llama3.1",
        help="Ollama model to use (default: llama3.1)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public URL via Gradio's tunneling service",
    )
    return parser.parse_args()


args = parse_args()
MODEL_NAME = args.model

# ── Machine identity (shown in UI so audience knows it's live & local) ──
HOSTNAME = platform.node()
STARTED_AT = datetime.now().strftime("%B %d, %Y at %I:%M:%S %p")

# ── Initialize LLM ────────────────────────────────────────────────
print("⚙️  Starting CivicHacks BYOD AI Assistant...")
print(f"   Host: {HOSTNAME}")
print(f"   Time: {STARTED_AT}")
print(f"   Connecting to Ollama ({MODEL_NAME})...")
Settings.llm = Ollama(model=MODEL_NAME, request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
print("   ✅ Ready!\n")

# ── Build the Gradio UI ──────────────────────────────────────────

THEME = gr.themes.Soft(primary_hue="red", secondary_hue="slate")
CSS = """
    .header { text-align: center; margin-bottom: 1rem; }
    .header h1 { color: #CC0000; margin-bottom: 0.25rem; }
    .header h3 { color: #555; margin-top: 0; margin-bottom: 0.5rem; font-weight: normal; }
    .footer { text-align: center; font-size: 0.85rem; color: #888; margin-top: 1rem; }
    .file-info { background: #f8f9fa; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; }
"""

with gr.Blocks(title="CivicHacks BYOD AI Assistant") as app:

    # -- State --
    index_state = gr.State(value=None)
    loaded_files_state = gr.State(value=[])

    # -- Header --
    header = gr.HTML(build_header_html())

    # -- File Loading Section --
    with gr.Accordion("📂 Load Your Data", open=True):
        with gr.Tabs():
            with gr.Tab("Upload Files"):
                file_upload = gr.File(
                    label="Drop files here (.txt, .pdf, .csv, .docx)",
                    file_types=[".txt", ".pdf", ".csv", ".docx"],
                    file_count="multiple",
                    type="filepath",
                )
                upload_btn = gr.Button("Load & Analyze", variant="primary")

            with gr.Tab("From userdata/ Directory"):
                userdata_choices = [f.name for f in find_userdata_files()]
                userdata_files = gr.CheckboxGroup(
                    choices=userdata_choices,
                    label="Select files from userdata/",
                    info=f"{len(userdata_choices)} file(s) found" if userdata_choices else "No files found — drop files into the userdata/ directory",
                )
                with gr.Row():
                    load_selected_btn = gr.Button("Load Selected", variant="primary")
                    load_all_btn = gr.Button("Load ALL", variant="secondary")
                    refresh_btn = gr.Button("↻ Refresh", variant="secondary", size="sm")

    # -- File Analysis Display --
    file_info = gr.Markdown(
        value="*Upload files above or select from userdata/ to get started.*",
        elem_classes=["file-info"],
    )

    # -- Chat Interface --
    chatbot = gr.Chatbot(
        label="BYOD AI Chat",
        height=420,
        avatar_images=(
            None,
            "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Red_Hat_logo.svg/120px-Red_Hat_logo.svg.png",
        ),
    )

    with gr.Row():
        question_input = gr.Textbox(
            label="Your Question",
            placeholder="Ask anything about your loaded data...",
            scale=4,
            lines=1,
        )
        submit_btn = gr.Button("Ask", variant="primary", scale=1)

    # -- Footer --
    gr.HTML(f"""
    <div class="footer">
        <strong>Stack:</strong> Ollama + LlamaIndex + Gradio ·
        <strong>Model:</strong> {MODEL_NAME} ·
        <strong>Host:</strong> {HOSTNAME} ·
        <strong>Data Privacy:</strong> 100% local ·
        <strong>Cost:</strong> per-query estimate shown in each response<br>
        Built for <strong>CivicHacks 2026</strong> at Boston University ·
        Templates at <a href="https://aitemplates.io" target="_blank">aitemplates.io</a>
    </div>
    """)

    # ── Wire up events ─────────────────────────────────────────────
    load_outputs = [file_info, chatbot, index_state, loaded_files_state, header]

    upload_btn.click(
        fn=on_upload_files,
        inputs=[file_upload, index_state, loaded_files_state],
        outputs=load_outputs,
    )

    load_selected_btn.click(
        fn=on_load_userdata,
        inputs=[userdata_files, index_state, loaded_files_state],
        outputs=load_outputs,
    )

    load_all_btn.click(
        fn=on_load_all_userdata,
        inputs=[index_state, loaded_files_state],
        outputs=load_outputs,
    )

    refresh_btn.click(
        fn=refresh_userdata_list,
        inputs=[],
        outputs=[userdata_files],
    )

    submit_btn.click(
        fn=query_byod_data,
        inputs=[question_input, chatbot, index_state, loaded_files_state],
        outputs=[chatbot, question_input],
    )
    question_input.submit(
        fn=query_byod_data,
        inputs=[question_input, chatbot, index_state, loaded_files_state],
        outputs=[chatbot, question_input],
    )

# ── Launch ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        theme=THEME,
        css=CSS,
    )
