"""
╔══════════════════════════════════════════════════════════════╗
║  CIVICHACKS 2026 — LIVE DEMO STEP 3                        ║
║  "From Script to Web App in 5 Lines of UI Code"            ║
║                                                              ║
║  Proves: Wrapping your AI in a shareable web interface      ║
║  takes minutes, not days                                    ║
║  Time on stage: ~60 seconds (just run it, browser opens)    ║
╚══════════════════════════════════════════════════════════════╝

Run this during the Templates & Resources segment (0:28-0:38).
The audience watches a terminal script become a real web app.

PREREQUISITES:
  $ ollama pull llama3.1
  $ pip install llama-index llama-index-llms-ollama llama-index-embeddings-huggingface gradio
"""

import argparse
import os
import gradio as gr
import platform
import time
from datetime import datetime
from pathlib import Path

# Suppress harmless "embeddings.position_ids UNEXPECTED" warning and noisy
# progress bars from HuggingFace model loader (keeps demo output clean)
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

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from cost_estimator import format_cost_short

# ── Data files for each track ──────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"

TRACKS = {
    "🌿 EcoHack — Boston Environment": "ecohack_boston_environment.txt",
    "🏙️ CityHack — Boston 311 Services": "cityhack_boston_311.txt",
    "📚 EduHack — Boston Public Schools": "eduhack_boston_schools.txt",
    "⚖️ JusticeHack — MA Criminal Justice": "justicehack_ma_justice.txt",
}

EXAMPLE_QUESTIONS = {
    "🌿 EcoHack — Boston Environment": [
        "Which neighborhoods face the worst environmental injustice?",
        "What are the biggest climate threats to Boston?",
        "How does tree canopy coverage affect neighborhood temperatures?",
    ],
    "🏙️ CityHack — Boston 311 Services": [
        "Which neighborhoods wait longest for city services?",
        "What equity gaps exist in 311 service delivery?",
        "How are non-English speakers underserved?",
    ],
    "📚 EduHack — Boston Public Schools": [
        "What are the biggest achievement gaps by race?",
        "How does transportation affect student attendance?",
        "What barriers exist for English Language Learners?",
    ],
    "⚖️ JusticeHack — MA Criminal Justice": [
        "What racial disparities exist in pretrial detention?",
        "How effective are reentry programs?",
        "What patterns appear in Boston police stop data?",
    ],
}

# ── Global state ───────────────────────────────────────────────────
indices = {}  # Cache built indices

def build_index(track_name):
    """Build or retrieve cached vector index for a track."""
    if track_name in indices:
        return indices[track_name]

    filename = TRACKS[track_name]
    filepath = DATA_DIR / filename
    documents = SimpleDirectoryReader(input_files=[str(filepath)]).load_data()
    index = VectorStoreIndex.from_documents(documents)
    indices[track_name] = index
    return index

def query_civic_data(question, track_name, history):
    """Query the civic dataset and stream the response."""
    if not question.strip():
        return history, ""

    # Add user message to history
    history = history + [{"role": "user", "content": question}]

    # Build/get index
    index = build_index(track_name)
    query_engine = index.as_query_engine(similarity_top_k=3)

    # Query
    start = time.time()
    response = query_engine.query(question)
    elapsed = time.time() - start

    answer = str(response)

    # Estimate tokens for cost comparison
    est_output_tokens = int(len(answer.split()) * 1.3)
    est_input_tokens = int(len(question.split()) * 1.3) + 200
    cost_info = format_cost_short(elapsed, est_input_tokens, est_output_tokens)

    answer += f"\n\n---\n*⏱️ {elapsed:.1f}s · 🤖 llama3.1 on {HOSTNAME} · 💰 {cost_info}*"

    history = history + [{"role": "assistant", "content": answer}]
    return history, ""

def update_examples(track_name):
    """Update example questions when track changes."""
    examples = EXAMPLE_QUESTIONS.get(track_name, [])
    return gr.update(samples=[[q] for q in examples])

# ── Parse arguments ───────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="CivicHacks 2026 — Step 3: Civic AI Web Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
What this script does:
  Launches a Gradio web application that lets users select a hackathon
  track and ask questions about Boston & Massachusetts civic data. The
  app uses RAG (Retrieval Augmented Generation) with a local Llama 3.1
  model via Ollama — no cloud, no API keys, no cost.

Features:
  - Track selector dropdown (EcoHack, CityHack, EduHack, JusticeHack)
  - Chat interface with message history
  - Pre-built example questions per track
  - Live hostname and timestamp in the UI

Prerequisites:
  1. Install Ollama        https://ollama.com
  2. Pull the model        ollama pull llama3.1
  3. Install dependencies  pip install -r requirements.txt

Examples:
  python scripts/demo_step3_app.py              # Launch on port 7860
  python scripts/demo_step3_app.py --port 8080  # Launch on custom port
  python scripts/demo_step3_app.py --share      # Get a public URL
        """,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the web server on (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public URL via Gradio's tunneling service",
    )
    return parser.parse_args()

args = parse_args()

# ── Machine identity (shown in UI so audience knows it's live & local) ──
HOSTNAME = platform.node()
STARTED_AT = datetime.now().strftime("%B %d, %Y at %I:%M:%S %p")

# ── Initialize LLM ────────────────────────────────────────────────
print("⚙️  Starting CivicHacks AI Assistant...")
print(f"   Host: {HOSTNAME}")
print(f"   Time: {STARTED_AT}")
print("   Connecting to Ollama (llama3.1)...")
Settings.llm = Ollama(model="llama3.1", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
print("   ✅ Ready!\n")

# ── Build the Gradio UI ───────────────────────────────────────────
# THIS IS THE "5 LINES OF UI" MOMENT — Gradio makes it trivial

THEME = gr.themes.Soft(primary_hue="red", secondary_hue="slate")
CSS = """
    .header { text-align: center; margin-bottom: 1rem; }
    .header h1 { color: #CC0000; margin-bottom: 0.25rem; }
    .footer { text-align: center; font-size: 0.85rem; color: #888; margin-top: 1rem; }
"""

with gr.Blocks(title="CivicHacks AI Assistant") as app:

    gr.HTML(f"""
    <div class="header">
        <h1>🏛️ CivicHacks AI Assistant</h1>
        <p>Ask questions about real Boston &amp; Massachusetts civic data.<br>
        Powered by <strong>open source AI</strong> running locally on <strong>{HOSTNAME}</strong><br>
        <em>Started: {STARTED_AT}</em> — no cloud, no cost, no data leaving this machine.</p>
    </div>
    """)

    with gr.Row():
        track_selector = gr.Dropdown(
            choices=list(TRACKS.keys()),
            value=list(TRACKS.keys())[1],  # Default to CityHack
            label="Select Your Track",
            interactive=True,
        )

    chatbot = gr.Chatbot(
        label="Civic AI Chat",
        height=420,
        avatar_images=(None, "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Red_Hat_logo.svg/120px-Red_Hat_logo.svg.png"),
    )

    with gr.Row():
        question_input = gr.Textbox(
            label="Your Question",
            placeholder="Ask anything about the civic data...",
            scale=4,
            lines=1,
        )
        submit_btn = gr.Button("Ask", variant="primary", scale=1)

    examples = gr.Examples(
        examples=[[q] for q in EXAMPLE_QUESTIONS[list(TRACKS.keys())[1]]],
        inputs=[question_input],
        label="Try these questions:",
    )

    gr.HTML(f"""
    <div class="footer">
        <strong>Stack:</strong> Ollama + LlamaIndex + Gradio ·
        <strong>Model:</strong> Llama 3.1 8B ·
        <strong>Host:</strong> {HOSTNAME} ·
        <strong>Data Privacy:</strong> 100% local ·
        <strong>Cost:</strong> per-query estimate shown in each response<br>
        Built for <strong>CivicHacks 2026</strong> at Boston University ·
        Templates at <a href="https://aitemplates.io" target="_blank">aitemplates.io</a>
    </div>
    """)

    # ── Wire up events ─────────────────────────────────────────────
    submit_btn.click(
        fn=query_civic_data,
        inputs=[question_input, track_selector, chatbot],
        outputs=[chatbot, question_input],
    )
    question_input.submit(
        fn=query_civic_data,
        inputs=[question_input, track_selector, chatbot],
        outputs=[chatbot, question_input],
    )

    # Note: Dynamic example updating requires more complex Gradio patterns
    # For the demo, the default examples work great

# ── Launch ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        theme=THEME,
        css=CSS,
    )
