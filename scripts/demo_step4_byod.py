"""
╔══════════════════════════════════════════════════════════════╗
║  CIVICHACKS 2026 — LIVE DEMO STEP 4                        ║
║  "Bring Your Own Data"                                       ║
║                                                              ║
║  Proves: You can plug in ANY file and start querying it     ║
║  with AI instantly — no code changes needed                 ║
║  Time on stage: ~3-5 minutes (interactive)                  ║
╚══════════════════════════════════════════════════════════════╝

Run this after the main demo (Steps 1-3) during the hands-on
segment. An attendee brings their own file, the script analyzes
it, and they can ask questions about their data interactively.

PREREQUISITES:
  $ ollama pull llama3.1
  $ pip install llama-index llama-index-llms-ollama llama-index-embeddings-huggingface llama-index-readers-file
"""

import argparse
import os
import platform
import sys
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

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from cost_estimator import format_cost_comparison

# ── Constants ─────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".csv", ".docx"}

FILE_TYPE_NAMES = {
    ".txt": "Plain text",
    ".pdf": "PDF document",
    ".csv": "CSV spreadsheet",
    ".docx": "Word document",
}

SUMMARY_PROMPT = (
    "You are analyzing a document that was just loaded. "
    "Provide a concise summary covering: "
    "1) What this document is about (topic and scope), "
    "2) Key data points or findings (cite specific numbers if present), "
    "3) Three questions someone might want to ask about this data. "
    "Keep it under 200 words."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="CivicHacks 2026 — Step 4: Bring Your Own Data (BYOD)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
What this script does:
  Loads any data file you provide (.txt, .pdf, .csv, .docx), builds a
  vector search index, generates an AI summary of the contents, and
  then enters an interactive Q&A loop where you can ask questions
  about YOUR data — grounded in actual content via RAG.

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
  python scripts/demo_step4_byod.py myfile.txt
  python scripts/demo_step4_byod.py ~/Documents/report.pdf
  python scripts/demo_step4_byod.py data/cityhack_boston_311.txt
  python scripts/demo_step4_byod.py                              # prompted for file path
  python scripts/demo_step4_byod.py myfile.txt --model phi3:mini # use a different model
        """,
    )
    parser.add_argument(
        "file",
        nargs="?",
        default=None,
        help="Path to your data file (.txt, .pdf, .csv, .docx). If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--model",
        default="llama3.1",
        help="Ollama model to use (default: llama3.1)",
    )
    return parser.parse_args()


def validate_file(filepath_str):
    """Validate and resolve a user-provided file path."""
    filepath = Path(filepath_str).expanduser().resolve()

    if not filepath.exists():
        print(f"\n  ❌ File not found: {filepath}")
        print("     Check the path and try again.\n")
        sys.exit(1)

    if not filepath.is_file():
        print(f"\n  ❌ '{filepath}' is a directory. Please provide a file path.\n")
        sys.exit(1)

    ext = filepath.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        print(f"\n  ❌ Unsupported file type: {ext}")
        print(f"     Supported: {supported}\n")
        sys.exit(1)

    if filepath.stat().st_size == 0:
        print(f"\n  ❌ File is empty: {filepath}\n")
        sys.exit(1)

    size_bytes = filepath.stat().st_size
    if size_bytes > 10_000_000:  # 10 MB
        print(f"\n  ⚠️  Large file ({format_file_size(size_bytes)}). Indexing may take a moment...")

    return filepath


def format_file_size(size_bytes):
    """Convert bytes to a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def analyze_file(filepath):
    """Print file metadata and load documents via LlamaIndex."""
    ext = filepath.suffix.lower()
    file_type = FILE_TYPE_NAMES.get(ext, ext)
    size = format_file_size(filepath.stat().st_size)
    modified = datetime.fromtimestamp(filepath.stat().st_mtime).strftime("%B %d, %Y")

    print(f"{'─' * 60}")
    print(f"  📄 File Analysis")
    print(f"{'─' * 60}\n")
    print(f"   File:      {filepath.name}")
    print(f"   Path:      {filepath}")
    print(f"   Type:      {file_type}")
    print(f"   Size:      {size}")
    print(f"   Modified:  {modified}\n")

    # Load with LlamaIndex
    print("   Loading file...")
    try:
        documents = SimpleDirectoryReader(input_files=[str(filepath)]).load_data()
    except Exception as e:
        print(f"\n  ❌ Could not read file: {e}")
        print("     The file may be encrypted, image-only (for PDFs), or in an unsupported encoding.\n")
        sys.exit(1)

    if not documents or all(len(d.text.strip()) == 0 for d in documents):
        print("\n  ❌ Could not extract text from this file.")
        print("     For PDFs, it may be image-based or encrypted. Try a text-based file.\n")
        sys.exit(1)

    total_chars = sum(len(d.text) for d in documents)
    total_words = sum(len(d.text.split()) for d in documents)

    print(f"   Content:   {len(documents)} document(s), {total_chars:,} characters, ~{total_words:,} words\n")

    # Content preview (first 300 chars)
    preview_text = documents[0].text[:300].replace("\n", " ").strip()
    if len(documents[0].text) > 300:
        preview_text += "..."
    print(f"   Preview:")
    print(f'   "{preview_text}"\n')
    print(f"{'─' * 60}\n")

    return documents


def generate_summary(query_engine, filepath_name):
    """Generate an AI summary of the loaded data."""
    print(f"{'─' * 60}")
    print(f"  🤖 AI Summary of: {filepath_name}")
    print(f"{'─' * 60}\n")

    start = time.time()
    response = query_engine.query(SUMMARY_PROMPT)
    response.print_response_stream()
    elapsed = time.time() - start

    # Estimate tokens (~1.3 tokens per word for English)
    response_text = str(response)
    est_output_tokens = int(len(response_text.split()) * 1.3)
    est_input_tokens = int(len(SUMMARY_PROMPT.split()) * 1.3) + 200

    cost_line = format_cost_comparison(elapsed, est_input_tokens, est_output_tokens)
    print(f"\n\n⏱️  {elapsed:.1f}s · ~{est_output_tokens} tokens")
    print(cost_line)
    print()


def interactive_loop(query_engine, filepath_name, hostname):
    """Interactive Q&A loop — ask questions about your data."""
    print(f"{'═' * 60}")
    print(f"  💬 Interactive Q&A — Ask anything about your data")
    print(f"     Type 'quit' to end | 'help' for commands")
    print(f"{'═' * 60}\n")

    question_count = 0

    while True:
        try:
            user_input = input("  [You] >> ").strip()
        except EOFError:
            break

        if not user_input:
            print("  (Type a question, or 'quit' to exit.)\n")
            continue

        cmd = user_input.lower()
        if cmd in ("quit", "exit", "q"):
            break
        elif cmd == "help":
            print()
            print("  Commands:")
            print("    quit / exit / q   End the session")
            print("    summary           Re-generate the AI summary")
            print("    help              Show this message")
            print("    (anything else)   Ask a question about your data")
            print()
            continue
        elif cmd == "summary":
            generate_summary(query_engine, filepath_name)
            continue

        # Regular question
        question_count += 1
        print(f"\n{'─' * 60}")
        print(f"  💬 Question: {user_input}")
        print(f"{'─' * 60}\n")
        print("  🤖 Answer:\n")

        start = time.time()
        response = query_engine.query(user_input)
        response.print_response_stream()
        elapsed = time.time() - start

        # Estimate tokens (~1.3 tokens per word for English)
        response_text = str(response)
        est_output_tokens = int(len(response_text.split()) * 1.3)
        est_input_tokens = int(len(user_input.split()) * 1.3) + 200

        cost_line = format_cost_comparison(elapsed, est_input_tokens, est_output_tokens)
        print(f"\n\n⏱️  {elapsed:.1f}s · ~{est_output_tokens} tokens")
        print(cost_line)
        print()

    # Session summary
    print(f"\n{'═' * 60}")
    print(f"  ✅ Session complete — {question_count} question{'s' if question_count != 1 else ''} answered")
    print(f"     All processing done locally on {hostname}.")
    print(f"     Zero data sent to the cloud.")
    print(f"{'═' * 60}\n")


def main():
    args = parse_args()

    hostname = platform.node()
    now = datetime.now().strftime("%B %d, %Y at %I:%M:%S %p")

    # ── Banner ────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  🏛️  CIVICHACKS 2026 — Bring Your Own Data")
    print(f"{'═' * 60}\n")

    # ── Step A: Get the file path ─────────────────────────────────
    if args.file:
        filepath = validate_file(args.file)
    else:
        print("  No file specified. Drag a file into the terminal or type a path:\n")
        try:
            raw = input("  File path >> ").strip().strip("'\"")
        except (EOFError, KeyboardInterrupt):
            print("\n")
            sys.exit(0)
        if not raw:
            print("\n  No file provided. Exiting.\n")
            sys.exit(0)
        filepath = validate_file(raw)

    # ── Step B: Configure local AI stack ──────────────────────────
    print(f"\n⚙️  Configuring local AI stack...")
    print(f"   Host: {hostname}")
    print(f"   Time: {now}")
    print(f"   Model: {args.model} (via Ollama — running on {hostname})")
    print(f"   Embeddings: all-MiniLM-L6-v2 (runs on CPU)\n")

    Settings.llm = Ollama(model=args.model, request_timeout=120.0)
    Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

    # ── Step C: Analyze and load the file ─────────────────────────
    documents = analyze_file(filepath)

    # ── Step D: Build the vector index ────────────────────────────
    print("🔍 Building vector index (this is the 'RAG' magic)...")
    start = time.time()
    index = VectorStoreIndex.from_documents(documents)
    elapsed = time.time() - start
    print(f"   Index built in {elapsed:.1f}s\n")

    query_engine = index.as_query_engine(streaming=True, similarity_top_k=3)

    # ── Step E: Generate AI summary ───────────────────────────────
    generate_summary(query_engine, filepath.name)

    # ── Step F: Interactive Q&A ───────────────────────────────────
    interactive_loop(query_engine, filepath.name, hostname)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  👋 Interrupted. Goodbye!\n")
        sys.exit(0)
    except ConnectionError:
        print("\n  ❌ Could not connect to Ollama. Is it running?")
        print("     Try: ollama serve\n")
        sys.exit(1)
