"""
╔══════════════════════════════════════════════════════════════╗
║  CIVICHACKS 2026 — LIVE DEMO STEP 2                          ║
║  "Connecting AI to Real Civic Data"                          ║
║                                                              ║
║  Proves: In ~15 lines of code, you can query real civic      ║
║  documents with a local AI — no APIs, no cost                ║
║  Time on stage: ~90 seconds (code is pre-written, just run)  ║
╚══════════════════════════════════════════════════════════════╝

Run this during the Stack Evaluation segment (0:15-0:28).
The audience sees their voted track's data get loaded and queried.

PREREQUISITES:
  $ ollama pull llama3.1
  $ pip install llama-index llama-index-llms-ollama llama-index-embeddings-huggingface
"""

import argparse
import os
import platform
import sys
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
from cost_estimator import format_cost_comparison

# ── Track configuration ────────────────────────────────────────────
# The audience voted on a track — swap the data file & sample query
TRACKS = {
    "eco": {
        "name": "🌿 EcoHack",
        "file": "ecohack_boston_environment.txt",
        "queries": [
            "Which Boston neighborhoods have the worst air quality and why?",
            "What are the biggest environmental justice concerns in this data?",
            "How is climate change specifically threatening Boston's coastline?",
        ],
    },
    "city": {
        "name": "🏙️ CityHack",
        "file": "cityhack_boston_311.txt",
        "queries": [
            "Which neighborhoods have the longest 311 response times and what are the equity implications?",
            "What are the biggest service gaps for non-English speaking residents?",
            "What patterns suggest systemic inequity in city service delivery?",
        ],
    },
    "edu": {
        "name": "📚 EduHack",
        "file": "eduhack_boston_schools.txt",
        "queries": [
            "What are the most significant achievement gaps in Boston public schools?",
            "How does transportation affect student attendance and outcomes?",
            "What technology access barriers exist for students and teachers?",
        ],
    },
    "justice": {
        "name": "⚖️ JusticeHack",
        "file": "justicehack_ma_justice.txt",
        "queries": [
            "What racial disparities exist in pretrial detention in Massachusetts?",
            "How effective are reentry programs at reducing recidivism?",
            "What does the data reveal about policing patterns in Boston?",
        ],
    },
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="CivicHacks 2026 — Step 2: Connecting AI to Real Civic Data (RAG)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
What this script does:
  Loads a track-specific civic dataset, builds a vector search index,
  and queries it using a local Llama 3.1 model via Ollama. The AI
  response is grounded in actual data — citing real statistics and
  findings from the documents (Retrieval Augmented Generation).

Available tracks:
  eco       🌿 EcoHack — Boston environmental quality data
  city      🏙️  CityHack — Boston 311 service request data (default)
  edu       📚 EduHack — Boston public schools equity data
  justice   ⚖️  JusticeHack — MA criminal justice reform data

Each track has 3 sample questions (numbered 1-3). By default a random
question is selected. Use -q to pick a specific one, or --all for all three.

Prerequisites:
  1. Install Ollama        https://ollama.com
  2. Pull the model        ollama pull llama3.1
  3. Install dependencies  pip install -r requirements.txt

Examples:
  python scripts/demo_step2_rag.py city          # Random question from CityHack
  python scripts/demo_step2_rag.py justice 2     # Question 2 from JusticeHack
  python scripts/demo_step2_rag.py eco --all     # All 3 EcoHack questions
  python scripts/demo_step2_rag.py               # Random question, city track
        """,
    )
    parser.add_argument(
        "track",
        nargs="?",
        default="city",
        choices=list(TRACKS.keys()),
        help="Hackathon track to query (default: city)",
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        type=int,
        help="Question number to ask (1-3). If omitted, picks a random question.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_queries",
        help="Run all 3 sample questions for the track",
    )
    args = parser.parse_args()

    # Validate question number
    if args.question is not None:
        if args.question < 1 or args.question > 3:
            parser.error(f"Question number must be 1-3, got {args.question}")
        if args.all_queries:
            parser.error("Cannot use --all with a specific question number")

    return args

def main():
    args = parse_args()

    track = TRACKS[args.track]
    data_dir = Path(__file__).parent.parent / "data"
    data_file = data_dir / track["file"]

    hostname = platform.node()
    now = datetime.now().strftime("%B %d, %Y at %I:%M:%S %p")

    print(f"\n{'═' * 60}")
    print(f"  CIVICHACKS 2026 — RAG Demo: {track['name']}")
    print(f"{'═' * 60}\n")

    # ── Step A: Configure local AI (no API keys!) ──────────────────
    print("⚙️  Configuring local AI stack...")
    print(f"   Host: {hostname}")
    print(f"   Time: {now}")
    print(f"   Model: llama3.1 (via Ollama — running on {hostname})")
    print("   Embeddings: all-MiniLM-L6-v2 (runs on CPU)\n")

    Settings.llm = Ollama(model="llama3.1", request_timeout=120.0)
    Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

    # ── Step B: Load civic data ────────────────────────────────────
    print(f"📄 Loading civic data: {track['file']}")
    documents = SimpleDirectoryReader(input_files=[str(data_file)]).load_data()
    print(f"   Loaded {len(documents)} document(s), {sum(len(d.text) for d in documents):,} characters\n")

    # ── Step C: Build search index ─────────────────────────────────
    print("🔍 Building vector index (this is the 'RAG' magic)...")
    start = time.time()
    index = VectorStoreIndex.from_documents(documents)
    elapsed = time.time() - start
    print(f"   Index built in {elapsed:.1f}s\n")

    # ── Step D: Query! ─────────────────────────────────────────────
    query_engine = index.as_query_engine(streaming=True, similarity_top_k=3)

    # Decide which questions to run
    all_queries = track["queries"]
    if args.all_queries:
        selected = list(enumerate(all_queries, 1))
    elif args.question is not None:
        idx = args.question - 1
        selected = [(args.question, all_queries[idx])]
    else:
        # Pick a random question so repeated runs feel fresh
        import random
        idx = random.randrange(len(all_queries))
        selected = [(idx + 1, all_queries[idx])]

    for qnum, query in selected:
        print(f"{'─' * 60}")
        print(f"💬 Question {qnum}/{len(all_queries)}: {query}\n")
        print("🤖 Answer:\n")

        start = time.time()
        response = query_engine.query(query)
        response.print_response_stream()
        elapsed = time.time() - start

        # Estimate tokens from response text (~1.3 tokens per word for English)
        response_text = str(response)
        est_output_tokens = int(len(response_text.split()) * 1.3)
        est_input_tokens = int(len(query.split()) * 1.3) + 200  # query + RAG context

        cost_line = format_cost_comparison(elapsed, est_input_tokens, est_output_tokens)
        print(f"\n\n⏱️  {elapsed:.1f}s · ~{est_output_tokens} tokens")
        print(cost_line)
        print()

    print(f"{'═' * 60}")
    print(f"✅ Real civic data + local AI + zero cost = civic tech prototype")
    print(f"{'═' * 60}\n")

if __name__ == "__main__":
    main()
