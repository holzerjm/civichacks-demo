"""
╔══════════════════════════════════════════════════════════════╗
║  CIVICHACKS 2026 — LIVE DEMO STEP 2                        ║
║  "Connecting AI to Real Civic Data"                         ║
║                                                              ║
║  Proves: In ~15 lines of code, you can query real civic     ║
║  documents with a local AI — no APIs, no cost               ║
║  Time on stage: ~90 seconds (code is pre-written, just run) ║
╚══════════════════════════════════════════════════════════════╝

Run this during the Stack Evaluation segment (0:15-0:28).
The audience sees their voted track's data get loaded and queried.

PREREQUISITES:
  $ ollama pull llama3.1
  $ pip install llama-index llama-index-llms-ollama llama-index-embeddings-huggingface
"""

import platform
import sys
import time
from datetime import datetime
from pathlib import Path

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

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

def main():
    # ── Parse track from command line ──────────────────────────────
    track_key = sys.argv[1] if len(sys.argv) > 1 else "city"
    if track_key not in TRACKS:
        print(f"Usage: python demo_step2_rag.py [{'|'.join(TRACKS.keys())}]")
        sys.exit(1)

    track = TRACKS[track_key]
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

    for i, query in enumerate(track["queries"], 1):
        print(f"{'─' * 60}")
        print(f"💬 Question {i}: {query}\n")
        print("🤖 Answer:\n")

        start = time.time()
        response = query_engine.query(query)
        response.print_response_stream()
        elapsed = time.time() - start

        print(f"\n\n⏱️  Answered in {elapsed:.1f}s | Cost: $0.00")
        print()

        # In live demo, you might only do 1 query and take audience questions
        if "--all" not in sys.argv:
            break

    print(f"{'═' * 60}")
    print(f"✅ Real civic data + local AI + zero cost = civic tech prototype")
    print(f"{'═' * 60}\n")

if __name__ == "__main__":
    main()
