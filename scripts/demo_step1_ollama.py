"""
╔══════════════════════════════════════════════════════════════╗
║  CIVICHACKS 2026 — LIVE DEMO STEP 1                        ║
║  "The 60-Second AI"                                         ║
║                                                              ║
║  Proves: You can run a GPT-4-class model locally, for free  ║
║  Time on stage: ~60 seconds                                 ║
╚══════════════════════════════════════════════════════════════╝

PREREQUISITE: Ollama installed and model pulled
  $ ollama pull llama3.1
"""

import argparse
import ollama
import platform
import sys
import time
from datetime import datetime
from cost_estimator import format_cost_comparison

# ── A civic-flavored prompt to make the demo relevant ──────────────
PROMPT = """You are a civic technology advisor. In 3 concise bullet points,
explain why open source AI matters for building tools that serve
communities — especially for students at a hackathon who want to
make a real impact this weekend."""

def parse_args():
    parser = argparse.ArgumentParser(
        description="CivicHacks 2026 — Step 1: The 60-Second AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
What this script does:
  Sends a civic-themed prompt to a local Llama 3.1 model via Ollama
  and streams the response token by token. Displays the hostname,
  timestamp, elapsed time, and $0.00 cost — proving that powerful AI
  runs locally for free.

Prerequisites:
  1. Install Ollama        https://ollama.com
  2. Pull the model        ollama pull llama3.1
  3. Start Ollama          ollama serve

Examples:
  python scripts/demo_step1_ollama.py          # Run the demo
  python scripts/demo_step1_ollama.py --help   # Show this help
        """,
    )
    return parser.parse_args()

def main():
    parse_args()

    hostname = platform.node()
    now = datetime.now().strftime("%B %d, %Y at %I:%M:%S %p")

    print("\n🏛️  CivicHacks 2026 — Open Source AI, Running Locally\n")
    print(f"📡 Model: llama3.1 (8B) — running on {hostname}")
    print(f"🕐 Time: {now}")
    print(f"🔒 Data: never leaves {hostname}\n")
    print("─" * 60)
    print(f"\n💬 Prompt: {PROMPT.strip()}\n")
    print("─" * 60)
    print("\n🤖 Response:\n")

    start = time.time()

    # Stream the response so the audience watches it generate
    stream = ollama.chat(
        model="llama3.1",
        messages=[{"role": "user", "content": PROMPT}],
        stream=True,
    )

    last_chunk = None
    for chunk in stream:
        content = chunk["message"]["content"]
        print(content, end="", flush=True)
        last_chunk = chunk

    elapsed = time.time() - start

    # Extract token counts from Ollama's final streaming chunk
    input_tokens = getattr(last_chunk, "prompt_eval_count", 0) or 0
    output_tokens = getattr(last_chunk, "eval_count", 0) or 0

    cost_line = format_cost_comparison(elapsed, input_tokens, output_tokens)

    print(f"\n\n─" + "─" * 59)
    print(f"⏱️  {elapsed:.1f}s · {output_tokens} tokens · {output_tokens/elapsed:.0f} tok/s")
    print(f"{cost_line}")
    print(f"─" * 60)
    print(f"\n✅ That's it. Local AI. Private. And virtually free.\n")

if __name__ == "__main__":
    main()
