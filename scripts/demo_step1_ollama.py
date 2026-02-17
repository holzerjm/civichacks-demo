"""
╔══════════════════════════════════════════════════════════════╗
║  CIVICHACKS 2026 — LIVE DEMO STEP 1                        ║
║  "The 60-Second AI"                                         ║
║                                                              ║
║  Proves: You can run a GPT-4-class model locally, for free  ║
║  Time on stage: ~60 seconds                                 ║
╚══════════════════════════════════════════════════════════════╝

Run this during the opening segment (0:00-0:05) right after
the DeepSeek R1 story. The audience sees a local model respond
in real time — no API key, no cloud, no cost.

PREREQUISITE: Ollama installed and model pulled
  $ ollama pull llama3.1
"""

import ollama
import platform
import sys
import time
from datetime import datetime

# ── A civic-flavored prompt to make the demo relevant ──────────────
PROMPT = """You are a civic technology advisor. In 3 concise bullet points,
explain why open source AI matters for building tools that serve
communities — especially for students at a hackathon who want to
make a real impact this weekend."""

def main():
    hostname = platform.node()
    now = datetime.now().strftime("%B %d, %Y at %I:%M:%S %p")

    print("\n🏛️  CivicHacks 2026 — Open Source AI, Running Locally\n")
    print(f"📡 Model: llama3.1 (8B) — running on {hostname}")
    print(f"🕐 Time: {now}")
    print(f"💰 Cost: $0.00")
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

    token_count = 0
    for chunk in stream:
        content = chunk["message"]["content"]
        print(content, end="", flush=True)
        token_count += len(content.split())

    elapsed = time.time() - start
    print(f"\n\n─" + "─" * 59)
    print(f"⏱️  Generated in {elapsed:.1f}s  |  ~{token_count} words  |  Cost: $0.00")
    print(f"─" * 60)
    print(f"\n✅ That's it. Local AI. Free. Private. Ready to build with.\n")

if __name__ == "__main__":
    main()
