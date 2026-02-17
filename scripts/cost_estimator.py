"""
Cost estimation for local LLM inference vs. cloud APIs.

Calculates real energy cost based on inference duration and estimated
system wattage, then compares against published cloud API pricing.
No extra dependencies required — uses only Ollama's built-in metrics
and basic math.
"""

import platform

# ── Hardware power profiles (watts during inference) ─────────────────
# Conservative whole-system estimates including screen, SSD, RAM
POWER_PROFILES = {
    "apple_base":  15,   # M1/M2/M3/M4 base (MacBook Air / 13" Pro)
    "apple_pro":   35,   # M_-Pro chips (14"/16" MacBook Pro)
    "apple_max":   50,   # M_-Max chips
    "laptop_cpu":  45,   # x86 laptop, CPU-only inference
    "desktop_cpu": 120,  # x86 desktop, CPU-only inference
    "desktop_gpu": 250,  # Desktop with discrete GPU (RTX 3060-4090)
}

# US average residential electricity rate ($/kWh) — EIA Feb 2026
ELECTRICITY_RATE = 0.18

# ── Cloud API pricing (per 1M tokens, early 2026) ───────────────────
# Each entry: (input_price_per_1M, output_price_per_1M)
CLOUD_PRICING = {
    "GPT-4o":              (2.50,  10.00),
    "GPT-4o-mini":         (0.15,   0.60),
    "Claude 3.5 Sonnet":   (3.00,  15.00),
    "Claude 3.5 Haiku":    (0.80,   4.00),
    "Gemini 2.5 Flash":    (0.15,   0.60),
    "Groq Llama 3.1 8B":   (0.05,   0.08),
}

# Default comparison model (the one most people know)
DEFAULT_COMPARISON = "GPT-4o"


def detect_power_watts():
    """Auto-detect a reasonable wattage estimate based on hardware."""
    machine = platform.machine().lower()
    proc = platform.processor().lower()

    if "arm" in machine or "aarch64" in machine:
        # Apple Silicon — check if it's a higher-end chip via sysctl
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2,
            )
            brand = result.stdout.strip().lower()
            if "max" in brand or "ultra" in brand:
                return POWER_PROFILES["apple_max"], "Apple Silicon Max/Ultra"
            elif "pro" in brand:
                return POWER_PROFILES["apple_pro"], "Apple Silicon Pro"
        except Exception:
            pass
        return POWER_PROFILES["apple_base"], "Apple Silicon"

    # x86 — assume laptop unless we can detect otherwise
    return POWER_PROFILES["laptop_cpu"], "x86 CPU"


def estimate_local_cost(duration_secs, watts=None):
    """
    Estimate the electricity cost of a local inference run.

    Args:
        duration_secs: Wall-clock seconds of generation
        watts: System power draw in watts (auto-detected if None)

    Returns:
        dict with energy_wh, cost_usd, watts, label
    """
    if watts is None:
        watts, label = detect_power_watts()
    else:
        label = f"{watts}W (manual)"

    energy_wh = (watts / 1000) * (duration_secs / 3600) * 1000  # milliwatt-hours → Wh
    # Fix: watts * seconds / 3600 = Wh
    energy_wh = watts * duration_secs / 3600
    cost_usd = (energy_wh / 1000) * ELECTRICITY_RATE  # Wh → kWh → $

    return {
        "energy_wh": energy_wh,
        "cost_usd": cost_usd,
        "watts": watts,
        "label": label,
    }


def estimate_cloud_cost(input_tokens, output_tokens, model=DEFAULT_COMPARISON):
    """
    Estimate what a cloud API would charge for the same query.

    Args:
        input_tokens: Number of prompt/input tokens
        output_tokens: Number of generated output tokens
        model: Cloud model name (key in CLOUD_PRICING)

    Returns:
        dict with cost_usd, model
    """
    if model not in CLOUD_PRICING:
        model = DEFAULT_COMPARISON

    input_rate, output_rate = CLOUD_PRICING[model]
    cost = (input_tokens / 1_000_000) * input_rate + \
           (output_tokens / 1_000_000) * output_rate

    return {"cost_usd": cost, "model": model}


def format_cost_comparison(duration_secs, input_tokens, output_tokens, watts=None):
    """
    Build a formatted cost comparison string for display.

    Returns a one-line summary like:
      ⚡ Local: $0.000008 (0.05 Wh @ 15W) · GPT-4o equivalent: $0.0021 (265x more)
    """
    local = estimate_local_cost(duration_secs, watts)
    cloud = estimate_cloud_cost(input_tokens, output_tokens)

    if local["cost_usd"] > 0:
        ratio = cloud["cost_usd"] / local["cost_usd"]
        ratio_str = f"{ratio:,.0f}x more"
    else:
        ratio_str = "∞x more"

    return (
        f"⚡ Local: ${local['cost_usd']:.6f} "
        f"({local['energy_wh']:.3f} Wh @ {local['watts']}W) · "
        f"{cloud['model']}: ${cloud['cost_usd']:.4f} ({ratio_str})"
    )


def format_cost_short(duration_secs, input_tokens, output_tokens, watts=None):
    """
    Shorter format for inline display (e.g. Gradio chat metadata).

    Returns something like:
      💰 $0.000008 local · $0.0021 on GPT-4o (265x)
    """
    local = estimate_local_cost(duration_secs, watts)
    cloud = estimate_cloud_cost(input_tokens, output_tokens)

    if local["cost_usd"] > 0:
        ratio = cloud["cost_usd"] / local["cost_usd"]
        ratio_str = f"{ratio:,.0f}x"
    else:
        ratio_str = "∞x"

    return (
        f"${local['cost_usd']:.6f} local · "
        f"${cloud['cost_usd']:.4f} on {cloud['model']} ({ratio_str})"
    )
