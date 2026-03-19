"""
researcher.py – Gemini-powered research agent for ACRO.

Reads the experiment history, analyses trends, and proposes the next
experiment using the Gemini API with a Ph.D. Chemist persona.

All AI reasoning is printed to the terminal so the operator can follow
Dr. ACRO's thinking step-by-step.
"""

import json
import os
import random
import re
import textwrap
import time

import pandas as pd
from langchain_core.messages import SystemMessage, HumanMessage
from llm_client import build_llm

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
MODEL_ID    = os.getenv("ACRO_MODEL", "gemini-2.0-flash")
MAX_RETRIES = 3
WIDTH       = 70

# Parameter bounds (must match lab_sim.py)
BOUNDS = {
    "temp":         (30.0, 150.0),
    "catalyst_pct": (0.1,  5.0),
    "polarity":     (0.1,  1.0),
}


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _random_experiment() -> dict:
    """Generate a random initial guess within the parameter bounds."""
    return {
        "temp":         round(random.uniform(*BOUNDS["temp"]), 1),
        "catalyst_pct": round(random.uniform(*BOUNDS["catalyst_pct"]), 2),
        "polarity":     round(random.uniform(*BOUNDS["polarity"]), 2),
        "hypothesis":   "Initial random exploration of the parameter space.",
    }


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _build_system_prompt(spec_text: str) -> str:
    return (
        "You are Dr. ACRO, a computational chemist with a Ph.D. in "
        "organometallic chemistry from MIT. You have 12 years of experience "
        "optimizing Suzuki-Miyaura and related Pd-catalysed cross-coupling "
        "reactions.\n\n"
        "Your task is to propose the single NEXT experiment that will most "
        "efficiently find the global yield maximum for the reaction described "
        "in the research specification below.\n\n"
        "## Research Specification\n\n"
        f"{spec_text}\n\n"
        "## Rules\n"
        "1. Analyse ALL previous experiments for trends before proposing.\n"
        "2. Balance exploitation (refining near the best result) with "
        "exploration (probing under-sampled regions).\n"
        "3. Provide a 1-2 sentence chemical HYPOTHESIS justifying your choice.\n"
        "4. Reply with ONLY a single JSON object — no markdown fencing, no "
        "extra text.\n"
        '5. JSON keys: "temp", "catalyst_pct", "polarity", "hypothesis".\n'
        "6. All numeric values must be within the bounds defined above.\n"
        "7. DO NOT provide any text AFTER the JSON object. The JSON must be the LAST thing in your response."
    )


def _build_user_prompt(df: pd.DataFrame) -> str:
    # To prevent model confusion and context overflow, we only show:
    # 1. The best experiment found so far
    # 2. The most recent 8 experiments
    
    best_idx = df["yield"].idxmax()
    best_row = df.loc[[best_idx]]
    
    recent_df = df.tail(8)
    
    # Combine and remove duplicates (if best is in recent)
    summary_df = pd.concat([best_row, recent_df]).drop_duplicates().sort_values("iteration")

    lines = [
        "Here is a summary of the most relevant experiment history so far:\n",
        summary_df.to_string(index=False),
        "\n\nBased on these results, propose the next experiment. "
        "Remember to conclude your response with a valid JSON object."
    ]
    return "\n".join(lines)


def _extract_json(text: str) -> dict:
    """Try to extract a JSON object from the model response."""
    # 1. Try to find content within markdown code blocks first
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if code_block:
        try:
            return json.loads(code_block.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 2. If no code blocks, look for the first '{' and last '}'
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. Fallback: try raw strip
    return json.loads(text.strip())


def _validate_and_clamp(experiment: dict) -> dict:
    """Ensure all values are within bounds; clamp if needed."""
    return {
        "temp":         round(_clamp(float(experiment["temp"]), *BOUNDS["temp"]), 1),
        "catalyst_pct": round(_clamp(float(experiment["catalyst_pct"]), *BOUNDS["catalyst_pct"]), 2),
        "polarity":     round(_clamp(float(experiment["polarity"]), *BOUNDS["polarity"]), 2),
        "hypothesis":   str(experiment.get("hypothesis", "No reasoning provided.")),
    }


def _wrap_print(text: str, indent: int = 5):
    """Print text wrapped to terminal width."""
    pad = " " * indent
    for chunk in textwrap.wrap(text, width=WIDTH - indent):
        print(f"{pad}{chunk}")


def _print_reasoning_block(raw_response: str):
    """
    Extract and pretty-print the AI's reasoning from the raw response.
    Parses the JSON to find the hypothesis and prints the full raw response
    to the terminal so every thought is visible.
    """
    print()
    print("  ┌─ 🧠 Dr. ACRO's Raw Reasoning ─────────────────────────────")

    # Ensure raw_response is a string. Some models return a list of content blocks.
    if isinstance(raw_response, list):
        text_parts = []
        for part in raw_response:
            if isinstance(part, dict):
                # Handle standard 'text' blocks
                if "text" in part:
                    text_parts.append(part["text"])
                # Handle specialized 'thinking' blocks (common in new reasoning models)
                elif "thinking" in part:
                    # Mark thinking blocks visually
                    text_parts.append(f"[THINKING]\n{part['thinking']}\n[/THINKING]")
                else:
                    text_parts.append(str(part))
            else:
                text_parts.append(str(part))
        raw_response = "\n\n".join(text_parts)
    elif not isinstance(raw_response, str):
        raw_response = str(raw_response)

    # Print every line of the raw response, indented
    for line in raw_response.strip().splitlines():
        stripped = line.strip()
        if stripped:
            _wrap_print(stripped, indent=5)

    print(f"  └{'─' * (WIDTH - 3)}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def propose_next_experiment(
    history_path: str = "experiment_history.csv",
    spec_path: str    = "research_spec.md",
) -> dict:
    """
    Propose the next experiment parameters (CLI interface).

    Prints the AI's raw reasoning to the terminal at each call.

    Returns
    -------
    dict with keys: temp, catalyst_pct, polarity, hypothesis
    """
    df = pd.read_csv(history_path)

    if df.empty:
        print("  📋 No prior experiments — generating random initial guess.")
        return _random_experiment()

    # --- call LLM ---
    llm = build_llm()
    if not llm:
        print("  🔄 Falling back to random guess.")
        return _random_experiment()

    # Load research spec
    with open(spec_path, "r", encoding="utf-8") as f:
        spec_text = f.read()

    system_prompt = _build_system_prompt(spec_text)
    user_prompt = _build_user_prompt(df)

    BACKOFF_SECONDS = [5, 20, 60]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = llm.invoke(messages)
            raw = response.content
            
            # Ensure raw is a string (handle potential list from some providers)
            if isinstance(raw, list):
                text_parts = []
                for p in raw:
                    if isinstance(p, dict):
                        # For JSON extraction, we primarily want the 'text' parts.
                        # We include everything just in case the JSON is in a different block type.
                        text_parts.append(p.get("text", p.get("thinking", str(p))))
                    else:
                        text_parts.append(str(p))
                raw = "\n\n".join(text_parts)
            elif not isinstance(raw, str):
                raw = str(raw)

            # ── Print the AI's full raw response ──
            _print_reasoning_block(raw)

            experiment = _extract_json(raw)
            return _validate_and_clamp(experiment)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"  ⚠️  Attempt {attempt}/{MAX_RETRIES} – parse error: {e}")
            if "Expecting value" in str(e):
                print("     (Tip: LiteLLM response content might be missing a valid JSON block.)")
            if attempt == MAX_RETRIES:
                print("  🔄 All retries exhausted — falling back to random.")
                return _random_experiment()

        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
            wait = BACKOFF_SECONDS[attempt - 1] if attempt <= len(BACKOFF_SECONDS) else 60

            if is_rate_limit:
                print(f"  ⏳ Rate limited (attempt {attempt}/{MAX_RETRIES}) — waiting {wait}s …")
                time.sleep(wait)
            else:
                print(f"  ❌ API error on attempt {attempt}/{MAX_RETRIES}: {e}")

            if attempt == MAX_RETRIES:
                print("  🔄 All retries exhausted — falling back to random.")
                return _random_experiment()


# ──────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    exp = propose_next_experiment()
    print(json.dumps(exp, indent=2))
