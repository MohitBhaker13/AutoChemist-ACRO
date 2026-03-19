"""
main.py – ACRO Autonomous Chemical Research Orchestrator
=========================================================
Terminal-first runner.  Run with:   python main.py
"""

import os
import sys
import time
import textwrap

import pandas as pd

from lab_sim import simulate_reaction
from researcher import propose_next_experiment

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
MAX_ITERATIONS = 20
API_COOLDOWN   = 4          # seconds between iterations (free-tier rate limit)
HISTORY_FILE   = "experiment_history.csv"
SPEC_FILE      = "research_spec.md"
HEADERS        = ["iteration", "temp", "catalyst_pct", "polarity", "yield", "hypothesis"]
WIDTH          = 70         # console column width

# ──────────────────────────────────────────────────────────────────────────────
# Console helpers
# ──────────────────────────────────────────────────────────────────────────────

def _bar(char="─"):
    print(char * WIDTH)

def _header(title: str, char="═"):
    print()
    print(char * WIDTH)
    print(f"  {title}")
    print(char * WIDTH)

def _section(title: str):
    print()
    print(f"  ┌─ {title}")

def _line(label: str, value):
    print(f"  │  {label:<22} {value}")

def _end():
    print(f"  └{'─' * (WIDTH - 3)}")

def _wrap(text: str, indent: int = 5) -> str:
    """Wrap long strings to fit the terminal width."""
    pad = " " * indent
    lines = textwrap.wrap(text, width=WIDTH - indent)
    return ("\n" + pad).join(lines)


def _reset_history_file():
    """Always create a fresh, empty history file at the start of a run."""
    pd.DataFrame(columns=HEADERS).to_csv(HISTORY_FILE, index=False)


# ──────────────────────────────────────────────────────────────────────────────
# Print routines
# ──────────────────────────────────────────────────────────────────────────────

def _print_welcome():
    model_name = os.getenv("ACRO_MODEL", "gemini/gemini-2.0-flash")
    print()
    print("═" * WIDTH)
    print("  🧬  ACRO — Autonomous Chemical Research Orchestrator".center(WIDTH))
    print("      Suzuki-Miyaura Coupling Optimisation".center(WIDTH))
    print(f"      🤖 Model: {model_name}  (via LiteLLM)".center(WIDTH))
    print("═" * WIDTH)
    
    # Check for keys. LiteLLM is provider-agnostic, but we should at least check for ACRO_API_KEY 
    # or the corresponding provider key.
    if not os.getenv("ACRO_API_KEY") and not os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print()
        print("  ⚠️  No API key detected.")
        print("     Dr. ACRO will use random exploration for ALL iterations.")
        print("     Set ACRO_API_KEY or your provider's native key to enable AI search.")
    print()


def _print_iteration_header(i: int, total: int):
    print()
    print("─" * WIDTH)
    print(f"  ITERATION {i} / {total}".center(WIDTH))
    print("─" * WIDTH)


def _print_thinking(iteration: int):
    if iteration == 1:
        print("\n  🤔 Dr. ACRO thinking …")
        print("     No prior data — starting with an exploratory random point.\n")
    else:
        print(f"\n  🤔 Dr. ACRO analysing {iteration - 1} prior experiment(s) …")
        # Show best result so far from saved history
        try:
            df = pd.read_csv(HISTORY_FILE)
            if not df.empty:
                best = df.loc[df["yield"].idxmax()]
                print(f"     📌 Best yield so far: {best['yield']:.2f}% "
                      f"(Iteration {int(best['iteration'])})")
        except Exception:
            pass
        print()


def _print_proposal(params: dict):
    _section("Dr. ACRO's Proposal")
    _line("🌡️  Temperature:",    f"{params['temp']:.1f} °C")
    _line("⚗️  Catalyst load:",  f"{params['catalyst_pct']:.2f} mol%")
    _line("🧪 Solvent polarity:", f"{params['polarity']:.2f}")
    _end()
    print()
    print("  💡 Hypothesis:")
    print(f"     {_wrap(params['hypothesis'])}")


def _print_simulation():
    print()
    print("  ⚙️  Running surrogate reactor simulation …")


def _print_result(params: dict, yield_pct: float, iteration: int):
    _section(f"Result — Iteration {iteration}")
    _line("📈 Yield:",           f"{yield_pct:.2f} %")
    _line("🌡️  Temperature:",   f"{params['temp']:.1f} °C")
    _line("⚗️  Catalyst load:", f"{params['catalyst_pct']:.2f} mol%")
    _line("🧪 Polarity:",        f"{params['polarity']:.2f}")
    _end()


def _print_summary(df: pd.DataFrame):
    print()
    print("═" * WIDTH)
    print("  🏁  ACRO RUN COMPLETE".center(WIDTH))
    print("═" * WIDTH)

    best_idx = df["yield"].idxmax()
    best = df.loc[best_idx]

    _section("🏆 Best Experiment Found")
    _line("Yield:",           f"{best['yield']:.2f} %")
    _line("Found at iter.:",  int(best['iteration']))
    _line("Temperature:",     f"{best['temp']:.1f} °C")
    _line("Catalyst load:",   f"{best['catalyst_pct']:.2f} mol%")
    _line("Polarity:",        f"{best['polarity']:.2f}")
    _end()
    print()
    print(f"  💡 Winning hypothesis:")
    print(f"     {_wrap(best['hypothesis'])}")
    print()

    # Yield progression table
    _section("Yield Progression")
    print(f"  │  {'Iter':>4}  {'Temp':>6}  {'Cat%':>5}  {'Polar':>5}  {'Yield':>6}")
    print(f"  │  {'─'*4}  {'─'*6}  {'─'*5}  {'─'*5}  {'─'*6}")
    for _, row in df.iterrows():
        marker = " ◀ best" if int(row["iteration"]) == int(best["iteration"]) else ""
        print(f"  │  {int(row['iteration']):>4}  {row['temp']:>6.1f}  "
              f"{row['catalyst_pct']:>5.2f}  {row['polarity']:>5.2f}  "
              f"{row['yield']:>6.2f}%{marker}")
    _end()

    print()
    print(f"  📁 Results saved → {HISTORY_FILE}")
    print("═" * WIDTH)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────

def run():
    _reset_history_file()
    _print_welcome()

    # Allow user to select depth of research
    try:
        val = input(f"  📝 How many iterations for this session? (Default {MAX_ITERATIONS}): ").strip()
        iters = int(val) if val else MAX_ITERATIONS
    except ValueError:
        iters = MAX_ITERATIONS
    
    print(f"  🚀 Starting {iters} iterations of automated research...")

    for i in range(1, iters + 1):
        _print_iteration_header(i, iters)

        # 1. Show thinking state
        _print_thinking(i)

        # 2. Get proposal from Dr. ACRO (researcher.py prints raw reasoning)
        params = propose_next_experiment(HISTORY_FILE, SPEC_FILE)

        # 3. Show what was proposed
        _print_proposal(params)

        # 4. Simulate
        _print_simulation()
        yield_pct = simulate_reaction(
            temp=params["temp"],
            catalyst_pct=params["catalyst_pct"],
            polarity=params["polarity"],
        )

        # 5. Show result
        _print_result(params, yield_pct, i)

        # 6. Persist
        new_row = pd.DataFrame([{
            "iteration":    i,
            "temp":         params["temp"],
            "catalyst_pct": params["catalyst_pct"],
            "polarity":     params["polarity"],
            "yield":        round(yield_pct, 2),
            "hypothesis":   params["hypothesis"],
        }])
        new_row.to_csv(HISTORY_FILE, mode="a", header=False, index=False)

        # 7. Rate-limit cooldown
        if i < MAX_ITERATIONS:
            print(f"\n  ⏳ Cooling down {API_COOLDOWN}s before next iteration …")
            time.sleep(API_COOLDOWN)

    # 8. Final summary
    df = pd.read_csv(HISTORY_FILE)
    if not df.empty:
        _print_summary(df)
    else:
        print("\n  ℹ️  No experiments were completed.")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\n\n  ⏹️  Run interrupted by user.\n")
        sys.exit(0)
