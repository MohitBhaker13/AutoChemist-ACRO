# ACRO: Autonomous Chemical Research Orchestrator🧬🧪

ACRO is a terminal-first autonomous agent that helps optimize chemical reactions. It combines LLM reasoning with a high-quality simulator to find the best reaction conditions like temperature, catalyst amount, and solvent type through repeated experiments.

## 🌟 Inspiration

ACRO is inspired by **Andrej Karpathy** and his idea of **autonomous research agents**. The goal is to build systems that keep improving on their own without human involvement.

ACRO works in a continuous loop. It runs an experiment, checks the result such as reaction yield, updates its understanding, and tries again with better settings. Each cycle builds on the previous one, similar to how an agent improves a training system step by step to get better performance over time. The focus is on continuous improvement, not just generating outputs.

---

## 🏗️ Architecture

ACRO is built on a modular, provider-agnostic architecture using the latest in AI orchestration:

- **Orchestration**: Built using **LangChain** to manage the workflow while keeping research logic separate from model APIs
- **Connectivity**: Uses **LiteLLM** so it can switch between providers like OpenAI, Gemini, Anthropic, and OpenRouter with simple configuration
- **Simulation**: A specialized `lab_sim.py` module that models a Pd-catalyzed Suzuki-Miyaura cross-coupling, accounting for factors like catalyst decomposition and reactant solubility.
- **Smart Context**: Implements a "Sliding Window" memory that prioritizes the **Best Overall Result** and the **Most Recent History**, ensuring the AI stays focused even during deep optimization runs.
- **Parsing Layer**: A regex-based JSON extractor that can handle structured outputs even when models include extra reasoning text

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- A terminal (PowerShell, Bash, or CMD)

### 2. Installation
Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Configuration
ACRO uses two primary environment variables to control its AI client:

**Windows (PowerShell):**
```powershell
$env:ACRO_MODEL = "openrouter/nvidia/nemotron-3-super-120b-a12b:free" # Or your preferred model
$env:ACRO_API_KEY = "your-api-key-here"
```

**macOS / Linux:**
```bash
export ACRO_MODEL="gemini/gemini-2.0-flash"
export ACRO_API_KEY="your-api-key-here"
```

---

## 🧪 Usage

Simply run the main orchestrator:
```bash
python main.py
```

### What happens next?
1. **Interactive Setup**: ACRO will ask you how many iterations you want for the session.
2. **Fresh Start**: The experiment history is reset to ensure a clean research slate.
3. **The Loop**: 
   - Dr. ACRO (the AI) analyzes your data and proposes a chemical hypothesis.
   - The proposal is sent to the lab simulator.
   - The result is recorded, and the AI refines its strategy for the next step.
4. **Final Summary**: At the end of the run, ACRO provides a detailed report of the best conditions discovered.

---

## 📁 Project Structure

- `main.py`: The terminal UI and experiment loop engine.
- `researcher.py`: The AI researcher logic and prompt engineering.
- `llm_client.py`: The universal LLM provider adapter.
- `lab_sim.py`: The underlying chemical reaction model.
- `research_spec.md`: The detailed research objective and constraints.

---

*“Building the future of autonomous discovery, today.”* 🧪✨
