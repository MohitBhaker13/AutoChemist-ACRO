# ACRO: Autonomous Chemical Research Orchestrator 🧬🧪

ACRO is a terminal-first autonomous agent designed to optimize chemical reactions. It combines the reasoning power of Large Language Models (LLMs) with a high-fidelity chemical simulator to discover optimal reaction conditions (temperature, catalyst loading, and solvent polarity) through iterative experimentation.

## 🌟 Inspiration

The vision for ACRO was inspired by **Andrej Karpathy**'s (Co-founder of OpenAI) core vision for **Autonomous Training Agents**—AI systems capable of iterative, self-correcting exploration. ACRO embodies this "self-improving" loop: it observes its own experimental failures (low yields), updates its chemical hypothesis, and corrects its parameters in the next iteration. It moves beyond simple text generation to become a living, closed-loop research engine.

---

## 🏗️ Architecture

ACRO is built on a modular, provider-agnostic architecture using the latest in AI orchestration:

- **Orchestration**: Built with **LangChain**, utilizing standard message abstractions to decouple research logic from API formats.
- **Connectivity**: Integrated with **LiteLLM**, allowing the system to use any provider (Gemini, OpenAI, Anthropic, OpenRouter, etc.) by simply changing an environment variable.
- **Simulation**: A specialized `lab_sim.py` module that models a Pd-catalyzed Suzuki-Miyaura cross-coupling, accounting for factors like catalyst decomposition and reactant solubility.
- **Smart Context**: Implements a "Sliding Window" memory that prioritizes the **Best Overall Result** and the **Most Recent History**, ensuring the AI stays focused even during deep optimization runs.
- **Robust Parsing**: Uses a custom regex-based JSON extractor capable of handling specialized "Thinking" blocks and conversational reasoning from advanced models like NVIDIA Nemotron.

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
