import os
import sys
import litellm
from langchain_litellm import ChatLiteLLM
from langchain_core.messages import SystemMessage, HumanMessage

def build_llm():
    """
    Builds a ChatLiteLLM instance based on environment variables.
    
    Expected Env Vars:
    - ACRO_MODEL: The model string (e.g., 'gemini/gemini-2.0-flash', 'openai/gpt-4o')
    - ACRO_API_KEY: (Optional) The API key for the chosen provider. 
      If not set, LiteLLM will look for standard env vars like GEMINI_API_KEY, OPENAI_API_KEY, etc.
    """
    model_name = os.getenv("ACRO_MODEL", "gemini/gemini-2.0-flash")
    api_key = os.getenv("ACRO_API_KEY")
    
    # LiteLLM automatically handles mapping the prefix to the correct provider.
    # We pass the api_key if provided, otherwise LiteLLM uses standard env vars.
    try:
        # We increase the max_tokens to accommodate reasoning-heavy models.
        llm = ChatLiteLLM(
            model=model_name,
            api_key=api_key,
            temperature=0.7,
            max_tokens=2048,
            # drop_params=True ensures that parameters not supported by certain providers
            # (like 'system' or certain penalties) are filtered out automatically.
            drop_params=True,
            # Increasing timeout for slow OpenRouter free tier models
            request_timeout=60
        )
        return llm
    except Exception as e:
        print(f"  ❌ Error initializing LLM: {e}")
        if "Provider NOT provided" in str(e):
            print("\n  💡 TIP: LiteLLM needs a provider prefix in the ACRO_MODEL string.")
            print(f"     Example: 'openrouter/{model_name}' or 'nvidia_nim/{model_name}'")
        return None

if __name__ == "__main__":
    # Quick test
    llm = build_llm()
    if llm:
        print(f"  ✅ LLM initialized: {os.getenv('ACRO_MODEL', 'gemini/gemini-2.0-flash')}")
