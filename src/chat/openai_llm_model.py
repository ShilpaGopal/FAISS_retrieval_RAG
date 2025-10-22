import getpass
import os
from typing import Optional
from langchain.chat_models import init_chat_model


def _ensure_api_key() -> str:
    """Fetch the OpenAI API key from env or prompt the user once."""
    key: Optional[str] = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    key = getpass.getpass("Enter API key for OpenAI: ")
    os.environ["OPENAI_API_KEY"] = key
    return key

def fetch_openai_model():
    """Fetch the OpenAI LLM model after ensuring the API key is set."""
    api_key = _ensure_api_key()
    llm = init_chat_model("gpt-4o-mini", model_provider="openai")
    return llm


   