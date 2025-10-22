"""Embedding backends for the indexing pipeline."""

from __future__ import annotations

import getpass
import os
from typing import Optional

from dotenv import load_dotenv
from huggingface_hub import login
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

# Load .env variables (no-op if the file is absent)
load_dotenv()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class OpenAIEmbedding:
    """Wrapper around LangChain's OpenAI embeddings."""

    def __init__(self, model_name: str = "text-embedding-3-small") -> None:
        api_key = self._ensure_api_key()
        # LangChain reads the key from the environment, but we keep it explicit here.
        self.embeddings_model = OpenAIEmbeddings(model=model_name, api_key=api_key)

    def embed_query(self, text: str) -> list[float]:
        """Return embeddings for the provided text."""
        return self.embeddings_model.embed_query(text)

    @staticmethod
    def _ensure_api_key() -> str:
        """Fetch the OpenAI API key from env or prompt the user once."""
        key: Optional[str] = os.getenv("OPENAI_API_KEY")
        if key:
            return key
        key = getpass.getpass("Enter API key for OpenAI: ")
        os.environ["OPENAI_API_KEY"] = key
        return key


class Gemma300MEmbeddings:
    """Wrapper for the Gemma 3.0 embedding model."""

    def __init__(self) -> None:
        hf_token = self._ensure_hf_token()
        login(token=hf_token)
        self.embedding_model = SentenceTransformer("google/embeddinggemma-300m")

    def embed_query(self, text: str) -> list[float]:
        embeddings = self.embedding_model.encode([text])
        return embeddings[0].tolist()

    @staticmethod
    def _ensure_hf_token() -> str:
        """Fetch the Hugging Face token from env or prompt the user."""
        key: Optional[str] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if key:
            return key
        key = getpass.getpass("Enter Hugging Face token: ")
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = key
        return key
