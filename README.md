# FAISS Retrieval RAG

Build an end‑to‑end Retrieval-Augmented Generation (RAG) prototype with LangChain, FAISS, and a Gemma-based/OpenAI embedding model. The project includes a small indexing pipeline that fetches a sample blog post, chunks it, and stores embeddings in FAISS, plus an interactive question-answer graph retrieved context and answer.

## Prerequisites
- Python 3.10+
- `make`
- Access tokens:
  - `OPENAI_API_KEY` for the chat model
  - `HUGGINGFACEHUB_API_TOKEN` for the Gemma 3.0 embedding model
  - (Optional) `USER_AGENT` if you want the HTML loader to avoid anonymous warnings

## Create the Virtual Environment
```bash
make venv
source .venv/bin/activate
```

This provisions `.venv/` and installs everything in `requirements.txt`. Run `make clean` to delete the environment.

## Quickstart: Ask a Question
```bash
python src/qna.py
```

What happens:
1. The indexing pipeline downloads the sample Lilian Weng article, prints tidy step-by-step logs, splits the text into overlapping chunks, and populates an in-memory FAISS index.
2. A LangGraph state machine runs two nodes: `retrieve` (similarity search) and `generate` (LLM call).  
3. The script prints a concise, indented summary of the retrieved context and then the generated answer.

Set `OPENAI_API_KEY` and `HUGGINGFACEHUB_API_TOKEN` in your shell or `.env` before running. The code automatically disables tokenizer parallelism to avoid Hugging Face fork warnings; override by exporting `TOKENIZERS_PARALLELISM=true` if desired.

### Sample Output (trimmed)
```
[Indexer] Load step
  • Loaded 1 document(s) from https://lilianweng.github.io/posts/2023-06-23-agent/
    - Doc 1: https://... (43047 chars)
      Preview: LLM-powered autonomous agents are …

[Indexer] Persist step
  • Added 63 chunk(s) to the FAISS store.

[QnA] Running graph
  • Question: What is Task Decomposition?
...
[QnA] Answer
  • Task Decomposition is the process of breaking down...
```
