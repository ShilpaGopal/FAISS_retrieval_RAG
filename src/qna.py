from functools import partial
from textwrap import indent, shorten
from langgraph.graph import START, StateGraph
from retriever.local_faiss import State, retrieve, generate

from chat.openai_llm_model import fetch_openai_model
from indexer.pipeline import build_vector_store_from_url

def build_qna_graph(vector_store, llm):
    graph_builder = StateGraph(State)
    graph_builder.add_node("retrieve", partial(retrieve, vector_store=vector_store))
    graph_builder.add_node("generate", partial(generate, llm=llm))
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    return graph_builder.compile()


def _pretty_print_context(docs):
    print("\n[QnA] Retrieved context")
    if not docs:
        print("  • No documents returned.")
        return
    for idx, doc in enumerate(docs, start=1):
        snippet = shorten(doc.page_content.replace("\n", " "), width=500, placeholder="…")
        source = doc.metadata.get("source", "Unknown source")
        print(f"  • Chunk {idx} from {source}")
        print(indent(f"Snippet: {snippet}", prefix="    "))


if __name__ == "__main__":
    llm = fetch_openai_model()
    vector_store = build_vector_store_from_url()

    qna_graph = build_qna_graph(vector_store=vector_store, llm=llm)

    question = "What is Task Decomposition?"
    print("\n[QnA] Running graph")
    print(f"  • Question: {question}")
    result = qna_graph.invoke({"question": question})

    _pretty_print_context(result["context"])
    print("\n[QnA] Answer")
    print(f"  • {result['answer']}")
