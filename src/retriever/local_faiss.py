from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from chat.prompt import get_simple_qna_prompt


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State, vector_store):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State, llm):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = get_simple_qna_prompt(question=state["question"], context=docs_content)
    response = llm.invoke(messages)
    return {"answer": response.content}