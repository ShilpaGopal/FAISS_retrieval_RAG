SIMPLE_QNA_PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

def get_simple_qna_prompt(question: str, context: str) -> str:
    """Generate a prompt for simple question-answering tasks."""
    return SIMPLE_QNA_PROMPT_TEMPLATE.format(question=question, context=context)