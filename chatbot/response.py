from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from config import OLLAMA_CHAT_MODEL, OLLAMA_BASE_URL
from knowledge_base.retriever import retrieve

def answer_with_rag(message, history):
    docs = retrieve(message, k=4)
    context = "\n\n".join(d.page_content for d in docs)

    llm = ChatOllama(
        model=OLLAMA_CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful customer-support assistant.
Answer using only the supplied knowledge-base context.
Do not invent company policies.
If there is not enough information, say so and suggest human escalation.

Knowledge base:
{context}"""),
        ("human", "Recent history: {history}\nCustomer: {message}")
    ])

    return (prompt | llm).invoke({
        "context": context,
        "history": history[-6:],
        "message": message
    }).content
