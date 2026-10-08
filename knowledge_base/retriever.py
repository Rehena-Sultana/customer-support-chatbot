import json
import os
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from config import OLLAMA_EMBEDDING_MODEL, OLLAMA_BASE_URL, CHROMA_DIR

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAQ_PATH = os.path.join(BASE_DIR, "data", "faq.json")

def get_vectorstore():
    embeddings = OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    # If database already exists, load and return it
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        return Chroma(
            collection_name="customer_support_faq",
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )

    # If not, read faq.json and automatically build it
    with open(FAQ_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)

    docs = [
        Document(
            page_content=f"Question: {x['question']}\nAnswer: {x['answer']}",
            metadata={"id": x["id"], "category": x["category"]}
        )
        for x in items
    ]

    db = Chroma(
        collection_name="customer_support_faq",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    db.add_documents(docs, ids=[x["id"] for x in items])
    return db

def retrieve(query, k=4):
    return get_vectorstore().similarity_search(query, k=k)
