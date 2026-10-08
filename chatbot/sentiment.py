from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from config import OLLAMA_CHAT_MODEL, OLLAMA_BASE_URL

class SentimentResult(BaseModel):
    sentiment: str = Field(description="positive, neutral, negative, frustrated, or urgent")
    confidence: float = Field(ge=0, le=1)

def analyze_sentiment(message):
    llm = ChatOllama(
        model=OLLAMA_CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0
    )
    structured = llm.with_structured_output(SentimentResult)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Analyze sentiment. Use exactly one label: positive, neutral, negative, frustrated, urgent."),
        ("human", "{message}")
    ])
    return (prompt | structured).invoke({"message": message})
