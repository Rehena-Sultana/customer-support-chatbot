"""
Module for classifying user messages into customer support intents using Ollama LLM and LangChain.
"""

# Import BaseModel and Field from Pydantic for data modeling and schema validation
from pydantic import BaseModel, Field
# Import ChatOllama interface for communicating with Ollama models
from langchain_ollama import ChatOllama
# Import ChatPromptTemplate for building structured chat prompts
from langchain_core.prompts import ChatPromptTemplate
# Import model configurations (model name and base server URL) from local config
from config import OLLAMA_CHAT_MODEL, OLLAMA_BASE_URL


# Define schema for structured intent classification output
class IntentResult(BaseModel):
    """Data model representing the structured result of an intent classification."""
    # Predicted customer support intent category
    intent: str = Field(description="Customer support intent")
    # Confidence score restricted between 0.0 and 1.0
    confidence: float = Field(ge=0, le=1)


# List of valid target intent categories for classification
INTENTS = [
    "order_status", "return", "refund", "cancellation", "payment_issue",
    "shipping", "product_information", "account_issue", "technical_issue",
    "complaint", "human_agent", "general_question"
]


def classify_intent(message):
    """
    Classifies an incoming customer message into a predefined support intent category.

    Args:
        message (str): The text message sent by the customer.

    Returns:
        IntentResult: Parsed object containing the predicted intent and confidence score.
    """
    # Instantiate the ChatOllama model client
    llm = ChatOllama(
        # Set the Ollama chat model identifier
        model=OLLAMA_CHAT_MODEL,
        # Set the host URL for the Ollama server instance
        base_url=OLLAMA_BASE_URL,
        # Set temperature to zero for deterministic output
        temperature=0
    )
    # Configure the LLM to return structured output matching the IntentResult Pydantic schema
    structured = llm.with_structured_output(IntentResult)
    # Define system prompt and human prompt template sequence
    prompt = ChatPromptTemplate.from_messages([
        # System instructions listing all valid intent options
        ("system", "Classify the message into one of these intents: " + ", ".join(INTENTS)),
        # Placeholder for the user's message
        ("human", "{message}")
    ])
    # "chain these two things together, left to right." It says: "first run the prompt template, then feed its output into the structured AI call."
    # Pipe the prompt template into the structured LLM runner and invoke it with user input
    return (prompt | structured).invoke({"message": message})
