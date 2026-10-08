from langchain_ollama import ChatOllama
from config import OLLAMA_CHAT_MODEL, OLLAMA_BASE_URL

def should_escalate(intent, sentiment, state):
    reasons = []

    if intent == "human_agent":
        reasons.append("Customer explicitly requested a human agent.")

    if sentiment in {"frustrated", "urgent"}:
        reasons.append(f"Customer sentiment is {sentiment}.")

    if state.get("failed_attempts", 0) >= 2:
        reasons.append("Automated resolution has failed multiple times.")

    if intent in {"payment_issue", "complaint"} and sentiment in {"negative", "frustrated", "urgent"}:
        reasons.append("Sensitive or unresolved support issue.")

    return bool(reasons), " ".join(reasons)

def create_escalation_summary(state):
    history = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in state["messages"]
    )

    prompt = """Create a concise handoff summary for a human customer-support agent.
Include the issue, known customer/order information, intent, sentiment,
what has already been tried, and what the human should do next.

Intent: {intent}
Sentiment: {sentiment}
Customer: {customer}

Conversation:
{history}
""".format(
        intent=state.get("intent"),
        sentiment=state.get("sentiment"),
        customer=state.get("customer"),
        history=history,
    )

    return ChatOllama(
        model=OLLAMA_CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0
    ).invoke(prompt).content
