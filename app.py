import streamlit as st

from chatbot.state import new_state, add_message
from chatbot.intent import classify_intent
from chatbot.sentiment import analyze_sentiment
from chatbot.escalation import should_escalate, create_escalation_summary
from chatbot.response import answer_with_rag
from utils.logger import init_db, save_conversation
from admin.dashboard import show_dashboard

st.set_page_config(
    page_title="Customer Support Chatbot",
    page_icon="💬",
    layout="wide"
)

init_db()

if "state" not in st.session_state:
    st.session_state.state = new_state()

state = st.session_state.state

page = st.sidebar.radio(
    "Navigate",
    ["Customer Chat", "Admin Dashboard"]
)

if page == "Admin Dashboard":
    show_dashboard()
    st.stop()

st.title("💬 Customer Support Chatbot")
st.caption("RAG + Intent Classification + Sentiment + Human Escalation")

for msg in state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_message = st.chat_input("How can we help you?") # user message

if user_message:
    add_message(state, "user", user_message)

    with st.spinner("Analyzing your request..."):
        intent = classify_intent(user_message)
        sentiment = analyze_sentiment(user_message)

        state["intent"] = intent.intent
        state["sentiment"] = sentiment.sentiment

        escalate, reason = should_escalate(
            state["intent"],
            state["sentiment"],
            state
        )

        if escalate:
            state["route"] = "escalation"

            state["escalation"].update({
                "required": True,
                "reason": reason,
                "status": "pending_human",
                "summary": create_escalation_summary(state),
            })

            response = (
                "I’m going to escalate this conversation to a human support agent. "
                "I’ve included your conversation context so you do not need to repeat everything."
            )

        else:
            state["route"] = "rag"

            try:
                response = answer_with_rag(
                    user_message,
                    state["messages"]
                )
            except Exception:
                state["failed_attempts"] += 1
                response = (
                    "I couldn't retrieve the relevant support information right now. "
                    "Please try again or ask for a human agent."
                )

    add_message(state, "assistant", response)
    save_conversation(state)
    st.rerun()

with st.sidebar.expander("Current state"):
    st.json({
        "intent": state["intent"],
        "sentiment": state["sentiment"],
        "route": state["route"],
        "escalated": state["escalation"]["required"],
        "failed_attempts": state["failed_attempts"],
    })

if state["escalation"]["required"]:
    st.warning("This conversation is marked for human escalation.")

    with st.expander("Human-agent handoff summary"):
        st.write(state["escalation"]["summary"])
