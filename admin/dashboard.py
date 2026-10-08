import pandas as pd
import streamlit as st
from utils.logger import load_conversations

def show_dashboard():
    st.title("Admin Dashboard")

    rows = load_conversations()

    if not rows:
        st.info("No conversations logged yet.")
        return

    df = pd.DataFrame(rows)
    total = len(df)
    escalated = int(df["escalated"].sum())
    rate = escalated / total * 100 if total else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total conversations", total)
    c2.metric("Escalated", escalated)
    c3.metric("Escalation rate", f"{rate:.1f}%")

    st.subheader("Common topics")
    st.bar_chart(df["intent"].value_counts())

    st.subheader("Recent conversations")
    st.dataframe(
        df[["id", "created_at", "intent", "sentiment", "route", "escalated", "reason"]],
        use_container_width=True
    )

    st.subheader("Conversation detail")
    selected = st.selectbox("Select conversation", df["id"].tolist())
    row = df[df["id"] == selected].iloc[0]

    st.write("**Handoff summary:**", row["summary"] or "Not escalated.")
    st.json(row["messages"])
