def new_state():
    return {
        "messages": [],
        "intent": None,
        "sentiment": None,
        "route": None,
        "failed_attempts": 0,
        "customer": {"name": None, "email": None, "order_id": None},
        "escalation": {
            "required": False,
            "reason": None,
            "summary": None,
            "status": "not_escalated",
        },
    }

def add_message(state, role, content):
    state["messages"].append({"role": role, "content": content})
