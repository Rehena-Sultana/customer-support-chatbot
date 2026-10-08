import os
import sqlite3
import json
from datetime import datetime, timezone
from config import LOG_DB

def init_db():
    os.makedirs(os.path.dirname(LOG_DB), exist_ok=True)
    with sqlite3.connect(LOG_DB) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            intent TEXT,
            sentiment TEXT,
            route TEXT,
            escalated INTEGER,
            reason TEXT,
            summary TEXT,
            messages TEXT
        )""")
        conn.commit()

def save_conversation(state):
    init_db()
    e = state["escalation"]

    with sqlite3.connect(LOG_DB) as conn:
        conn.execute(
            """INSERT INTO conversations
            (created_at, intent, sentiment, route, escalated, reason, summary, messages)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                state.get("intent"),
                state.get("sentiment"),
                state.get("route"),
                int(e["required"]),
                e["reason"],
                e["summary"],
                json.dumps(state["messages"]),
            )
        )
        conn.commit()

def load_conversations():
    init_db()
    with sqlite3.connect(LOG_DB) as conn:
        conn.row_factory = sqlite3.Row
        return [
            dict(x)
            for x in conn.execute(
                "SELECT * FROM conversations ORDER BY id DESC"
            ).fetchall()
        ]
