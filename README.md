# 💬 Autonomous Customer Support Chatbot & Escalation Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-121212.svg)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-6A0DAD.svg)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama%20(100%25%20Local)-000000.svg)](https://ollama.ai/)

An end-to-end, **100% local and privacy-focused** AI Customer Support Chatbot featuring Retrieval-Augmented Generation (RAG), intent classification, real-time sentiment analysis, automated human agent escalation, and a real-time admin analytics dashboard.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Quick Start & Installation](#-quick-start--installation)
- [Configuration](#-configuration)
- [Application Features](#-application-features)
  - [1. Customer Chat Interface](#1-customer-chat-interface)
  - [2. Human Handoff & Escalation Logic](#2-human-handoff--escalation-logic)
  - [3. Admin Analytics Dashboard](#3-admin-analytics-dashboard)
- [Customizing Knowledge Base](#-customizing-knowledge-base)
- [Troubleshooting & FAQ](#-troubleshooting--faq)

---

## 🔒 Overview

Modern customer support systems demand speed, accuracy, and customer empathy—without exposing sensitive company or customer data to third-party APIs. 

This project delivers a **production-ready architecture running entirely on-premise/locally** via [Ollama](https://ollama.ai/). By integrating **Qwen2.5 (1.5B)** for structured reasoning/generation and **Nomic Embed Text** for semantic vector search, it provides instant FAQ resolution while identifying frustrated customers and automatically routing them to human support representatives with context-rich summaries.

---

## ✨ Key Features

- **🔒 100% Privacy & Local Execution**: Zero cloud dependencies or external API costs; runs completely offline on your infrastructure.
- **📚 Smart RAG Engine**: Leverages ChromaDB and LangChain to perform vector similarity search over support documentation (`data/faq.json`).
- **🎯 Structured Intent Recognition**: Classifies customer queries across 12 support categories (e.g., `order_status`, `refund`, `shipping`, `complaint`, `human_agent`) using Pydantic schema validation.
- **😠 Real-Time Sentiment Detection**: Dynamically categorizes customer emotion (`positive`, `neutral`, `negative`, `frustrated`, `urgent`) to gauge satisfaction.
- **🚨 Automated Escalation System**: Evaluates sentiment, intent, and failed resolution attempts to gracefully transition conversations to human support reps.
- **📝 Automated Handoff Summaries**: Generates concise, structured summaries of the interaction so human agents never ask customers to repeat themselves.
- **📊 Admin Analytics Dashboard**: Built-in monitoring tab tracks total conversations, escalation rates, common intent distributions, and detailed conversation logs stored in SQLite.

---

## 🏗️ System Architecture

```text
                    ┌───────────────────┐
                    │  Customer User    │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Streamlit Chat UI │
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
          ┌──────────────────┐  ┌──────────────────┐
          │ Intent Classifier│  │Sentiment Analyzer│
          │ Pydantic+Ollama  │  │ Pydantic+Ollama  │
          └────────┬─────────┘  └────────┬─────────┘
                   │                     │
                   └──────────┬──────────┘
                              ▼
                    ┌───────────────────┐
                    │ Escalation Router │
                    └─────────┬─────────┘
                              │
                 ┌────────────┴────────────┐
                 │                         │
        Standard Query             High Frustration /
                 │                    Urgent / Failed
                 ▼                         ▼
        ┌─────────────────┐       ┌───────────────────┐
        │   RAG Engine    │       │ Handoff Summarizer│
        └────────┬────────┘       └─────────┬─────────┘
                 │                          │
                 ▼                          ▼
        ┌─────────────────┐       ┌───────────────────┐
        │    ChromaDB     │       │ Human Agent       │
        │  Vector Store   │       │    Dashboard      │
        └────────┬────────┘       └───────────────────┘
                 │
                 ▼
        ┌─────────────────┐
        │Response Generator│
        │    Ollama LLM   │
        └────────┬────────┘
                 │
                 ▼
        ┌───────────────────┐
        │ Streamlit Chat UI │
        │     (Reply)       │
        └───────────────────┘

              ┌─────────────────┐
              │   SQLite DB     │
              │  Audit / Logs   │
              └────────┬────────┘
                       ▼
              ┌───────────────────┐
              │ Admin Analytics   │
              │    Dashboard      │
              └───────────────────┘
```

---

## 📁 Project Structure

```text
customer-support-chatbot/
├── admin/
│   └── dashboard.py          # Admin analytics dashboard & conversation viewer
├── chatbot/
│   ├── escalation.py         # Human escalation & handoff summary generation
│   ├── intent.py             # Intent classification logic (Pydantic schema)
│   ├── response.py           # RAG chain generator using Ollama
│   ├── sentiment.py          # Sentiment analysis engine
│   └── state.py              # Session state initialization & message handlers
├── data/
│   └── faq.json              # Knowledge base data source (FAQs)
├── knowledge_base/
│   └── retriever.py          # ChromaDB vectorstore initialization & similarity search
├── logs/
│   └── support.db            # SQLite database storing conversation histories & metrics
├── utils/
│   └── logger.py             # Database initialization & logging utilities
├── vectorstore/
│   └── chroma/               # Local persistent Chroma vector database directory
├── app.py                    # Main Streamlit application entry point
├── config.py                 # Centralized configuration & environment loader
├── .env.example              # Template for environment variables
├── requirements.txt          # Python dependency specifications
└── README.md                 # Project documentation
```

---

## 📋 Prerequisites

Before running the application, ensure you have the following installed on your system:

1. **Python**: Version `3.11` 
2. **Ollama**: Download and install from [ollama.ai](https://ollama.ai/).

Ensure the Ollama service is running locally, then pull the required models:

```bash
# Pull the chat LLM (Qwen 2.5 1.5B)
ollama pull qwen2.5:1.5b

# Pull the embedding model (Nomic Embed Text)
ollama pull nomic-embed-text
```

---

## 🚀 Quick Start & Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Rehena-Sultana/customer-support-chatbot.git
cd customer-support-chatbot
```

### Step 2: Create & Activate Virtual Environment

**On Linux/macOS:**
```bash
python3 -m venv myenv
source myenv/bin/activate
```

**On Windows (PowerShell):**
```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Environment Configuration

Create a `.env` file from the provided example template:

```bash
cp .env.example .env
```

Default configuration in `.env`:
```env
OLLAMA_CHAT_MODEL=qwen2.5:1.5b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_BASE_URL=http://localhost:11434
```

### Step 5: Launch the Application

Run the Streamlit application:

```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`.

---

## ⚙️ Configuration

All configuration variables are managed in [`config.py`](file:///d:/github_repositories/customer-support-chatbot/config.py) and can be customized via environment variables:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Base endpoint URL for the local Ollama service. |
| `OLLAMA_CHAT_MODEL` | `qwen2.5:1.5b` | Model used for intent, sentiment, RAG, and summaries. |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model used for Chroma vector indexing. |
| `CHROMA_DIR` | `vectorstore/chroma` | Directory for storing persistent vector embeddings. |
| `LOG_DB` | `logs/support.db` | Path to SQLite database for conversation logging. |

---

## 🎯 Application Features

### 1. Customer Chat Interface
- Clean, intuitive chat interface powered by Streamlit.
- Real-time response generation with background RAG querying.
- Expandable sidebar displaying active metadata (Intent, Sentiment, Route, Failed Attempts).

### 2. Human Handoff & Escalation Logic
Automatic escalation is triggered under any of the following conditions:
- **Direct Request**: Customer specifies intent for a `human_agent`.
- **Negative Sentiment**: Emotion is categorized as `frustrated` or `urgent`.
- **Repeated Failures**: Two or more consecutive failed retrieval attempts.
- **Sensitive Topics**: High-risk intents (`payment_issue`, `complaint`) paired with negative sentiment.

When triggered, an automated summary is compiled for human agents containing:
- Primary issue & customer intent
- Detected customer sentiment
- Summary of dialogue history
- Recommended next steps for the human representative

### 3. Admin Analytics Dashboard
Access the dashboard via the sidebar navigation (`Admin Dashboard`):
- **Key Metrics**: Total Conversations, Total Escalations, and Escalation Rate percentage.
- **Topic Analytics**: Bar chart visualizing common customer intent distributions.
- **Conversation Logs**: Interactive table of logged sessions with detailed handoff summaries and raw JSON message histories.

---

## 🛠️ Customizing Knowledge Base

To update the chatbot's knowledge base with your own company FAQs:

1. Edit or replace the [`data/faq.json`](file:///d:/github_repositories/customer-support-chatbot/data/faq.json) file following this JSON structure:

```json
[
  {
    "id": "faq_001",
    "category": "Shipping",
    "question": "How long does standard shipping take?",
    "answer": "Standard shipping typically takes 3-5 business days."
  }
]
```

2. Reset the Chroma vector store cache so the new documents are indexed:
   - Remove the contents of `vectorstore/chroma/` (or delete the folder).
   - Restart the app (`streamlit run app.py`). The vectorstore will automatically rebuild on startup.

---
