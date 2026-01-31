---
title: Agentic AI eBook Chatbot
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.28.0"
app_file: streamlit_app/app.py
pinned: false
license: mit
---

# Agentic AI eBook RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions strictly from the Agentic AI eBook.

## Features
- 🔍 **Semantic Search**: Uses sentence-transformers for document retrieval
- 📚 **Grounded Answers**: All answers are strictly based on retrieved document chunks
- 📊 **Confidence Scores**: Shows how confident the system is in its answers
- 🔄 **Dual Mode**: LLM generation (with OpenAI key) or extractive fallback

## Setup

### Environment Variables (Set in Space Settings → Secrets)

| Variable | Required | Description |
|----------|----------|-------------|
| `PINECONE_API_KEY` | Yes | Your Pinecone API key |
| `PINECONE_INDEX` | No | Index name (default: `agentic-ai-ebook`) |
| `OPENAI_API_KEY` | No | For LLM-powered answers |

### Usage

1. Set your Pinecone API key in the sidebar
2. Optionally set OpenAI API key for better answers
3. Ask questions about the Agentic AI eBook!

## Tech Stack
- LangGraph for RAG orchestration
- Pinecone for vector storage
- Sentence-Transformers for embeddings
- Streamlit for UI

## Limitations
- Only answers questions from the Agentic AI eBook
- Requires pre-ingested document in Pinecone index
- May not answer questions outside the document scope

---

Built for AI Engineer Intern Assignment
