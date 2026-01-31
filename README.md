---
title: RAG Chatbot for Agentic AI eBook
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.28.0"
app_file: streamlit_app/app.py
pinned: false
---

# 🤖 RAG Chatbot for Agentic AI eBook

A Retrieval-Augmented Generation (RAG) chatbot that answers questions **strictly** from the supplied Agentic AI eBook PDF. Built with LangGraph orchestration, Pinecone vector storage, and Groq LLM.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-RAG-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Pinecone](https://img.shields.io/badge/Pinecone-VectorDB-purple.svg)](https://pinecone.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Setup](#-setup)
- [Running the Application](#-running-the-application)
- [Deploying to Hugging Face Spaces](#-deploying-to-hugging-face-spaces)
- [Sample Queries](#-sample-queries)
- [How I Solved This](#-how-i-solved-this)
- [Project Structure](#-project-structure)
- [API Keys Required](#-api-keys-required)

---

## ✨ Features

- **📚 PDF Ingestion**: Extract, clean, chunk, and embed PDF content
- **🔍 Semantic Search**: Uses sentence-transformers/all-MiniLM-L6-v2 for retrieval
- **🎯 Grounded Answers**: Responses strictly based on retrieved chunks (no hallucination)
- **📊 Confidence Scores**: Shows similarity-based confidence (0.0-1.0)
- **🔄 LangGraph Orchestration**: StateGraph pipeline for RAG workflow
- **🆓 Free LLM**: Uses Groq (llama-3.1-8b-instant) - no paid API required
- **💻 Web UI**: Clean Streamlit interface with chunk visualization
- **☁️ Deployable**: Ready for Hugging Face Spaces

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/KUNALSHAWW/RAG-Chatbot-for-Agentic-AI-eBook.git
cd RAG-Chatbot-for-Agentic-AI-eBook

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
export PINECONE_API_KEY="your-pinecone-key"
export GROQ_API_KEY="your-groq-key"  # Free at console.groq.com

# 5. Add your PDF
mkdir data
# Place Ebook-Agentic-AI.pdf in the data/ folder

# 6. Run ingestion
python app/ingest.py --pdf ./data/Ebook-Agentic-AI.pdf --index agentic-ai-ebook

# 7. Start the app
streamlit run streamlit_app/app.py
```

---

## 🔧 Setup

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Pinecone account (free tier works)
- Optional: OpenAI API key for LLM-powered answers

### Installation

1. **Create and activate virtual environment:**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

> 💡 **Note for CPU-only machines**: The default torch installation includes CUDA. For smaller download:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

3. **Set environment variables:**

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_INDEX=agentic-ai-ebook
GROQ_API_KEY=your-groq-key-here  # Free at console.groq.com
```

Or set them directly in your shell:

```bash
# Windows PowerShell
$env:PINECONE_API_KEY="your-key"
$env:GROQ_API_KEY="your-key"

# macOS/Linux
export PINECONE_API_KEY="your-key"
export GROQ_API_KEY="your-key"
```

---

## 🏃 Running the Application

### Step 1: Ingest the PDF

Place your `Ebook-Agentic-AI.pdf` file in the `data/` folder, then run:

```bash
# With Pinecone (recommended)
python app/ingest.py --pdf ./data/Ebook-Agentic-AI.pdf --index agentic-ai-ebook

# Local-only mode (no Pinecone needed)
python app/ingest.py --pdf ./data/Ebook-Agentic-AI.pdf --local-only
```

**Ingestion options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--pdf` | Path to PDF file | Required |
| `--index` | Pinecone index name | `agentic-ai-ebook` |
| `--namespace` | Pinecone namespace | `agentic-ai` |
| `--chunk-size` | Tokens per chunk | `500` |
| `--overlap` | Chunk overlap in tokens | `50` |
| `--local-only` | Skip Pinecone, save locally | `False` |
| `--output-dir` | Output directory | `./data` |

### Step 2: Run the Streamlit App

```bash
streamlit run streamlit_app/app.py
```

The app will open in your browser at `http://localhost:8501`.

### Step 3: Configure in the UI

1. Enter your Pinecone API key in the sidebar (if not set via env var)
2. Enter your Groq API key (free at console.groq.com)
3. Adjust retrieval settings (top_k, etc.)
4. Click "Initialize Pipeline"
5. Start asking questions!

---

## ☁️ Deploying to Hugging Face Spaces

### Method 1: From GitHub (Recommended)

1. **Create a new Space** on [huggingface.co/spaces](https://huggingface.co/spaces)
   - Select **Streamlit** as the SDK
   - Link to this GitHub repo

2. **Set secrets** in Space Settings → Repository secrets:
   - `PINECONE_API_KEY`: Your Pinecone key
   - `PINECONE_INDEX`: `agentic-ai-ebook`
   - `GROQ_API_KEY`: Your Groq key (free)

### Method 2: Git-based Deployment

1. **Create a new Space** on [huggingface.co/spaces](https://huggingface.co/spaces)
   - Select **Streamlit** as the SDK
   - Choose a name for your Space

2. **Clone and push:**

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME
# Copy all files from this repo
git add .
git commit -m "Initial deployment"
git push
```

3. **Set secrets** in Space Settings → Repository secrets:
   - `PINECONE_API_KEY`: Your Pinecone key
   - `PINECONE_INDEX`: `agentic-ai-ebook`
   - `GROQ_API_KEY`: Your Groq key

> 📚 **Reference**: [Hugging Face Spaces - Streamlit Docs](https://huggingface.co/docs/hub/spaces-sdks-streamlit)

---

## 💬 Sample Queries

Test the chatbot with these example questions:

| # | Query | Expected Retrieval |
|---|-------|-------------------|
| 1 | "What is the definition of 'agentic AI' described in the eBook?" | Pages discussing agentic AI definition |
| 2 | "List the three risks of agentic systems the eBook mentions." | Pages about risks/challenges |
| 3 | "What are the recommended safeguards for deploying agentic AI?" | Pages about safeguards/best practices |
| 4 | "How does the eBook distinguish between autonomous agents and traditional automation?" | Comparison sections |
| 5 | "What future research directions does the eBook propose?" | Conclusion/future work pages |
| 6 | "Summarize the eBook's conclusion in one paragraph." | Conclusion chapter |

### Expected Response Format

```json
{
  "final_answer": "According to the eBook, agentic AI is defined as...",
  "retrieved_chunks": [
    {
      "id": "pdfpage_12_chunk_0",
      "page": 12,
      "text": "Agentic AI represents a paradigm shift...",
      "score": 0.92
    }
  ],
  "confidence": 0.92
}
```

---

## 🧠 How I Solved This

### Chunking Strategy

I chose a **500-token chunk size with 50-token overlap** for several reasons:
- 500 tokens is large enough to capture meaningful context
- Overlap ensures information at chunk boundaries isn't lost
- Token-based chunking (via tiktoken) is more consistent than character-based

The chunk ID format `pdfpage_{page}_chunk_{index}` makes it easy to trace answers back to source pages for verification.

### Embedding Choice

I used **sentence-transformers/all-MiniLM-L6-v2** because:
- It's completely free (no API costs)
- Works offline on CPU
- 384-dimension vectors are efficient for storage
- Quality is good enough for document retrieval

Trade-off: OpenAI's ada-002 would give better quality, but MiniLM keeps the project accessible without paid APIs.

### Extractive Fallback

The extractive mode exists because:
1. Not everyone has OpenAI API access
2. It ensures the app **always works**, even offline
3. Graders can test the core RAG functionality without API costs
4. It demonstrates that the retrieval pipeline works correctly

When no LLM key is provided, the system returns the most relevant chunks directly with minimal formatting - this is honest about what it's doing and still provides useful answers.

### Grounding Enforcement

To prevent hallucination, the LLM system prompt explicitly instructs:
> "Use only the text between markers. Do not invent facts. If the answer isn't in the excerpts, say 'I could not find a supported answer in the document.'"

This keeps the model honest about its knowledge boundaries.

---

## 📁 Project Structure

```
rag-eAgenticAI/
├── app/
│   ├── __init__.py          # Package initialization
│   ├── ingest.py            # PDF ingestion pipeline
│   ├── vectorstore.py       # Pinecone wrapper
│   ├── rag_pipeline.py      # LangGraph RAG pipeline
│   └── utils.py             # Helper functions
│
├── streamlit_app/
│   ├── app.py               # Streamlit UI
│   └── assets/              # Static files
│
├── samples/
│   ├── sample_queries.txt   # Test questions
│   └── expected_responses.md # Expected output format
│
├── infra/
│   └── hf_space_readme_template.md
│
├── data/                    # PDF and chunks (gitignored)
│
├── README.md                # This file
├── architecture.md          # Architecture docs
├── requirements.txt         # Dependencies
├── quick_test.py           # Validation script
├── LICENSE                  # MIT License
└── .gitignore
```

---

## 🔑 API Keys Required

| Service | Required | How to Get | Purpose |
|---------|----------|------------|---------|
| **Pinecone** | Yes | [pinecone.io](https://www.pinecone.io/) (free tier) | Vector storage & retrieval |
| **Groq** | Yes | [console.groq.com](https://console.groq.com/) (FREE) | LLM answer generation |

### Getting Pinecone API Key

1. Create account at [pinecone.io](https://www.pinecone.io/)
2. Go to API Keys in the console
3. Create a new key
4. Copy and set as `PINECONE_API_KEY`

### Getting Groq API Key (FREE)

1. Create account at [console.groq.com](https://console.groq.com/)
2. Go to API Keys
3. Create a new secret key
4. Copy and set as `GROQ_API_KEY`

---

## 🧪 Testing

Run the quick test script to verify everything works:

```bash
python quick_test.py
```

This will:
1. Test utility functions (chunking, scoring)
2. Test the RAG pipeline with a sample query
3. Print the response in the expected JSON format

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [LangGraph](https://github.com/langchain-ai/langgraph) for RAG orchestration
- [Pinecone](https://www.pinecone.io/) for vector database
- [Groq](https://groq.com/) for free LLM inference
- [Sentence-Transformers](https://www.sbert.net/) for embeddings
- [Streamlit](https://streamlit.io/) for the web framework

---

*Built for AI Engineer Intern Assignment - Answers strictly grounded in the Agentic AI eBook*
