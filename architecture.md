# Architecture Overview

This document explains the architecture of the RAG (Retrieval-Augmented Generation) chatbot for the Agentic AI eBook.

## System Overview

The system follows a standard RAG pattern: documents are chunked and embedded into a vector database during ingestion, then at query time, relevant chunks are retrieved and used to generate grounded answers.

### Key Components

1. **Ingestion Pipeline** (`app/ingest.py`) - Processes the PDF, creates chunks, generates embeddings, and stores in Pinecone
2. **Vector Store** (`app/vectorstore.py`) - Wrapper around Pinecone for storing and retrieving vectors
3. **RAG Pipeline** (`app/rag_pipeline.py`) - LangGraph-based pipeline for query processing
4. **Streamlit UI** (`streamlit_app/app.py`) - Web interface for user interactions

---

## Architecture Diagram

```
                              INGESTION FLOW
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────────┐   │
│    │   PDF    │───▶│  Extract │───▶│  Clean   │───▶│     Chunk        │   │
│    │  File    │    │   Text   │    │   Text   │    │  (500 tokens,    │   │
│    │          │    │ by Page  │    │          │    │   50 overlap)    │   │
│    └──────────┘    └──────────┘    └──────────┘    └────────┬─────────┘   │
│                                                              │             │
│                                                              ▼             │
│    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│    │     Pinecone     │◀───│     Upsert       │◀───│   Embeddings     │   │
│    │   Vector Store   │    │    Vectors       │    │  (MiniLM-L6-v2)  │   │
│    │                  │    │                  │    │   384 dims       │   │
│    └──────────────────┘    └──────────────────┘    └──────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


                               QUERY FLOW
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│    ┌──────────┐                                                             │
│    │   User   │                                                             │
│    │  Query   │                                                             │
│    └────┬─────┘                                                             │
│         │                                                                   │
│         ▼                                                                   │
│    ┌──────────────────────────────────────────────────────────────────┐    │
│    │                    LANGGRAPH PIPELINE                             │    │
│    │                                                                   │    │
│    │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐            │    │
│    │  │   Embed     │──▶│  Retrieve   │──▶│  Calculate  │            │    │
│    │  │   Query     │   │   Top-K     │   │ Confidence  │            │    │
│    │  │             │   │  Chunks     │   │             │            │    │
│    │  └─────────────┘   └──────┬──────┘   └──────┬──────┘            │    │
│    │                           │                 │                    │    │
│    │                           ▼                 ▼                    │    │
│    │                    ┌─────────────────────────────┐               │    │
│    │                    │     Generate Answer         │               │    │
│    │                    │                             │               │    │
│    │                    │  ┌─────────────────────┐   │               │    │
│    │                    │  │ If OpenAI Key:      │   │               │    │
│    │                    │  │  → LLM Generation   │   │               │    │
│    │                    │  │  (grounded prompt)  │   │               │    │
│    │                    │  ├─────────────────────┤   │               │    │
│    │                    │  │ Else:               │   │               │    │
│    │                    │  │  → Extractive Mode  │   │               │    │
│    │                    │  │  (return chunks)    │   │               │    │
│    │                    │  └─────────────────────┘   │               │    │
│    │                    └─────────────┬───────────────┘               │    │
│    │                                  │                               │    │
│    └──────────────────────────────────┼───────────────────────────────┘    │
│                                       │                                     │
│                                       ▼                                     │
│    ┌──────────────────────────────────────────────────────────────────┐    │
│    │                         RESPONSE                                  │    │
│    │  {                                                                │    │
│    │    "final_answer": "...",                                        │    │
│    │    "retrieved_chunks": [...],                                    │    │
│    │    "confidence": 0.92                                            │    │
│    │  }                                                                │    │
│    └──────────────────────────────────────────────────────────────────┘    │
│                                       │                                     │
│                                       ▼                                     │
│    ┌──────────────────────────────────────────────────────────────────┐    │
│    │                      STREAMLIT UI                                 │    │
│    │  ┌──────────────────┐  ┌───────────────────────────────────┐    │    │
│    │  │  Chat Interface  │  │  Retrieved Chunks Panel           │    │    │
│    │  │  - Question box  │  │  - Chunk text                     │    │    │
│    │  │  - Answer card   │  │  - Page numbers                   │    │    │
│    │  │  - Confidence    │  │  - Relevance scores               │    │    │
│    │  └──────────────────┘  └───────────────────────────────────┘    │    │
│    └──────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Design Decisions

### 1. Chunking Strategy

We use **500 tokens** as the target chunk size with **50-100 token overlap**. This provides:
- Enough context for meaningful retrieval
- Overlap ensures important information spanning chunk boundaries isn't lost
- Token counting via tiktoken ensures consistent chunk sizes across different text densities

**Chunk ID Format**: `pdfpage_{page}_chunk_{index}` - This makes it easy to trace retrieved content back to the source PDF page for verification.

### 2. Embedding Model Choice

We use **sentence-transformers/all-MiniLM-L6-v2**:
- Open source and free (no API costs)
- Small model (384 dimensions) = fast inference and lower storage costs
- Good quality for semantic similarity tasks
- Can run entirely on CPU

Trade-off: Larger models like OpenAI's ada-002 (1536 dims) may provide better retrieval quality, but MiniLM offers excellent cost/performance ratio for this use case.

### 3. LangGraph Pipeline

The RAG pipeline uses LangGraph for orchestration because:
- Clear separation of pipeline stages (embed → retrieve → generate)
- Easy to add/modify nodes (e.g., reranking, query expansion)
- Built-in state management
- Aligns with modern LLM application patterns

### 4. Dual-Mode Answer Generation

The system supports two modes:

**LLM Generation Mode** (with OpenAI key):
- Uses GPT-3.5-turbo for natural language generation
- System prompt strictly instructs the model to only use provided chunks
- Produces more readable, synthesized answers

**Extractive Fallback Mode** (no API key):
- Returns relevant chunks directly with minimal formatting
- Always works, even offline
- Ensures the app is functional without paid APIs

This design choice ensures the application is **always functional** regardless of API availability.

### 5. Confidence Score Computation

Confidence is computed from retrieval similarity scores:

```python
# Normalize cosine similarity from [-1, 1] to [0, 1]
normalized = (score + 1) / 2

# Use maximum normalized score as confidence
confidence = max(normalized_scores)
```

This gives users an intuitive sense of how well the retrieved chunks match their query.

---

## File Structure

```
rag-eAgenticAI/
├── app/
│   ├── __init__.py          # Package exports
│   ├── ingest.py            # PDF → chunks → embeddings → Pinecone
│   ├── vectorstore.py       # Pinecone wrapper (create, upsert, query)
│   ├── rag_pipeline.py      # LangGraph pipeline + answer generation
│   └── utils.py             # Chunking, cleaning, confidence calculation
│
├── streamlit_app/
│   ├── app.py               # Main Streamlit application
│   └── assets/              # Static assets (images, CSS)
│
├── samples/
│   ├── sample_queries.txt   # Example questions to test
│   └── expected_responses.md # Expected JSON response format
│
├── infra/
│   └── hf_space_readme_template.md  # Hugging Face Spaces config
│
├── data/                    # PDF files and generated chunks (gitignored)
│
├── README.md                # Main documentation
├── architecture.md          # This file
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT License
└── .gitignore              # Git ignore rules
```

---

## Data Flow Summary

1. **Ingestion** (run once):
   - PDF → pdfplumber → raw text by page
   - Text → clean_text() → cleaned text
   - Cleaned text → chunk_text() → chunks with metadata
   - Chunks → SentenceTransformer → embeddings
   - Embeddings → Pinecone upsert → stored vectors

2. **Query** (each user question):
   - Question → SentenceTransformer → query embedding
   - Query embedding → Pinecone query → top-k chunks
   - Chunks + scores → compute_confidence() → confidence score
   - Chunks + question → LLM/extractive → final answer
   - Answer + chunks + confidence → JSON response → Streamlit UI
