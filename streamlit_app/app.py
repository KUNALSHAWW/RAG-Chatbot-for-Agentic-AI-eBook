"""
Streamlit App for RAG Chatbot - Agentic AI eBook
Modern ChatGPT/Gemini-style UI

Usage:
    streamlit run streamlit_app/app.py
"""

import os
import sys
import json
import streamlit as st
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(env_path)

# Import RAG pipeline
from app.rag_pipeline import RAGPipeline


# ============================================================================
# Helper to get secrets securely
# ============================================================================

def get_secret(key: str, default: str = "") -> str:
    """
    Get secret from Streamlit secrets first, then env vars.
    This ensures API keys are never displayed openly.
    """
    # Try Streamlit secrets first (for deployed apps)
    try:
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    # Fall back to environment variables
    return os.getenv(key, default)


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Agentic AI eBook Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Modern CSS styling
st.markdown("""
<style>
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Header styling */
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Chat message styling */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
        word-wrap: break-word;
    }
    
    .assistant-message {
        background: #f0f2f6;
        color: #1a1a2e;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 0.5rem 0;
        max-width: 90%;
        border: 1px solid #e0e0e0;
        word-wrap: break-word;
    }
    
    /* Confidence badge */
    .confidence-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    .conf-high {
        background: #d4edda;
        color: #155724;
    }
    
    .conf-medium {
        background: #fff3cd;
        color: #856404;
    }
    
    .conf-low {
        background: #f8d7da;
        color: #721c24;
    }
    
    /* Source chips */
    .source-chip {
        display: inline-block;
        background: #e9ecef;
        color: #495057;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.7rem;
        margin: 0.1rem;
    }
    
    /* Welcome container */
    .welcome-box {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-radius: 20px;
        margin: 2rem 0;
    }
    
    /* Sample question buttons */
    .stButton > button {
        border-radius: 20px !important;
        border: 1px solid #667eea !important;
        background: white !important;
        color: #667eea !important;
        font-size: 0.85rem !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background: #667eea !important;
        color: white !important;
    }
    
    /* Status indicator */
    .status-ready {
        color: #28a745;
        font-weight: 600;
    }
    
    .status-not-ready {
        color: #dc3545;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Session State
# ============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "last_response" not in st.session_state:
    st.session_state.last_response = None

if "processing" not in st.session_state:
    st.session_state.processing = False

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# Store API keys in session state (hidden from UI once entered)
if "api_keys_configured" not in st.session_state:
    st.session_state.api_keys_configured = False

if "stored_pinecone_key" not in st.session_state:
    st.session_state.stored_pinecone_key = get_secret("PINECONE_API_KEY", "")

if "stored_index_name" not in st.session_state:
    st.session_state.stored_index_name = get_secret("PINECONE_INDEX", "agentic-ai-ebook")

if "stored_groq_key" not in st.session_state:
    st.session_state.stored_groq_key = get_secret("GROQ_API_KEY", "")

if "stored_openai_key" not in st.session_state:
    st.session_state.stored_openai_key = get_secret("OPENAI_API_KEY", "")


# ============================================================================
# Helper Functions
# ============================================================================

def initialize_pipeline(pinecone_key, index_name, openai_key, groq_key, local_mode):
    """Initialize the RAG pipeline with given credentials."""
    try:
        pipeline = RAGPipeline(
            pinecone_api_key=pinecone_key if not local_mode else None,
            index_name=index_name,
            namespace="agentic-ai",
            openai_api_key=openai_key if openai_key else None,
            groq_api_key=groq_key if groq_key else None,
            local_only=local_mode
        )
        return pipeline, None
    except Exception as e:
        return None, str(e)


def get_confidence_class(confidence):
    """Get CSS class based on confidence score."""
    if confidence >= 0.7:
        return "conf-high"
    elif confidence >= 0.4:
        return "conf-medium"
    return "conf-low"


# ============================================================================
# Sidebar for Settings
# ============================================================================

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # Show configuration status
    if st.session_state.api_keys_configured:
        st.success("🔐 API Keys Configured")
        st.info("Keys are stored securely in session")
        
        if st.button("🔄 Reconfigure Keys", use_container_width=True):
            st.session_state.api_keys_configured = False
            st.session_state.pipeline = None
            st.rerun()
    else:
        st.markdown("### 🔑 Enter API Keys")
        st.caption("Keys are stored securely and hidden after configuration")
        
        pinecone_key = st.text_input(
            "Pinecone API Key",
            type="password",
            value="",
            key="pinecone_key_input",
            placeholder="Enter your Pinecone API key"
        )
        
        index_name = st.text_input(
            "Pinecone Index",
            value=st.session_state.stored_index_name,
            key="index_name_input"
        )
        
        groq_key = st.text_input(
            "Groq API Key (FREE)",
            type="password",
            value="",
            key="groq_key_input",
            help="Get free key at console.groq.com",
            placeholder="Enter your Groq API key"
        )
        
        openai_key = st.text_input(
            "OpenAI Key (optional)",
            type="password",
            value="",
            key="openai_key_input",
            placeholder="Optional - for OpenAI models"
        )
        
        if st.button("💾 Save & Configure", type="primary", use_container_width=True):
            # Store keys in session state (not displayed)
            if pinecone_key:
                st.session_state.stored_pinecone_key = pinecone_key
            if groq_key:
                st.session_state.stored_groq_key = groq_key
            if openai_key:
                st.session_state.stored_openai_key = openai_key
            st.session_state.stored_index_name = index_name
            st.session_state.api_keys_configured = True
            st.rerun()
    
    st.markdown("---")
    
    top_k = st.slider("Chunks to retrieve", 1, 10, 6, key="top_k")
    use_llm = st.checkbox("Use LLM", value=True, key="use_llm")
    local_mode = st.checkbox("Local Mode", value=False, key="local_mode")
    show_chunks = st.checkbox("Show retrieved chunks", value=False, key="show_chunks", 
                              help="Enable to see the top-k retrieved document chunks")
    
    st.markdown("---")
    
    if st.button("🚀 Initialize Pipeline", type="primary", use_container_width=True):
        with st.spinner("Initializing..."):
            pipeline, error = initialize_pipeline(
                st.session_state.stored_pinecone_key, 
                st.session_state.stored_index_name, 
                st.session_state.stored_openai_key, 
                st.session_state.stored_groq_key, 
                local_mode
            )
            if error:
                st.error(f"❌ {error}")
            else:
                st.session_state.pipeline = pipeline
                st.success("✅ Ready!")
    
    # Status
    if st.session_state.pipeline:
        st.markdown('<p class="status-ready">● Pipeline Ready</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-not-ready">● Not Initialized</p>', unsafe_allow_html=True)


# ============================================================================
# Auto-initialize if env vars/secrets are set
# ============================================================================

if st.session_state.pipeline is None and not st.session_state.processing:
    pk = st.session_state.stored_pinecone_key
    gk = st.session_state.stored_groq_key
    
    if pk and gk:
        pipeline, _ = initialize_pipeline(
            pk,
            st.session_state.stored_index_name,
            st.session_state.stored_openai_key,
            gk,
            False
        )
        if pipeline:
            st.session_state.pipeline = pipeline
            st.session_state.api_keys_configured = True


# ============================================================================
# Main UI
# ============================================================================

# Header
st.markdown('<h1 class="main-title">🤖 Agentic AI Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Ask questions about the Agentic AI eBook • Grounded answers only</p>', unsafe_allow_html=True)


# ============================================================================
# Chat Display
# ============================================================================

# Welcome screen if no messages
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-box">
        <h2>👋 Welcome!</h2>
        <p style="color: #666; max-width: 500px; margin: 0 auto 1rem auto;">
            I'm your AI assistant for the Agentic AI eBook. 
            Ask me anything about the document and I'll find relevant answers.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 💡 Try these questions:")
    
    sample_questions = [
        "What is the definition of agentic AI?",
        "What are the key characteristics of agentic systems?",
        "What risks does the eBook mention?",
        "What safeguards are recommended?"
    ]
    
    cols = st.columns(2)
    for i, q in enumerate(sample_questions):
        with cols[i % 2]:
            if st.button(q, key=f"sample_{i}", use_container_width=True):
                if st.session_state.pipeline:
                    st.session_state.messages.append({"role": "user", "content": q})
                    st.rerun()
                else:
                    st.warning("Please initialize the pipeline first (click sidebar)")

else:
    # Display all messages
    for idx, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-end; margin: 1rem 0;">
                <div class="user-message">{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            content = message["content"]
            confidence = message.get("confidence", 0)
            sources = message.get("sources", [])
            chunks = message.get("chunks", [])
            
            conf_class = get_confidence_class(confidence)
            
            # Build sources HTML
            sources_html = ""
            if sources:
                chips = " ".join([f'<span class="source-chip">📄 Page {s}</span>' for s in sources[:5]])
                sources_html = f'<div style="margin-top: 0.5rem;">{chips}</div>'
            
            st.markdown(f"""
            <div style="margin: 1rem 0;">
                <div class="assistant-message">
                    <div style="white-space: pre-wrap;">{content}</div>
                    <div style="margin-top: 0.75rem;">
                        <span class="confidence-badge {conf_class}">
                            Confidence: {confidence:.0%}
                        </span>
                    </div>
                    {sources_html}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show retrieved chunks if enabled and chunks exist
            if st.session_state.get("show_chunks", False) and chunks:
                with st.expander(f"📚 View Top {len(chunks)} Retrieved Chunks", expanded=False):
                    for i, chunk in enumerate(chunks, 1):
                        chunk_text = chunk.get("text", "")[:500] + "..." if len(chunk.get("text", "")) > 500 else chunk.get("text", "")
                        chunk_page = chunk.get("page", "?")
                        chunk_score = chunk.get("score", 0)
                        st.markdown(f"""
                        **Chunk {i}** (Page {chunk_page}, Score: {chunk_score:.3f})
                        > {chunk_text}
                        ---
                        """)
    
    # Clear chat button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_response = None
            st.session_state.processing = False
            st.session_state.pending_query = None
            st.rerun()


# ============================================================================
# Chat Input (MUST be at top level, not in any container)
# ============================================================================

user_input = st.chat_input("Ask a question about the Agentic AI eBook...")

# Handle new user input
if user_input and not st.session_state.processing:
    if st.session_state.pipeline is None:
        st.warning("⚠️ Please initialize the pipeline first (open sidebar → click Initialize)")
    else:
        # Set processing flag to prevent duplicate queries
        st.session_state.processing = True
        st.session_state.pending_query = user_input
        st.rerun()

# Process pending query (runs after rerun to ensure single execution)
if st.session_state.processing and st.session_state.pending_query:
    query = st.session_state.pending_query
    
    # Add user message only if not already added (fix: use 'and' logic for proper deduplication)
    already_added = (
        st.session_state.messages 
        and st.session_state.messages[-1].get("role") == "user" 
        and st.session_state.messages[-1].get("content") == query
    )
    if not already_added:
        st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user message immediately
    st.markdown(f"""
    <div style="display: flex; justify-content: flex-end; margin: 1rem 0;">
        <div class="user-message">{query}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Get response
    try:
        with st.spinner("🔍 Searching document and generating answer..."):
            response = st.session_state.pipeline.query(
                query,
                top_k=st.session_state.get("top_k", 6),
                use_llm=st.session_state.get("use_llm", True)
            )
            
            # Extract data - single best answer
            answer = response.get("final_answer", "I couldn't find an answer.")
            confidence = response.get("confidence", 0.0)
            chunks = response.get("retrieved_chunks", [])
            
            # Get source pages
            sources = list(set([c.get("page", "?") for c in chunks]))
            sources.sort()
            
            # Add assistant message only if there's no pending assistant response for this query
            # Check if last message is already an assistant response (prevent duplicates)
            last_msg = st.session_state.messages[-1] if st.session_state.messages else None
            if not last_msg or last_msg.get("role") != "assistant":
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "confidence": confidence,
                    "sources": sources,
                    "chunks": chunks  # Store chunks for optional viewing
                })
            
            st.session_state.last_response = response
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        # Only add error message if not already an assistant response
        last_msg = st.session_state.messages[-1] if st.session_state.messages else None
        if not last_msg or last_msg.get("role") != "assistant":
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Sorry, an error occurred: {str(e)}",
                "confidence": 0,
                "sources": [],
                "chunks": []
            })
    
    # Clear processing state
    st.session_state.processing = False
    st.session_state.pending_query = None
    st.rerun()


# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.8rem; padding-bottom: 2rem;">
    Built with LangGraph • Pinecone • Groq • Streamlit<br>
    <em>Answers are strictly grounded in the Agentic AI eBook</em>
</div>
""", unsafe_allow_html=True)
