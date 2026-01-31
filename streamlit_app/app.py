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
    
    pinecone_key = st.text_input(
        "Pinecone API Key",
        type="password",
        value=os.getenv("PINECONE_API_KEY", ""),
        key="pinecone_key"
    )
    
    index_name = st.text_input(
        "Pinecone Index",
        value=os.getenv("PINECONE_INDEX", "agentic-ai-ebook"),
        key="index_name"
    )
    
    groq_key = st.text_input(
        "Groq API Key (FREE)",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        key="groq_key",
        help="Get free key at console.groq.com"
    )
    
    openai_key = st.text_input(
        "OpenAI Key (optional)",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        key="openai_key"
    )
    
    st.markdown("---")
    
    top_k = st.slider("Chunks to retrieve", 1, 10, 6, key="top_k")
    use_llm = st.checkbox("Use LLM", value=True, key="use_llm")
    local_mode = st.checkbox("Local Mode", value=False, key="local_mode")
    
    st.markdown("---")
    
    if st.button("🚀 Initialize", type="primary", use_container_width=True):
        with st.spinner("Initializing..."):
            pipeline, error = initialize_pipeline(
                pinecone_key, index_name, openai_key, groq_key, local_mode
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
# Auto-initialize if env vars are set
# ============================================================================

if st.session_state.pipeline is None:
    pk = os.getenv("PINECONE_API_KEY", "")
    gk = os.getenv("GROQ_API_KEY", "")
    
    if pk and gk:
        pipeline, _ = initialize_pipeline(
            pk,
            os.getenv("PINECONE_INDEX", "agentic-ai-ebook"),
            os.getenv("OPENAI_API_KEY", ""),
            gk,
            False
        )
        if pipeline:
            st.session_state.pipeline = pipeline


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
    for message in st.session_state.messages:
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
    
    # Clear chat button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_response = None
            st.rerun()


# ============================================================================
# Chat Input (MUST be at top level, not in any container)
# ============================================================================

user_input = st.chat_input("Ask a question about the Agentic AI eBook...")

if user_input:
    if st.session_state.pipeline is None:
        st.warning("⚠️ Please initialize the pipeline first (open sidebar → click Initialize)")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get response
        try:
            with st.spinner("🔍 Searching document..."):
                response = st.session_state.pipeline.query(
                    user_input,
                    top_k=st.session_state.get("top_k", 6),
                    use_llm=st.session_state.get("use_llm", True)
                )
                
                # Extract data
                answer = response.get("final_answer", "I couldn't find an answer.")
                confidence = response.get("confidence", 0.0)
                chunks = response.get("retrieved_chunks", [])
                
                # Get source pages
                sources = list(set([c.get("page", "?") for c in chunks]))
                sources.sort()
                
                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "confidence": confidence,
                    "sources": sources
                })
                
                st.session_state.last_response = response
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Sorry, an error occurred: {str(e)}",
                "confidence": 0,
                "sources": []
            })
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
