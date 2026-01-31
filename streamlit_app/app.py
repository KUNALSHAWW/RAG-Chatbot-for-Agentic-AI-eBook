"""
Streamlit App for RAG Chatbot - Agentic AI eBook

This is the main UI for the RAG chatbot. It provides:
- Chat interface for asking questions
- Configuration sidebar (API keys, top_k, etc.)
- Display of retrieved chunks and confidence scores
- Raw JSON response viewer

Usage:
    streamlit run streamlit_app/app.py

For Hugging Face Spaces deployment:
    - Set secrets in Space settings for PINECONE_API_KEY, OPENAI_API_KEY
    - Or let users input keys in the sidebar
"""

import os
import sys
import json
import streamlit as st
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from the project root .env file
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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Answer card styling */
    .answer-card {
        background-color: #f0f7ff;
        border-left: 4px solid #1E88E5;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    /* Confidence badge styling */
    .confidence-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-weight: bold;
        font-size: 0.9rem;
    }
    
    .confidence-high {
        background-color: #c8e6c9;
        color: #2e7d32;
    }
    
    .confidence-medium {
        background-color: #fff3e0;
        color: #ef6c00;
    }
    
    .confidence-low {
        background-color: #ffcdd2;
        color: #c62828;
    }
    
    /* Chunk card styling */
    .chunk-card {
        background-color: #fafafa;
        border: 1px solid #e0e0e0;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.8rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Session State Initialization
# ============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "last_response" not in st.session_state:
    st.session_state.last_response = None


# ============================================================================
# Sidebar Configuration
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.markdown("---")
    
    # API Keys section
    st.subheader("🔑 API Keys")
    
    # Pinecone API Key
    pinecone_key = st.text_input(
        "Pinecone API Key",
        type="password",
        value=os.getenv("PINECONE_API_KEY", ""),
        help="Required for vector search. Get your key at pinecone.io"
    )
    
    # Pinecone Index Name
    index_name = st.text_input(
        "Pinecone Index Name",
        value=os.getenv("PINECONE_INDEX", "agentic-ai-ebook"),
        help="Name of your Pinecone index"
    )
    
    # OpenAI API Key (optional)
    openai_key = st.text_input(
        "OpenAI API Key (optional)",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="For LLM-powered answers. Leave empty if using Groq."
    )
    
    # Groq API Key (optional - FREE!)
    groq_key = st.text_input(
        "Groq API Key (FREE LLM)",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Free LLM alternative! Get key at console.groq.com"
    )
    
    st.markdown("---")
    
    # Retrieval settings
    st.subheader("🔍 Retrieval Settings")
    
    top_k = st.slider(
        "Number of chunks to retrieve (top_k)",
        min_value=1,
        max_value=10,
        value=6,
        help="More chunks = more context but potentially more noise"
    )
    
    use_llm = st.checkbox(
        "Use LLM for answer generation",
        value=True,
        help="Uncheck to always use extractive mode"
    )
    
    local_mode = st.checkbox(
        "Local Mode (no Pinecone)",
        value=False,
        help="Use local vector storage instead of Pinecone"
    )
    
    st.markdown("---")
    
    # Initialize/Reinitialize button
    if st.button("🔄 Initialize Pipeline", use_container_width=True):
        with st.spinner("Initializing RAG pipeline..."):
            try:
                st.session_state.pipeline = RAGPipeline(
                    pinecone_api_key=pinecone_key if pinecone_key else None,
                    openai_api_key=openai_key if openai_key else None,
                    groq_api_key=groq_key if groq_key else None,
                    index_name=index_name,
                    local_only=local_mode,
                    top_k=top_k
                )
                st.success("✅ Pipeline initialized!")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Status indicator
    st.markdown("---")
    st.subheader("📊 Status")
    
    if st.session_state.pipeline:
        st.success("Pipeline: Ready")
        if st.session_state.pipeline.groq_client:
            st.info("Mode: Groq LLM (FREE)")
        elif st.session_state.pipeline.openai_client:
            st.info("Mode: OpenAI LLM")
        else:
            st.warning("Mode: Extractive (no LLM)")
    else:
        st.warning("Pipeline: Not initialized")
        st.caption("Click 'Initialize Pipeline' to start")


# ============================================================================
# Main Content Area
# ============================================================================

# Header
st.markdown('<div class="main-header">🤖 Agentic AI eBook Chatbot</div>', unsafe_allow_html=True)

st.markdown("""
<p style="text-align: center; color: #666;">
    Ask questions about the Agentic AI eBook. Answers are strictly grounded in the document.
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# Check if pipeline is initialized
if not st.session_state.pipeline:
    st.info("👈 Please configure your API keys and click 'Initialize Pipeline' in the sidebar to start.")
    
    # Show sample queries
    st.subheader("📝 Sample Questions to Try")
    sample_queries = [
        "What is the definition of 'agentic AI' described in the eBook?",
        "List the three risks of agentic systems the eBook mentions.",
        "What are the recommended safeguards for deploying agentic AI?",
        "How does the eBook distinguish between autonomous agents and traditional automation?",
        "What future research directions does the eBook propose?"
    ]
    
    for query in sample_queries:
        st.markdown(f"- {query}")

else:
    # Chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat")
        
        # Display chat history
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
        
        # Chat input
        user_input = st.chat_input("Ask a question about the Agentic AI eBook...")
        
        if user_input:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # Get response from pipeline
            with st.chat_message("assistant"):
                with st.spinner("Searching document and generating answer..."):
                    try:
                        response = st.session_state.pipeline.query(
                            user_input,
                            top_k=top_k,
                            use_llm=use_llm
                        )
                        
                        # Store response for display
                        st.session_state.last_response = response
                        
                        # Display answer
                        answer = response.get("final_answer", "No answer generated")
                        st.write(answer)
                        
                        # Display confidence
                        confidence = response.get("confidence", 0.0)
                        if confidence >= 0.7:
                            conf_class = "confidence-high"
                        elif confidence >= 0.4:
                            conf_class = "confidence-medium"
                        else:
                            conf_class = "confidence-low"
                        
                        st.markdown(
                            f'<span class="confidence-badge {conf_class}">Confidence: {confidence:.3f}</span>',
                            unsafe_allow_html=True
                        )
                        
                        # Add assistant message to chat
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer
                        })
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Error: {str(e)}"
                        })
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_response = None
            st.rerun()
    
    with col2:
        st.subheader("📚 Retrieved Chunks")
        
        if st.session_state.last_response:
            response = st.session_state.last_response
            chunks = response.get("retrieved_chunks", [])
            
            if chunks:
                for i, chunk in enumerate(chunks):
                    with st.expander(
                        f"Chunk {i+1} (Page {chunk.get('page', '?')}, Score: {chunk.get('score', 0):.3f})",
                        expanded=(i == 0)
                    ):
                        st.markdown(f"**ID:** `{chunk.get('id', 'unknown')}`")
                        st.markdown(f"**Page:** {chunk.get('page', 'unknown')}")
                        st.markdown(f"**Relevance Score:** {chunk.get('score', 0):.4f}")
                        st.markdown("**Text:**")
                        st.text_area(
                            "Chunk text",
                            value=chunk.get("text", ""),
                            height=150,
                            label_visibility="collapsed",
                            key=f"chunk_{i}"
                        )
            else:
                st.info("No chunks retrieved yet. Ask a question!")
            
            # Raw JSON viewer
            st.markdown("---")
            with st.expander("🔍 Show Raw JSON Response"):
                st.json(response)
        else:
            st.info("Ask a question to see retrieved chunks.")


# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown("""
<div class="footer">
    <p>
        <strong>Built for AI Engineer Intern Assignment</strong><br>
        Answers are strictly grounded in the Agentic AI eBook.<br>
        Using: LangGraph • Pinecone • Sentence-Transformers • Streamlit
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# Auto-initialize if env vars are set
# ============================================================================

# Try to auto-initialize on first load if env vars are present
if st.session_state.pipeline is None:
    env_pinecone = os.getenv("PINECONE_API_KEY")
    env_groq = os.getenv("GROQ_API_KEY")
    if env_pinecone:
        try:
            st.session_state.pipeline = RAGPipeline(
                pinecone_api_key=env_pinecone,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                groq_api_key=env_groq,
                index_name=os.getenv("PINECONE_INDEX", "agentic-ai-ebook"),
                local_only=False
            )
            # Debug: show which LLM is being used
            if st.session_state.pipeline.groq_client:
                st.sidebar.success("✅ Groq LLM connected!")
            elif st.session_state.pipeline.openai_client:
                st.sidebar.info("ℹ️ OpenAI LLM connected")
            else:
                st.sidebar.warning("⚠️ No LLM - using extractive mode")
        except Exception as e:
            st.sidebar.error(f"Auto-init failed: {e}")
