"""
Streamlit App for RAG Chatbot - Agentic AI eBook
Modern ChatGPT-style UI with chat_input at root level
"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(env_path)

# Import RAG pipeline
from app.rag_pipeline import RAGPipeline

# Page config
st.set_page_config(
    page_title="Agentic AI eBook Chatbot",
    page_icon="🤖",
    layout="centered"
)

# CSS styling
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #888;
        margin-bottom: 1.5rem;
    }
    .conf-high { background: #d4edda; color: #155724; padding: 0.2rem 0.6rem; border-radius: 10px; font-size: 0.8rem; }
    .conf-medium { background: #fff3cd; color: #856404; padding: 0.2rem 0.6rem; border-radius: 10px; font-size: 0.8rem; }
    .conf-low { background: #f8d7da; color: #721c24; padding: 0.2rem 0.6rem; border-radius: 10px; font-size: 0.8rem; }
    .source-chip { background: #e9ecef; color: #495057; padding: 0.15rem 0.5rem; border-radius: 10px; font-size: 0.75rem; margin: 0.1rem; display: inline-block; }
</style>
""", unsafe_allow_html=True)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    pinecone_key = st.text_input("Pinecone API Key", type="password", value=os.getenv("PINECONE_API_KEY", ""))
    index_name = st.text_input("Pinecone Index", value=os.getenv("PINECONE_INDEX", "agentic-ai-ebook"))
    groq_key = st.text_input("Groq API Key (FREE)", type="password", value=os.getenv("GROQ_API_KEY", ""))
    openai_key = st.text_input("OpenAI Key (optional)", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    
    st.markdown("---")
    top_k = st.slider("Chunks to retrieve", 1, 10, 6)
    use_llm = st.checkbox("Use LLM", value=True)
    
    st.markdown("---")
    if st.button("🚀 Initialize Pipeline", type="primary", use_container_width=True):
        with st.spinner("Initializing..."):
            try:
                st.session_state.pipeline = RAGPipeline(
                    pinecone_api_key=pinecone_key,
                    index_name=index_name,
                    namespace="agentic-ai",
                    openai_api_key=openai_key if openai_key else None,
                    groq_api_key=groq_key if groq_key else None,
                    local_only=False
                )
                st.success("✅ Ready!")
            except Exception as e:
                st.error(f"❌ {str(e)}")
    
    if st.session_state.pipeline:
        st.success("● Pipeline Ready")
    else:
        st.warning("● Not Initialized")

# Auto-initialize if env vars exist
if st.session_state.pipeline is None:
    pk = os.getenv("PINECONE_API_KEY", "")
    gk = os.getenv("GROQ_API_KEY", "")
    if pk and gk:
        try:
            st.session_state.pipeline = RAGPipeline(
                pinecone_api_key=pk,
                index_name=os.getenv("PINECONE_INDEX", "agentic-ai-ebook"),
                namespace="agentic-ai",
                openai_api_key=os.getenv("OPENAI_API_KEY", ""),
                groq_api_key=gk,
                local_only=False
            )
        except:
            pass

# Header
st.markdown('<h1 class="main-title">🤖 Agentic AI Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Ask questions about the Agentic AI eBook • Grounded answers only</p>', unsafe_allow_html=True)

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "confidence" in msg:
            conf = msg["confidence"]
            conf_class = "conf-high" if conf >= 0.7 else ("conf-medium" if conf >= 0.4 else "conf-low")
            st.markdown(f'<span class="{conf_class}">Confidence: {conf:.0%}</span>', unsafe_allow_html=True)
            if msg.get("sources"):
                sources_html = " ".join([f'<span class="source-chip">📄 P.{s}</span>' for s in msg["sources"][:5]])
                st.markdown(f"Sources: {sources_html}", unsafe_allow_html=True)

# Sample questions if empty
if not st.session_state.messages:
    st.markdown("### 💡 Try asking:")
    samples = [
        "What is the definition of agentic AI?",
        "What are the key characteristics of agentic systems?",
        "What risks does the eBook mention?",
        "What safeguards are recommended?"
    ]
    cols = st.columns(2)
    for i, q in enumerate(samples):
        with cols[i % 2]:
            if st.button(q, key=f"s{i}", use_container_width=True):
                if st.session_state.pipeline:
                    st.session_state.messages.append({"role": "user", "content": q})
                    st.rerun()
                else:
                    st.warning("Initialize pipeline first!")

# Clear button
if st.session_state.messages:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Chat input - MUST be at root level (not inside any container/column)
user_input = st.chat_input("Ask a question about the Agentic AI eBook...")

if user_input:
    if st.session_state.pipeline is None:
        st.warning("⚠️ Please initialize the pipeline first (sidebar → Initialize)")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get response
        try:
            response = st.session_state.pipeline.query(user_input, top_k=top_k, use_llm=use_llm)
            answer = response.get("final_answer", "No answer found.")
            confidence = response.get("confidence", 0.0)
            chunks = response.get("retrieved_chunks", [])
            sources = sorted(set([c.get("page", "?") for c in chunks]))
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "confidence": confidence,
                "sources": sources
            })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Error: {str(e)}",
                "confidence": 0,
                "sources": []
            })
        
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.8rem;">
Built with LangGraph • Pinecone • Groq • Streamlit
</div>
""", unsafe_allow_html=True)
