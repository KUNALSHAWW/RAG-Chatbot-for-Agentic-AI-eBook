"""
rag_pipeline.py - LangGraph RAG Pipeline

This module implements the RAG pipeline using LangGraph for orchestration:
1. Receive user query
2. Embed query using sentence-transformers
3. Query Pinecone for top-k similar chunks
4. Generate answer using LLM (if available) or extractive fallback
5. Return structured response with answer, chunks, and confidence

The pipeline enforces strict grounding - answers must come from retrieved chunks only.
"""

import os
import json
from typing import List, Dict, Any, Optional, TypedDict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import LangGraph components
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("WARNING: langgraph not installed. Using simplified pipeline.")

# Import embedding model
from sentence_transformers import SentenceTransformer

# Import local modules
from app.vectorstore import PineconeVectorStore, LocalVectorStore, get_vector_store
from app.utils import compute_confidence, normalize_score, format_chunks_for_llm, load_chunks_from_jsonl

# Try to import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try to import Groq (free LLM alternative)
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Cache the embedding model to avoid reloading
_EMBEDDING_MODEL_CACHE = {}

def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Load and cache the embedding model to avoid repeated loading."""
    if model_name not in _EMBEDDING_MODEL_CACHE:
        print(f"Loading embedding model: {model_name}")
        # Use device='cpu' explicitly to avoid meta tensor issues
        model = SentenceTransformer(model_name, device='cpu')
        _EMBEDDING_MODEL_CACHE[model_name] = model
    return _EMBEDDING_MODEL_CACHE[model_name]


# ============================================================================
# LangGraph State Definition
# ============================================================================

class RAGState(TypedDict):
    """State object passed through the RAG pipeline."""
    query: str
    query_embedding: Optional[List[float]]
    retrieved_chunks: List[Dict]
    raw_scores: List[float]
    confidence: float
    final_answer: str
    use_llm: bool
    error: Optional[str]


# ============================================================================
# Pipeline Nodes (Functions)
# ============================================================================

class RAGPipeline:
    """
    RAG Pipeline implementation using LangGraph.
    
    The pipeline has the following stages:
    1. embed_query - Convert query to vector
    2. retrieve_chunks - Get relevant chunks from Pinecone
    3. compute_confidence - Calculate confidence score
    4. generate_answer - Use LLM or extractive fallback
    """
    
    # System prompt for LLM - VERY IMPORTANT for grounding
    SYSTEM_PROMPT = """You are an assistant answering questions based on provided document excerpts from an eBook about Agentic AI.

IMPORTANT INSTRUCTIONS:
1. Synthesize information from ALL the provided excerpts to give a comprehensive answer.
2. The excerpts may contain relevant information even if they don't directly state the answer - look for definitions, explanations, and examples.
3. Combine information from multiple excerpts when helpful.
4. Cite page numbers when referencing specific information.
5. Do NOT add any information that is not in the excerpts.
6. If absolutely no relevant information exists in ANY excerpt, only then say: "I could not find a supported answer in the document."

Be helpful and thorough - users want complete answers based on the document content."""

    def __init__(
        self,
        pinecone_api_key: Optional[str] = None,
        index_name: str = "agentic-ai-ebook",
        namespace: str = "agentic-ai",
        openai_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 6,
        local_only: bool = False,
        chunks_file: Optional[str] = None
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            pinecone_api_key: Pinecone API key (or set PINECONE_API_KEY env var)
            index_name: Name of Pinecone index
            namespace: Pinecone namespace
            openai_api_key: OpenAI API key for LLM (optional)
            groq_api_key: Groq API key for LLM (optional, free alternative)
            embedding_model_name: Name of embedding model
            top_k: Number of chunks to retrieve
            local_only: Use local vector store instead of Pinecone
            chunks_file: Path to chunks.jsonl for local retrieval
        """
        self.top_k = top_k
        self.local_only = local_only
        
        # Load embedding model (cached to avoid reloading)
        self.embedding_model = get_embedding_model(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize vector store
        if local_only:
            self.vector_store = LocalVectorStore(dimension=self.embedding_dim)
            # Load vectors from file if provided
            vectors_file = chunks_file.replace('chunks.jsonl', 'vectors.json') if chunks_file else './data/vectors.json'
            if os.path.exists(vectors_file):
                self.vector_store.load_from_file(vectors_file)
        else:
            api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
            self.vector_store = PineconeVectorStore(
                api_key=api_key,
                index_name=index_name,
                namespace=namespace,
                dimension=self.embedding_dim
            )
            # Connect to existing index
            if self.vector_store.pc:
                self.vector_store.index = self.vector_store.pc.Index(index_name)
        
        # Load chunks for full text retrieval
        if chunks_file and os.path.exists(chunks_file):
            self.vector_store.load_chunks_map(chunks_file)
        elif os.path.exists('./data/chunks.jsonl'):
            self.vector_store.load_chunks_map('./data/chunks.jsonl')
        
        # Initialize LLM client - prefer Groq (free), then OpenAI
        self.openai_client = None
        self.groq_client = None
        self.llm_provider = None
        
        # Try Groq first (it's free!)
        groq_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if groq_key and GROQ_AVAILABLE:
            self.groq_client = Groq(api_key=groq_key)
            self.llm_provider = "groq"
            print("Groq client initialized - will use Groq LLM for answer generation (FREE!)")
        else:
            # Fall back to OpenAI
            openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if openai_key and OPENAI_AVAILABLE:
                self.openai_client = OpenAI(api_key=openai_key)
                self.llm_provider = "openai"
                print("OpenAI client initialized - will use OpenAI LLM for answer generation")
            else:
                print("No LLM key - will use extractive answer fallback")
        
        # Build the LangGraph pipeline
        if LANGGRAPH_AVAILABLE:
            self.graph = self._build_graph()
        else:
            self.graph = None
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        
        # Create the graph
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("embed_query", self._embed_query_node)
        workflow.add_node("retrieve_chunks", self._retrieve_chunks_node)
        workflow.add_node("calculate_confidence", self._calculate_confidence_node)
        workflow.add_node("generate_answer", self._generate_answer_node)
        
        # Define the flow
        workflow.set_entry_point("embed_query")
        workflow.add_edge("embed_query", "retrieve_chunks")
        workflow.add_edge("retrieve_chunks", "calculate_confidence")
        workflow.add_edge("calculate_confidence", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        # Compile the graph
        return workflow.compile()
    
    def _embed_query_node(self, state: RAGState) -> Dict:
        """Embed the user query."""
        query = state["query"]
        
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(query, convert_to_numpy=True)
            return {"query_embedding": embedding.tolist()}
        except Exception as e:
            return {"error": f"Embedding failed: {str(e)}"}
    
    def _retrieve_chunks_node(self, state: RAGState) -> Dict:
        """Retrieve relevant chunks from vector store."""
        query_embedding = state.get("query_embedding")
        
        if not query_embedding:
            return {"error": "No query embedding available"}
        
        try:
            # Query vector store
            results = self.vector_store.query_top_k(
                query_vector=query_embedding,
                k=self.top_k
            )
            
            # Extract chunks and scores
            retrieved_chunks = []
            raw_scores = []
            
            for result in results:
                retrieved_chunks.append({
                    "id": result["id"],
                    "page": result["page"],
                    "text": result["text"],
                    "score": round(result["score"], 4)
                })
                raw_scores.append(result["score"])
            
            return {
                "retrieved_chunks": retrieved_chunks,
                "raw_scores": raw_scores
            }
            
        except Exception as e:
            return {"error": f"Retrieval failed: {str(e)}"}
    
    def _calculate_confidence_node(self, state: RAGState) -> Dict:
        """Calculate confidence score from retrieval scores."""
        raw_scores = state.get("raw_scores", [])
        
        if not raw_scores:
            return {"confidence": 0.0}
        
        # Compute confidence using max of normalized scores
        confidence = compute_confidence(raw_scores, method="max")
        return {"confidence": confidence}
    
    def _generate_answer_node(self, state: RAGState) -> Dict:
        """Generate the final answer using LLM or extractive fallback."""
        query = state["query"]
        chunks = state.get("retrieved_chunks", [])
        use_llm = state.get("use_llm", True)
        
        if not chunks:
            return {
                "final_answer": "I could not find any relevant information in the document."
            }
        
        # Format chunks for context
        context = format_chunks_for_llm(chunks)
        
        # Try LLM generation if available and requested
        if use_llm and (self.groq_client or self.openai_client):
            try:
                answer = self._generate_with_llm(query, context)
                return {"final_answer": answer}
            except Exception as e:
                print(f"LLM generation failed: {e}, falling back to extractive")
        
        # Extractive fallback - return the most relevant chunks
        answer = self._generate_extractive_answer(query, chunks)
        return {"final_answer": answer}
    
    def _generate_with_llm(self, query: str, context: str) -> str:
        """
        Generate answer using LLM (Groq or OpenAI).
        
        The prompt strictly instructs the model to only use provided context.
        """
        # Construct the user message with context
        user_message = f"""===BEGIN EXCERPTS===
{context}
===END EXCERPTS===

Question: {query}

Answer the question using ONLY the information from the excerpts above. If the answer is not in the excerpts, say "I could not find a supported answer in the document."
"""
        
        # Use Groq if available (it's free!)
        if self.groq_client:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Fast and free model
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1,  # Low temperature for more factual responses
                max_tokens=500
            )
            return response.choices[0].message.content
        
        # Fall back to OpenAI
        if self.openai_client:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1,
                max_tokens=500
            )
            return response.choices[0].message.content
        
        raise Exception("No LLM client available")
    
    def _generate_extractive_answer(self, query: str, chunks: List[Dict]) -> str:
        """
        Generate an extractive answer by returning the most relevant chunks.
        
        This is the fallback when no LLM is available.
        """
        # Header
        answer_parts = [
            "**Answer based on document excerpts:**\n",
            "*(No LLM available - showing relevant passages from the document)*\n"
        ]
        
        # Add top chunks (limit to top 2-3 for readability)
        top_chunks = chunks[:3]
        
        for i, chunk in enumerate(top_chunks, 1):
            page = chunk.get("page", "unknown")
            text = chunk.get("text", "")
            score = chunk.get("score", 0)
            
            # Truncate very long chunks
            if len(text) > 500:
                text = text[:500] + "..."
            
            answer_parts.append(f"\n**[Excerpt {i}, Page {page}]** (relevance: {score:.2f})")
            answer_parts.append(f"\n{text}\n")
        
        return "".join(answer_parts)
    
    def query(
        self,
        user_query: str,
        top_k: Optional[int] = None,
        use_llm: bool = True
    ) -> Dict[str, Any]:
        """
        Run a query through the RAG pipeline.
        
        Args:
            user_query: The user's question
            top_k: Number of chunks to retrieve (overrides default)
            use_llm: Whether to use LLM for generation
            
        Returns:
            Dict with final_answer, retrieved_chunks, and confidence
        """
        # Override top_k if provided
        if top_k:
            self.top_k = top_k
        
        # Initial state
        initial_state: RAGState = {
            "query": user_query,
            "query_embedding": None,
            "retrieved_chunks": [],
            "raw_scores": [],
            "confidence": 0.0,
            "final_answer": "",
            "use_llm": use_llm and (self.groq_client is not None or self.openai_client is not None),
            "error": None
        }
        
        # Run the pipeline
        if self.graph:
            # Use LangGraph
            final_state = self.graph.invoke(initial_state)
        else:
            # Fallback: run nodes manually
            final_state = initial_state
            
            # Embed query
            result = self._embed_query_node(final_state)
            final_state.update(result)
            
            # Retrieve chunks
            result = self._retrieve_chunks_node(final_state)
            final_state.update(result)
            
            # Calculate confidence
            result = self._calculate_confidence_node(final_state)
            final_state.update(result)
            
            # Generate answer
            result = self._generate_answer_node(final_state)
            final_state.update(result)
        
        # Format response
        response = {
            "final_answer": final_state.get("final_answer", ""),
            "retrieved_chunks": final_state.get("retrieved_chunks", []),
            "confidence": final_state.get("confidence", 0.0)
        }
        
        # Add error if any
        if final_state.get("error"):
            response["error"] = final_state["error"]
        
        return response


# ============================================================================
# Convenience function for simple usage
# ============================================================================

def create_rag_pipeline(
    pinecone_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    index_name: str = "agentic-ai-ebook",
    local_only: bool = False
) -> RAGPipeline:
    """
    Factory function to create a RAG pipeline.
    
    Args:
        pinecone_api_key: Pinecone API key
        openai_api_key: OpenAI API key
        index_name: Name of Pinecone index
        local_only: Use local storage instead of Pinecone
        
    Returns:
        Configured RAGPipeline instance
    """
    return RAGPipeline(
        pinecone_api_key=pinecone_api_key,
        openai_api_key=openai_api_key,
        index_name=index_name,
        local_only=local_only
    )


# ============================================================================
# Main - Quick test
# ============================================================================

if __name__ == "__main__":
    print("Testing RAG Pipeline...")
    print("=" * 60)
    
    # Check if we should use local mode
    local_mode = not os.getenv("PINECONE_API_KEY")
    
    if local_mode:
        print("No PINECONE_API_KEY found, using local mode")
        print("Make sure you have run ingest.py with --local-only first!")
    
    # Create pipeline
    pipeline = RAGPipeline(local_only=local_mode)
    
    # Test query
    test_query = "What is agentic AI?"
    print(f"\nTest query: {test_query}")
    print("-" * 40)
    
    result = pipeline.query(test_query)
    
    print(f"\nFinal Answer:")
    print(result["final_answer"])
    print(f"\nConfidence: {result['confidence']}")
    print(f"\nRetrieved Chunks: {len(result['retrieved_chunks'])}")
    
    for chunk in result["retrieved_chunks"]:
        print(f"  - {chunk['id']} (page {chunk['page']}, score: {chunk['score']})")
    
    print("\n" + "=" * 60)
    print("Pipeline test complete!")
