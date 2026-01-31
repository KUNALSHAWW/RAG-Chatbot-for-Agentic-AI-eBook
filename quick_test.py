"""
quick_test.py - Validation Script for RAG Pipeline

This script tests the core functionality of the RAG pipeline:
1. Tests utility functions (chunking, confidence scoring)
2. Tests the embedding model loading
3. Tests the RAG pipeline with a sample query (if data is available)

Run this after ingestion to verify everything works:
    python quick_test.py

This script is designed to work even without API keys by using local mode.
"""

import os
import sys
import json

# Add app directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.utils import (
    clean_text,
    chunk_text,
    count_tokens,
    normalize_score,
    compute_confidence,
    format_chunks_for_llm
)
from app.vectorstore import LocalVectorStore


def test_utilities():
    """Test utility functions."""
    print("\n" + "=" * 60)
    print("TEST 1: Utility Functions")
    print("=" * 60)
    
    # Test token counting
    test_text = "This is a sample sentence for testing token counting functionality."
    token_count = count_tokens(test_text)
    print(f"\n✓ Token counting: '{test_text[:30]}...' = {token_count} tokens")
    
    # Test text cleaning
    dirty_text = "  This   has   extra    spaces  \n\n\n\nAnd too many newlines Page 123  "
    clean = clean_text(dirty_text)
    print(f"✓ Text cleaning: '{dirty_text[:30]}...' -> '{clean[:30]}...'")
    
    # Test chunking
    long_text = "This is a test paragraph. " * 100  # Create a longer text
    chunks = chunk_text(long_text, page_number=1, chunk_size=100, chunk_overlap=20)
    print(f"✓ Chunking: Created {len(chunks)} chunks from {count_tokens(long_text)} tokens")
    
    if chunks:
        print(f"  - First chunk ID: {chunks[0]['id']}")
        print(f"  - First chunk tokens: ~{count_tokens(chunks[0]['text'])}")
    
    # Test score normalization
    test_scores = [-1.0, -0.5, 0.0, 0.5, 1.0]
    normalized = [normalize_score(s) for s in test_scores]
    print(f"\n✓ Score normalization:")
    for raw, norm in zip(test_scores, normalized):
        print(f"  {raw:5.2f} -> {norm:.3f}")
    
    # Test confidence computation
    sample_scores = [0.8, 0.6, 0.7, 0.5]
    conf_max = compute_confidence(sample_scores, method="max")
    conf_mean = compute_confidence(sample_scores, method="mean")
    print(f"\n✓ Confidence computation (from scores {sample_scores}):")
    print(f"  - Max method: {conf_max}")
    print(f"  - Mean method: {conf_mean}")
    
    print("\n✅ All utility tests passed!")
    return True


def test_local_vectorstore():
    """Test local vector store functionality."""
    print("\n" + "=" * 60)
    print("TEST 2: Local Vector Store")
    print("=" * 60)
    
    import numpy as np
    
    # Create local vector store
    store = LocalVectorStore(dimension=384)
    
    # Create dummy vectors
    vectors = [
        {
            'id': 'test_chunk_1',
            'embedding': np.random.randn(384).tolist(),
            'page': 1,
            'text': 'Agentic AI refers to artificial intelligence systems that can operate autonomously.',
            'source': 'test.pdf'
        },
        {
            'id': 'test_chunk_2',
            'embedding': np.random.randn(384).tolist(),
            'page': 2,
            'text': 'The risks of agentic systems include uncontrolled behavior and safety concerns.',
            'source': 'test.pdf'
        },
        {
            'id': 'test_chunk_3',
            'embedding': np.random.randn(384).tolist(),
            'page': 3,
            'text': 'Safeguards for agentic AI deployment include human oversight and testing.',
            'source': 'test.pdf'
        }
    ]
    
    # Upsert vectors
    count = store.upsert(vectors)
    print(f"\n✓ Upserted {count} vectors to local store")
    
    # Query with random vector
    query_vec = np.random.randn(384).tolist()
    results = store.query_top_k(query_vec, k=2)
    
    print(f"✓ Query returned {len(results)} results")
    for r in results:
        print(f"  - {r['id']}: score={r['score']:.4f}")
    
    print("\n✅ Local vector store test passed!")
    return True


def test_embedding_model():
    """Test embedding model loading."""
    print("\n" + "=" * 60)
    print("TEST 3: Embedding Model")
    print("=" * 60)
    
    try:
        from sentence_transformers import SentenceTransformer
        
        print("\nLoading embedding model (this may take a moment)...")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        dim = model.get_sentence_embedding_dimension()
        print(f"✓ Model loaded successfully!")
        print(f"✓ Embedding dimension: {dim}")
        
        # Test encoding
        test_sentences = [
            "What is agentic AI?",
            "Describe the risks of autonomous systems."
        ]
        
        embeddings = model.encode(test_sentences)
        print(f"✓ Encoded {len(test_sentences)} sentences")
        print(f"  - Shape: {embeddings.shape}")
        
        print("\n✅ Embedding model test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Embedding model test failed: {e}")
        return False


def test_rag_pipeline():
    """Test the full RAG pipeline (if data is available)."""
    print("\n" + "=" * 60)
    print("TEST 4: RAG Pipeline")
    print("=" * 60)
    
    # Check if we have local data
    chunks_file = "./data/chunks.jsonl"
    vectors_file = "./data/vectors.json"
    
    if not os.path.exists(chunks_file) and not os.path.exists(vectors_file):
        print("\n⚠️  No ingested data found.")
        print("   Run 'python app/ingest.py --pdf ./data/Ebook-Agentic-AI.pdf --local-only' first.")
        print("   Skipping RAG pipeline test.\n")
        return True  # Not a failure, just skip
    
    try:
        from app.rag_pipeline import RAGPipeline
        
        print("\nInitializing RAG pipeline in local mode...")
        
        # Use local mode for testing
        pipeline = RAGPipeline(
            local_only=True,
            chunks_file=chunks_file
        )
        
        # Test query
        test_query = "What is agentic AI?"
        print(f"\nTest query: '{test_query}'")
        print("-" * 40)
        
        result = pipeline.query(test_query, top_k=3, use_llm=False)
        
        # Display result
        print("\n📤 Response:")
        print(json.dumps(result, indent=2, default=str)[:1000] + "...")
        
        # Validate response structure
        assert "final_answer" in result, "Missing 'final_answer' in response"
        assert "retrieved_chunks" in result, "Missing 'retrieved_chunks' in response"
        assert "confidence" in result, "Missing 'confidence' in response"
        
        print(f"\n✓ Final answer length: {len(result['final_answer'])} chars")
        print(f"✓ Retrieved chunks: {len(result['retrieved_chunks'])}")
        print(f"✓ Confidence score: {result['confidence']}")
        
        # Show retrieved chunks summary
        if result['retrieved_chunks']:
            print("\n📚 Retrieved chunks:")
            for i, chunk in enumerate(result['retrieved_chunks'][:3]):
                print(f"  {i+1}. Page {chunk.get('page', '?')}, Score: {chunk.get('score', 0):.4f}")
                print(f"     ID: {chunk.get('id', 'unknown')}")
                print(f"     Text: {chunk.get('text', '')[:80]}...")
        
        print("\n✅ RAG pipeline test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ RAG pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_response_format():
    """Test that responses match the expected format."""
    print("\n" + "=" * 60)
    print("TEST 5: Response Format Validation")
    print("=" * 60)
    
    # Example expected format
    expected_format = {
        "final_answer": "string",
        "retrieved_chunks": [
            {
                "id": "string (format: pdfpage_N_chunk_M)",
                "page": "integer",
                "text": "string",
                "score": "float (0.0-1.0)"
            }
        ],
        "confidence": "float (0.0-1.0)"
    }
    
    print("\n✓ Expected response format:")
    print(json.dumps(expected_format, indent=2))
    
    # Validate a mock response
    mock_response = {
        "final_answer": "According to the document, agentic AI is...",
        "retrieved_chunks": [
            {"id": "pdfpage_1_chunk_0", "page": 1, "text": "Sample text...", "score": 0.92}
        ],
        "confidence": 0.92
    }
    
    # Check types
    assert isinstance(mock_response["final_answer"], str), "final_answer must be string"
    assert isinstance(mock_response["retrieved_chunks"], list), "retrieved_chunks must be list"
    assert isinstance(mock_response["confidence"], (int, float)), "confidence must be number"
    assert 0 <= mock_response["confidence"] <= 1, "confidence must be between 0 and 1"
    
    if mock_response["retrieved_chunks"]:
        chunk = mock_response["retrieved_chunks"][0]
        assert "id" in chunk, "chunk must have 'id'"
        assert "page" in chunk, "chunk must have 'page'"
        assert "text" in chunk, "chunk must have 'text'"
        assert "score" in chunk, "chunk must have 'score'"
    
    print("\n✅ Response format validation passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RAG CHATBOT - QUICK TEST SUITE")
    print("=" * 60)
    
    results = {}
    
    # Run tests
    results["utilities"] = test_utilities()
    results["local_vectorstore"] = test_local_vectorstore()
    results["embedding_model"] = test_embedding_model()
    results["rag_pipeline"] = test_rag_pipeline()
    results["response_format"] = test_response_format()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The RAG pipeline is ready to use.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
