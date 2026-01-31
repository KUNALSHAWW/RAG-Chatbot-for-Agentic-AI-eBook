"""
vectorstore.py - Pinecone Vector Database Wrapper

This module provides a clean wrapper around the Pinecone Python client for:
- Creating an index if it doesn't exist
- Upserting vectors in batches
- Querying for similar vectors (top-k retrieval)

Requires: PINECONE_API_KEY environment variable
"""

import os
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Try to import Pinecone
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    print("WARNING: pinecone-client not installed. Vector operations will be disabled.")


class PineconeVectorStore:
    """
    Wrapper class for Pinecone vector database operations.
    
    Provides simple methods for creating indexes, upserting vectors,
    and querying for similar vectors.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: str = "agentic-ai-ebook",
        namespace: str = "agentic-ai",
        dimension: int = 384,  # all-MiniLM-L6-v2 produces 384-dim vectors
        metric: str = "cosine"
    ):
        """
        Initialize the Pinecone vector store.
        
        Args:
            api_key: Pinecone API key (or set PINECONE_API_KEY env var)
            index_name: Name of the Pinecone index
            namespace: Namespace within the index
            dimension: Dimension of vectors (384 for all-MiniLM-L6-v2)
            metric: Similarity metric ('cosine', 'euclidean', 'dotproduct')
        """
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.index_name = index_name
        self.namespace = namespace
        self.dimension = dimension
        self.metric = metric
        
        self.pc = None
        self.index = None
        
        # Local chunk storage for retrieval (maps chunk_id -> chunk_data)
        self.chunks_map: Dict[str, Dict] = {}
        
        if self.api_key and PINECONE_AVAILABLE:
            self._initialize_pinecone()
        else:
            print("WARNING: Running without Pinecone. Use --local-only mode for local storage.")
    
    def _initialize_pinecone(self):
        """Initialize connection to Pinecone."""
        try:
            self.pc = Pinecone(api_key=self.api_key)
            print(f"Connected to Pinecone successfully!")
        except Exception as e:
            print(f"ERROR: Failed to connect to Pinecone: {e}")
            self.pc = None
    
    def create_index_if_missing(self) -> bool:
        """
        Create the Pinecone index if it doesn't exist.
        
        Returns:
            True if index exists or was created, False on error
        """
        if not self.pc:
            print("ERROR: Pinecone not initialized")
            return False
        
        try:
            # Get list of existing indexes
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                print(f"Creating new index: {self.index_name}")
                
                # Create serverless index (free tier compatible)
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"  # Free tier region
                    )
                )
                print(f"Index '{self.index_name}' created successfully!")
            else:
                print(f"Index '{self.index_name}' already exists")
            
            # Connect to the index
            self.index = self.pc.Index(self.index_name)
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to create/connect to index: {e}")
            return False
    
    def upsert(
        self,
        items: List[Dict],
        batch_size: int = 100
    ) -> int:
        """
        Upsert vectors to Pinecone in batches.
        
        Args:
            items: List of dicts with 'id', 'embedding', and metadata
            batch_size: Number of vectors per batch (default 100)
            
        Returns:
            Number of vectors upserted
        """
        if not self.index:
            print("ERROR: Index not initialized. Call create_index_if_missing() first.")
            return 0
        
        # Store chunks locally for retrieval
        for item in items:
            self.chunks_map[item['id']] = {
                'id': item['id'],
                'page': item.get('page', 0),
                'text': item.get('text', ''),
                'source': item.get('source', '')
            }
        
        # Prepare vectors for Pinecone format
        vectors = []
        for item in items:
            vector = {
                'id': item['id'],
                'values': item['embedding'],
                'metadata': {
                    'page': item.get('page', 0),
                    'text': item.get('text', '')[:1000],  # Pinecone metadata limit
                    'source': item.get('source', '')
                }
            }
            vectors.append(vector)
        
        # Upsert in batches
        total_upserted = 0
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            try:
                self.index.upsert(
                    vectors=batch,
                    namespace=self.namespace
                )
                total_upserted += len(batch)
                print(f"Upserted batch {i//batch_size + 1}: {len(batch)} vectors")
            except Exception as e:
                print(f"ERROR: Failed to upsert batch: {e}")
        
        print(f"Total vectors upserted: {total_upserted}")
        return total_upserted
    
    def query_top_k(
        self,
        query_vector: List[float],
        k: int = 5,
        include_metadata: bool = True
    ) -> List[Dict]:
        """
        Query Pinecone for top-k similar vectors.
        
        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of results with id, score, and metadata
        """
        if not self.index:
            print("ERROR: Index not initialized")
            return []
        
        try:
            results = self.index.query(
                vector=query_vector,
                top_k=k,
                namespace=self.namespace,
                include_metadata=include_metadata
            )
            
            # Format results
            formatted_results = []
            for match in results.get('matches', []):
                result = {
                    'id': match['id'],
                    'score': match['score'],
                    'page': match.get('metadata', {}).get('page', 0),
                    'text': match.get('metadata', {}).get('text', ''),
                    'source': match.get('metadata', {}).get('source', '')
                }
                
                # If text is truncated in metadata, try to get full text from local cache
                if result['id'] in self.chunks_map:
                    result['text'] = self.chunks_map[result['id']].get('text', result['text'])
                
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"ERROR: Query failed: {e}")
            return []
    
    def load_chunks_map(self, filepath: str):
        """
        Load chunk data from a JSONL file to enable full text retrieval.
        
        Args:
            filepath: Path to chunks.jsonl file
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        chunk = json.loads(line)
                        self.chunks_map[chunk['id']] = chunk
            print(f"Loaded {len(self.chunks_map)} chunks into memory")
        except FileNotFoundError:
            print(f"WARNING: {filepath} not found. Full text retrieval may be limited.")
        except Exception as e:
            print(f"ERROR: Failed to load chunks: {e}")
    
    def get_index_stats(self) -> Dict:
        """
        Get statistics about the Pinecone index.
        
        Returns:
            Dictionary with index statistics
        """
        if not self.index:
            return {"error": "Index not initialized"}
        
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.get('total_vector_count', 0),
                "namespaces": stats.get('namespaces', {}),
                "dimension": stats.get('dimension', self.dimension)
            }
        except Exception as e:
            return {"error": str(e)}


class LocalVectorStore:
    """
    Local vector store for testing without Pinecone.
    
    Stores vectors in memory and performs brute-force similarity search.
    Useful for --local-only mode and testing.
    """
    
    def __init__(self, dimension: int = 384):
        """
        Initialize local vector store.
        
        Args:
            dimension: Dimension of vectors
        """
        self.dimension = dimension
        self.vectors: Dict[str, Dict] = {}  # id -> {embedding, metadata}
        print("Using LOCAL vector store (no Pinecone)")
    
    def upsert(self, items: List[Dict]) -> int:
        """Add vectors to local store."""
        for item in items:
            self.vectors[item['id']] = {
                'embedding': item['embedding'],
                'page': item.get('page', 0),
                'text': item.get('text', ''),
                'source': item.get('source', '')
            }
        print(f"Stored {len(items)} vectors locally")
        return len(items)
    
    def query_top_k(
        self,
        query_vector: List[float],
        k: int = 5
    ) -> List[Dict]:
        """
        Brute-force similarity search.
        
        Args:
            query_vector: Query embedding
            k: Number of results
            
        Returns:
            Top-k results with scores
        """
        import numpy as np
        
        if not self.vectors:
            return []
        
        query_np = np.array(query_vector)
        
        # Compute cosine similarity with all vectors
        scores = []
        for vec_id, data in self.vectors.items():
            vec_np = np.array(data['embedding'])
            
            # Cosine similarity
            similarity = np.dot(query_np, vec_np) / (
                np.linalg.norm(query_np) * np.linalg.norm(vec_np) + 1e-8
            )
            
            scores.append({
                'id': vec_id,
                'score': float(similarity),
                'page': data['page'],
                'text': data['text'],
                'source': data['source']
            })
        
        # Sort by score descending and return top-k
        scores.sort(key=lambda x: x['score'], reverse=True)
        return scores[:k]
    
    def save_to_file(self, filepath: str):
        """Save vectors to JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.vectors, f)
        print(f"Saved {len(self.vectors)} vectors to {filepath}")
    
    def load_from_file(self, filepath: str):
        """Load vectors from JSON file."""
        import json
        try:
            with open(filepath, 'r') as f:
                self.vectors = json.load(f)
            print(f"Loaded {len(self.vectors)} vectors from {filepath}")
        except FileNotFoundError:
            print(f"WARNING: {filepath} not found")


def get_vector_store(
    local_only: bool = False,
    api_key: Optional[str] = None,
    index_name: str = "agentic-ai-ebook",
    **kwargs
):
    """
    Factory function to get the appropriate vector store.
    
    Args:
        local_only: If True, use local storage instead of Pinecone
        api_key: Pinecone API key
        index_name: Name of the index
        
    Returns:
        Vector store instance (Pinecone or Local)
    """
    if local_only or not PINECONE_AVAILABLE:
        return LocalVectorStore(**kwargs)
    
    return PineconeVectorStore(
        api_key=api_key,
        index_name=index_name,
        **kwargs
    )


if __name__ == "__main__":
    # Quick test
    print("Testing vectorstore.py...")
    
    # Test local vector store
    local_store = LocalVectorStore(dimension=384)
    
    # Add some dummy vectors
    import numpy as np
    test_items = [
        {
            'id': 'test_1',
            'embedding': np.random.randn(384).tolist(),
            'page': 1,
            'text': 'This is a test chunk about AI.',
            'source': 'test.pdf'
        },
        {
            'id': 'test_2',
            'embedding': np.random.randn(384).tolist(),
            'page': 2,
            'text': 'This chunk discusses machine learning.',
            'source': 'test.pdf'
        }
    ]
    
    local_store.upsert(test_items)
    
    # Query
    query_vec = np.random.randn(384).tolist()
    results = local_store.query_top_k(query_vec, k=2)
    
    print(f"\nQuery results: {len(results)} matches")
    for r in results:
        print(f"  - {r['id']}: score={r['score']:.3f}")
    
    print("\nLocal vector store test passed!")
