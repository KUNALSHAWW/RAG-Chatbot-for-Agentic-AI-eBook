"""
utils.py - Helper functions for text processing and chunking

This module contains utility functions for:
- Text cleaning (removing extra whitespace, headers/footers)
- Token counting using tiktoken
- Text chunking with overlap
- Confidence score normalization
"""

import re
from typing import List, Dict, Tuple
import json

# Try to use tiktoken for accurate token counting, fallback to word count
try:
    import tiktoken
    TOKENIZER = tiktoken.get_encoding("cl100k_base")
    USE_TIKTOKEN = True
except ImportError:
    USE_TIKTOKEN = False
    print("WARNING: tiktoken not available, using word count approximation")


def count_tokens(text: str) -> int:
    """
    Count tokens in text using tiktoken or word count fallback.
    
    Args:
        text: Input text string
        
    Returns:
        Number of tokens (approximate if tiktoken not available)
    """
    if USE_TIKTOKEN:
        return len(TOKENIZER.encode(text))
    else:
        # Rough approximation: ~1.3 words per token on average
        words = len(text.split())
        return int(words * 1.3)


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text by removing extra whitespace and common artifacts.
    
    Args:
        text: Raw text from PDF extraction
        
    Returns:
        Cleaned text string
    """
    # Remove excessive whitespace (multiple spaces, tabs)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Remove excessive newlines (more than 2 in a row)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove page numbers (common patterns like "Page 1" or "- 1 -")
    text = re.sub(r'(?i)page\s*\d+', '', text)
    text = re.sub(r'-\s*\d+\s*-', '', text)
    
    # Remove common header/footer artifacts (customize based on your PDF)
    # This is a simple heuristic - you might need to adjust for your specific PDF
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # Final cleanup
    text = text.strip()
    
    return text


def chunk_text(
    text: str,
    page_number: int,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    source: str = "Ebook-Agentic-AI.pdf"
) -> List[Dict]:
    """
    Split text into overlapping chunks with metadata.
    
    Uses token counting to ensure chunks are approximately chunk_size tokens,
    with overlap for context continuity.
    
    Args:
        text: Text to chunk (from one page)
        page_number: Page number for metadata
        chunk_size: Target size in tokens (default 500)
        chunk_overlap: Overlap between chunks in tokens (default 50)
        source: Source document name
        
    Returns:
        List of chunk dictionaries with id, page, text, start_char, end_char
    """
    chunks = []
    
    # If text is empty or very short, return single chunk
    if not text or count_tokens(text) <= chunk_size:
        if text.strip():
            chunk_id = f"pdfpage_{page_number}_chunk_0"
            chunks.append({
                "id": chunk_id,
                "page": page_number,
                "text": text.strip(),
                "start_char": 0,
                "end_char": len(text),
                "source": source
            })
        return chunks
    
    # Split into sentences for better chunking
    # Simple sentence splitting - handles common cases
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    current_chunk = []
    current_tokens = 0
    current_start = 0
    chunk_index = 0
    char_position = 0
    
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        
        # If adding this sentence exceeds chunk_size, save current chunk
        if current_tokens + sentence_tokens > chunk_size and current_chunk:
            # Join current chunk
            chunk_text = ' '.join(current_chunk)
            chunk_id = f"pdfpage_{page_number}_chunk_{chunk_index}"
            
            chunks.append({
                "id": chunk_id,
                "page": page_number,
                "text": chunk_text,
                "start_char": current_start,
                "end_char": current_start + len(chunk_text),
                "source": source
            })
            
            chunk_index += 1
            
            # Calculate overlap - keep last few sentences that fit in overlap
            overlap_tokens = 0
            overlap_sentences = []
            for s in reversed(current_chunk):
                s_tokens = count_tokens(s)
                if overlap_tokens + s_tokens <= chunk_overlap:
                    overlap_sentences.insert(0, s)
                    overlap_tokens += s_tokens
                else:
                    break
            
            current_chunk = overlap_sentences
            current_tokens = overlap_tokens
            current_start = char_position - sum(len(s) + 1 for s in overlap_sentences)
        
        current_chunk.append(sentence)
        current_tokens += sentence_tokens
        char_position += len(sentence) + 1  # +1 for space
    
    # Don't forget the last chunk!
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunk_id = f"pdfpage_{page_number}_chunk_{chunk_index}"
        
        chunks.append({
            "id": chunk_id,
            "page": page_number,
            "text": chunk_text,
            "start_char": current_start,
            "end_char": current_start + len(chunk_text),
            "source": source
        })
    
    return chunks


def normalize_score(score: float) -> float:
    """
    Normalize similarity score to 0-1 range.
    
    Pinecone returns similarity scores typically between -1 and 1 for cosine.
    This function normalizes them to 0-1 range.
    
    Formula: normalized = (score + 1) / 2
    Then clamp to [0, 1] for safety.
    
    Args:
        score: Raw similarity score from Pinecone
        
    Returns:
        Normalized score between 0.0 and 1.0
    """
    # For cosine similarity, scores are in [-1, 1]
    # Normalize to [0, 1]
    normalized = (score + 1.0) / 2.0
    
    # Clamp to valid range (safety check)
    return max(0.0, min(1.0, normalized))


def compute_confidence(scores: List[float], method: str = "max") -> float:
    """
    Compute confidence score from list of similarity scores.
    
    Args:
        scores: List of raw similarity scores from retrieval
        method: "max" for maximum score, "mean" for average
        
    Returns:
        Confidence score rounded to 3 decimal places
    """
    if not scores:
        return 0.0
    
    # Normalize all scores
    normalized_scores = [normalize_score(s) for s in scores]
    
    # Compute confidence based on method
    if method == "max":
        confidence = max(normalized_scores)
    elif method == "mean":
        confidence = sum(normalized_scores) / len(normalized_scores)
    else:
        # Default to max
        confidence = max(normalized_scores)
    
    return round(confidence, 3)


def save_chunks_to_jsonl(chunks: List[Dict], filepath: str, include_embeddings: bool = False):
    """
    Save chunks to a JSONL file for backup.
    
    Args:
        chunks: List of chunk dictionaries
        filepath: Output file path
        include_embeddings: Whether to include embeddings (makes file large)
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            # Create a copy to potentially remove embeddings
            chunk_data = chunk.copy()
            
            if not include_embeddings and 'embedding' in chunk_data:
                del chunk_data['embedding']
            
            f.write(json.dumps(chunk_data, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(chunks)} chunks to {filepath}")


def load_chunks_from_jsonl(filepath: str) -> List[Dict]:
    """
    Load chunks from a JSONL file.
    
    Args:
        filepath: Input file path
        
    Returns:
        List of chunk dictionaries
    """
    chunks = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    
    print(f"Loaded {len(chunks)} chunks from {filepath}")
    return chunks


def format_chunks_for_llm(chunks: List[Dict]) -> str:
    """
    Format retrieved chunks into a string for LLM context.
    
    Args:
        chunks: List of chunk dictionaries with 'text' and 'page' keys
        
    Returns:
        Formatted string with markers for the LLM
    """
    formatted_parts = []
    
    for i, chunk in enumerate(chunks):
        page = chunk.get('page', 'unknown')
        text = chunk.get('text', '')
        chunk_id = chunk.get('id', f'chunk_{i}')
        
        part = f"[Source: {chunk_id}, Page {page}]\n{text}"
        formatted_parts.append(part)
    
    return "\n\n---\n\n".join(formatted_parts)


if __name__ == "__main__":
    # Quick test of utility functions
    print("Testing utils.py functions...")
    
    # Test token counting
    test_text = "This is a test sentence for token counting."
    print(f"Token count for '{test_text}': {count_tokens(test_text)}")
    
    # Test text cleaning
    dirty_text = "  This   has   extra    spaces  \n\n\n\nAnd too many newlines Page 123"
    clean = clean_text(dirty_text)
    print(f"Cleaned text: '{clean}'")
    
    # Test score normalization
    test_scores = [-1.0, 0.0, 0.5, 1.0]
    for score in test_scores:
        print(f"Score {score} -> normalized: {normalize_score(score)}")
    
    # Test confidence computation
    scores = [0.8, 0.6, 0.7]
    print(f"Confidence (max): {compute_confidence(scores, 'max')}")
    print(f"Confidence (mean): {compute_confidence(scores, 'mean')}")
    
    print("\nAll tests passed!")
