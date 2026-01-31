"""
ingest.py - PDF Ingestion Pipeline

This script handles the complete ingestion workflow:
1. Read PDF file and extract text by page
2. Clean the extracted text
3. Chunk the text with overlap (500 tokens, 50-100 overlap)
4. Generate embeddings using sentence-transformers
5. Upsert to Pinecone (or save locally with --local-only)
6. Save chunks.jsonl as backup

Usage:
    python app/ingest.py --pdf ./data/Ebook-Agentic-AI.pdf --index agentic-ai
    python app/ingest.py --pdf ./data/Ebook-Agentic-AI.pdf --local-only  # No Pinecone

Requires:
    - PINECONE_API_KEY environment variable (unless using --local-only)
    - PDF file at specified path
"""

import os
import sys
import argparse
from typing import List, Dict, Tuple
from tqdm import tqdm
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports
from app.utils import clean_text, chunk_text, save_chunks_to_jsonl
from app.vectorstore import get_vector_store, PineconeVectorStore, LocalVectorStore

# Load environment variables
load_dotenv()

# Try to import PDF library
try:
    import pdfplumber
    PDF_LIBRARY = "pdfplumber"
except ImportError:
    try:
        import PyPDF2
        PDF_LIBRARY = "PyPDF2"
    except ImportError:
        print("ERROR: Neither pdfplumber nor PyPDF2 installed. Please install one.")
        sys.exit(1)

# Embedding model
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
except ImportError:
    print("ERROR: sentence-transformers not installed")
    sys.exit(1)


def extract_text_from_pdf(pdf_path: str) -> List[Tuple[int, str]]:
    """
    Extract text from PDF file, returning text by page.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of tuples: (page_number, page_text)
    """
    print(f"Extracting text from: {pdf_path}")
    pages = []
    
    if PDF_LIBRARY == "pdfplumber":
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages.append((i + 1, text))  # 1-indexed page numbers
                
    elif PDF_LIBRARY == "PyPDF2":
        import PyPDF2
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                pages.append((i + 1, text))
    
    print(f"Extracted {len(pages)} pages")
    return pages


def load_embedding_model():
    """
    Load the sentence-transformers embedding model.
    
    Returns:
        SentenceTransformer model instance
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"Model loaded! Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def generate_embeddings(
    chunks: List[Dict],
    model: SentenceTransformer,
    batch_size: int = 32
) -> List[Dict]:
    """
    Generate embeddings for all chunks.
    
    Args:
        chunks: List of chunk dictionaries (must have 'text' key)
        model: SentenceTransformer model
        batch_size: Batch size for embedding generation
        
    Returns:
        Chunks with 'embedding' field added
    """
    print(f"Generating embeddings for {len(chunks)} chunks...")
    
    # Extract texts
    texts = [chunk['text'] for chunk in chunks]
    
    # Generate embeddings in batches
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    # Add embeddings to chunks
    for i, chunk in enumerate(chunks):
        chunk['embedding'] = embeddings[i].tolist()
    
    print(f"Generated {len(embeddings)} embeddings")
    return chunks


def run_ingestion(
    pdf_path: str,
    index_name: str = "agentic-ai-ebook",
    namespace: str = "agentic-ai",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    local_only: bool = False,
    output_dir: str = "./data"
):
    """
    Run the complete ingestion pipeline.
    
    Args:
        pdf_path: Path to the PDF file
        index_name: Pinecone index name
        namespace: Pinecone namespace
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        local_only: If True, skip Pinecone and save locally only
        output_dir: Directory for output files
    """
    print("=" * 60)
    print("RAG Ingestion Pipeline")
    print("=" * 60)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Extract text from PDF
    print("\n[Step 1/5] Extracting text from PDF...")
    pages = extract_text_from_pdf(pdf_path)
    
    if not pages:
        print("ERROR: No text extracted from PDF")
        return
    
    # Step 2: Clean and chunk text
    print("\n[Step 2/5] Cleaning and chunking text...")
    all_chunks = []
    source_name = os.path.basename(pdf_path)
    
    for page_num, page_text in tqdm(pages, desc="Processing pages"):
        # Clean the text
        cleaned_text = clean_text(page_text)
        
        if not cleaned_text.strip():
            continue
        
        # Chunk the text
        page_chunks = chunk_text(
            text=cleaned_text,
            page_number=page_num,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            source=source_name
        )
        
        all_chunks.extend(page_chunks)
    
    print(f"Created {len(all_chunks)} chunks from {len(pages)} pages")
    
    if not all_chunks:
        print("ERROR: No chunks created")
        return
    
    # Step 3: Load embedding model
    print("\n[Step 3/5] Loading embedding model...")
    embedding_model = load_embedding_model()
    
    # Step 4: Generate embeddings
    print("\n[Step 4/5] Generating embeddings...")
    chunks_with_embeddings = generate_embeddings(all_chunks, embedding_model)
    
    # Step 5: Store vectors
    print("\n[Step 5/5] Storing vectors...")
    
    if local_only:
        # Save to local files only
        print("Running in LOCAL-ONLY mode (no Pinecone)")
        
        # Save chunks to JSONL (without embeddings for smaller file)
        chunks_file = os.path.join(output_dir, "chunks.jsonl")
        save_chunks_to_jsonl(chunks_with_embeddings, chunks_file, include_embeddings=False)
        
        # Save to local vector store
        local_store = LocalVectorStore(dimension=EMBEDDING_DIM)
        local_store.upsert(chunks_with_embeddings)
        
        # Save vectors to file for later use
        vectors_file = os.path.join(output_dir, "vectors.json")
        local_store.save_to_file(vectors_file)
        
        print(f"\nLocal files saved to {output_dir}/")
        
    else:
        # Upsert to Pinecone
        api_key = os.getenv("PINECONE_API_KEY")
        
        if not api_key:
            print("ERROR: PINECONE_API_KEY not set. Use --local-only to run without Pinecone.")
            # Fall back to local only
            print("Falling back to local-only mode...")
            chunks_file = os.path.join(output_dir, "chunks.jsonl")
            save_chunks_to_jsonl(chunks_with_embeddings, chunks_file, include_embeddings=False)
            return
        
        # Initialize Pinecone vector store
        vector_store = PineconeVectorStore(
            api_key=api_key,
            index_name=index_name,
            namespace=namespace,
            dimension=EMBEDDING_DIM
        )
        
        # Create index if needed
        if not vector_store.create_index_if_missing():
            print("ERROR: Failed to create/connect to Pinecone index")
            return
        
        # Upsert vectors
        upserted = vector_store.upsert(chunks_with_embeddings)
        
        # Also save chunks locally as backup
        chunks_file = os.path.join(output_dir, "chunks.jsonl")
        save_chunks_to_jsonl(chunks_with_embeddings, chunks_file, include_embeddings=False)
        
        # Print stats
        stats = vector_store.get_index_stats()
        print(f"\nPinecone index stats: {stats}")
    
    print("\n" + "=" * 60)
    print("Ingestion complete!")
    print("=" * 60)
    print(f"- Total chunks: {len(chunks_with_embeddings)}")
    print(f"- Chunks file: {os.path.join(output_dir, 'chunks.jsonl')}")
    if not local_only:
        print(f"- Pinecone index: {index_name}")
        print(f"- Namespace: {namespace}")
    print("=" * 60)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Ingest PDF into vector store for RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Ingest to Pinecone (requires PINECONE_API_KEY env var)
    python app/ingest.py --pdf ./data/Ebook-Agentic-AI.pdf --index agentic-ai
    
    # Local-only mode (no Pinecone needed)
    python app/ingest.py --pdf ./data/Ebook-Agentic-AI.pdf --local-only
    
    # Custom chunk size
    python app/ingest.py --pdf ./data/Ebook-Agentic-AI.pdf --chunk-size 400 --overlap 75
        """
    )
    
    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Path to the PDF file to ingest"
    )
    
    parser.add_argument(
        "--index",
        type=str,
        default="agentic-ai-ebook",
        help="Pinecone index name (default: agentic-ai-ebook)"
    )
    
    parser.add_argument(
        "--namespace",
        type=str,
        default="agentic-ai",
        help="Pinecone namespace (default: agentic-ai)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Target chunk size in tokens (default: 500)"
    )
    
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Chunk overlap in tokens (default: 50)"
    )
    
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Run without Pinecone, save vectors locally"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Output directory for local files (default: ./data)"
    )
    
    args = parser.parse_args()
    
    # Validate PDF path
    if not os.path.exists(args.pdf):
        print(f"ERROR: PDF file not found: {args.pdf}")
        sys.exit(1)
    
    # Run ingestion
    run_ingestion(
        pdf_path=args.pdf,
        index_name=args.index,
        namespace=args.namespace,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        local_only=args.local_only,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
