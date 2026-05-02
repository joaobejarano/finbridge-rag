"""
src/vectorstore.py — Create and load ChromaDB with embeddings
 
Matches Step 6 of the notebook.
 
Applied concepts:
  - LangChain docs: Chroma.from_documents() creates and stores embeddings
  - Chip Huyen (Ch. 6 — Retrieval Algorithms): semantic similarity
    search using vector distance
  - RAG From Scratch (video 2): embeddings capture meaning
 
This module has two main functions:
  - create_vectorstore(): create from scratch (first run)
  - load_vectorstore(): load from disk (later runs)
 
Usage:
    from src.vectorstore import create_vectorstore, load_vectorstore
 
    # First run: create
    vectorstore = create_vectorstore(chunks)
 
    # Later runs: load from disk
    vectorstore = load_vectorstore()
"""
 
import os
from typing import List
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
 
from config import CHROMA_DIR, CHROMA_COLLECTION, EMBEDDING_MODEL
 
 
def _get_embeddings():
    """Return the configured embeddings model."""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)
 
 
def create_vectorstore(chunks: List[Document]) -> Chroma:
    """
    Create ChromaDB from the chunks.
 
    This does two things:
    1. Calls the OpenAI API to generate an embedding for each chunk
    2. Stores everything in the local ChromaDB database (`chroma_db/`)
 
    Cost: `text-embedding-3-small` costs about $0.02 per 1M tokens.
    For 5 companies x 2 years, it should cost less than $0.01.
    """
    # Remove the old database if it exists
    if os.path.exists(CHROMA_DIR):
        import shutil
        shutil.rmtree(CHROMA_DIR)
        print(f"  🗑️ Old ChromaDB removed")
 
    print(f"\n🔄 Creating embeddings for {len(chunks)} chunks...")
    print(f"   Model: {EMBEDDING_MODEL}")
 
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=_get_embeddings(),
        persist_directory=CHROMA_DIR,
        collection_name=CHROMA_COLLECTION,
    )
 
    print(f"✅ ChromaDB created at {CHROMA_DIR}")
    return vectorstore
 
 
def load_vectorstore() -> Chroma:
    """
    Load ChromaDB from disk without downloading or embedding again.
 
    Use this when the database has already been created.
    """
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            f"ChromaDB not found at {CHROMA_DIR}. "
            f"Run create_vectorstore() first."
        )
 
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=_get_embeddings(),
        collection_name=CHROMA_COLLECTION,
    )
 
    print(f"✅ ChromaDB loaded from {CHROMA_DIR}")
    return vectorstore
 
 
def vectorstore_exists() -> bool:
    """Check whether ChromaDB has already been created."""
    return os.path.exists(CHROMA_DIR)
 
 
# =============================================================
# Standalone test
# =============================================================
 
if __name__ == "__main__":
    # Test: create with dummy data and run a search
    test_chunks = [
        Document(
            page_content="Petrobras has short-term debt of R$ 24 billion",
            metadata={"ticker": "PETR4", "section": "bpp"},
        ),
        Document(
            page_content="Petrobras reported R$ 511 billion in net revenue",
            metadata={"ticker": "PETR4", "section": "dre"},
        ),
    ]
 
    vs = create_vectorstore(test_chunks)
 
    # Test retrieval
    results = vs.similarity_search("Petrobras debt level", k=2)
    for r in results:
        print(f"  [{r.metadata['section']}] {r.page_content}")
