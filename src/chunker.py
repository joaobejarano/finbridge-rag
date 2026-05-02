"""
src/chunker.py — Split Documents into smaller chunks

Matches Step 5 of the notebook.

Applied concepts:
  - LangChain docs: RecursiveCharacterTextSplitter tries to split
    by paragraph, then sentence, then character
  - Chip Huyen (Ch. 6): chunks should be coherent semantic units
  - RAG From Scratch (videos 2-3): chunk_size and overlap are the
    most important parameters

Usage:
    from src.chunker import chunk_documents
    chunks = chunk_documents(documents)
"""
 
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
 
from config import CHUNK_SIZE, CHUNK_OVERLAP
 
 
def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split Documents into smaller chunks.

    Metadata is preserved automatically by LangChain —
    each chunk inherits the original Document metadata.

    For CVM tabular data (short texts), many Documents
    will become 1-2 chunks. That is expected. When you add
    longer texts (Risk Factors, MD&A), there will be more chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " "],
    )
 
    chunks = splitter.split_documents(documents)
 
    print(f"\n📄 {len(documents)} documents → {len(chunks)} chunks")
    print(f"   chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
 
    return chunks
 
 
# =============================================================
# Standalone test
# =============================================================
 
if __name__ == "__main__":
    # Create test documents to verify chunking
    test_docs = [
        Document(
            page_content="Petrobras (PETR4) — Income Statement — Fiscal Year 2023:\n\n"
            "  Net Revenue: R$ 511,000,000 thousand\n"
            "  EBIT: R$ 189,000,000 thousand\n"
            "  Financial Expenses: R$ -32,000,000 thousand\n"
            "  Net Income: R$ 125,000,000 thousand\n\n"
            "Calculated Credit Risk Indicators:\n"
            "  Interest Coverage: 5.91x",
            metadata={"ticker": "PETR4", "section": "dre", "fiscal_year": 2023},
        )
    ]
 
    chunks = chunk_documents(test_docs)

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Metadata: {chunk.metadata}")
        print(f"Text: {chunk.page_content}")
 
