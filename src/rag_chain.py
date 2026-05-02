"""
src/rag_chain.py — Prompt template + RAG chain
 
Matches Steps 7, 8, and 9 of the notebook.
 
Applied concepts:
  - LangChain docs: RetrievalQA or an LCEL chain
  - Chip Huyen (Ch. 6 — RAG Architecture): Retrieval + Generation
  - RAG From Scratch (video 1): Query → Retrieve → Augment → Generate
 
Usage:
    from src.rag_chain import create_chain, ask, debug_retrieval
 
    chain = create_chain(vectorstore)
    answer = ask(chain, "What is Petrobras's debt?")
    debug_retrieval(vectorstore, "What is Petrobras's debt?")
"""
 
from typing import List, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
 
from config import LLM_MODEL, LLM_TEMPERATURE, RETRIEVER_K
 
 
# =============================================================
# Credit risk prompt
# =============================================================
 
SYSTEM_PROMPT = """You are a credit risk analyst specialized in publicly traded Brazilian companies.

Answer the question in English using ONLY the information from the context below.

Rules:
- Always cite the company, period, and source (DFP/CVM)
- Use the calculated indicators when available
- If the question asks for a comparison, organize the answer in clear bullet points
- If there is not enough information in the context, say so clearly
- Values are reported in thousands of Brazilian reais (R$ thousand), following the CVM standard

Context:
{context}

Question: {question}

Answer:"""
 
 
# =============================================================
# Step 7: Build the chain
# =============================================================
 
def create_chain(vectorstore: Chroma):
    """
    Build the RAG chain by connecting retriever → prompt → LLM.
 
    Flow (RAG From Scratch, video 1):
      Question
        → retriever fetches the top-k most relevant chunks
        → format_docs joins the chunk texts
        → prompt assembles: context + question
        → LLM generates the answer
        → StrOutputParser extracts the text
    """
    # LLM
    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
 
    # Retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": RETRIEVER_K}
    )
 
    # Prompt
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
 
    # Chain (LCEL)
    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
 
    return chain
 
 
def _format_docs(docs: List[Document]) -> str:
    """Format retrieved chunks into a single text block for the prompt."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)
 
 
# =============================================================
# Step 8: Ask a question
# =============================================================
 
def ask(chain, question: str) -> str:
    """
    Ask a question to the RAG pipeline and return the answer.
 
    Simple usage:
        answer = ask(chain, "What is Petrobras's debt?")
        print(answer)
    """
    return chain.invoke(question)
 
 
# =============================================================
# Step 9: Debug — inspect what the retriever is returning
# =============================================================
 
def debug_retrieval(vectorstore: Chroma, question: str) -> List[Document]:
    """
    Show which chunks the retriever would return for a question.
 
    Chip Huyen (Ch. 6 — Retrieval Optimization):
    Before optimizing, understand what is being retrieved.
    If the chunks are wrong, the problem is in retrieval,
    not in the LLM.
 
    Usage:
        docs = debug_retrieval(vectorstore, "short-term debt")
        # Prints each chunk with its metadata
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
    docs = retriever.invoke(question)
 
    print(f"\n🔍 Question: {question}")
    print(f"   Chunks found: {len(docs)}")
 
    for i, doc in enumerate(docs):
        print(f"\n--- Chunk {i+1} ---")
        print(f"  Company: {doc.metadata.get('ticker', '?')}")
        print(f"  Section: {doc.metadata.get('section', '?')}")
        print(f"  Year:    {doc.metadata.get('fiscal_year', '?')}")
        print(f"  Text:\n{doc.page_content}")
 
    return docs
 
 
# =============================================================
# Standalone test
# =============================================================
 
if __name__ == "__main__":
    from vectorstore import load_vectorstore
 
    vs = load_vectorstore()
    chain = create_chain(vs)
 
    questions = [
        "What was Petrobras's debt level in 2023?",
        "Does Petrobras generate enough operating cash flow?",
        "What is Petrobras's current ratio?",
    ]
 
    for p in questions:
        print(f"\n{'─'*50}")
        print(f"❓ {p}")
        print(f"{'─'*50}")
        answer = ask(chain, p)
        print(f"\n💬 {answer}")
