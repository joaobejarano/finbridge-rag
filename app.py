"""
app.py — Streamlit interface for CreditLens RAG
 
Run with: streamlit run app.py
 
This file is the project entry point.
It imports everything from the modules in `src/` and builds the interface.
"""
 
import streamlit as st
from dotenv import load_dotenv
import os
 
# Load API key
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    load_dotenv()
 
# Import project modules
from src.loader import load_all_companies
from src.chunker import chunk_documents
from src.vectorstore import create_vectorstore, load_vectorstore, vectorstore_exists
from src.rag_chain import create_chain, ask
from config import EMPRESAS
 
 
# =============================================================
# Page configuration
# =============================================================
 
st.set_page_config(
    page_title="CreditLens RAG",
    page_icon="🔍",
    layout="wide",
)
 
st.title("🔍 CreditLens RAG")
st.caption("Credit Risk Analysis for Publicly Traded Brazilian Companies")
 
 
# =============================================================
# Sidebar
# =============================================================
 
with st.sidebar:
    st.header("⚙️ Configuration")
 
    # Button to rebuild the database
    if st.button("🔄 Reload CVM Data", use_container_width=True):
        with st.spinner("Downloading CVM data..."):
            documents = load_all_companies()
            st.session_state["doc_count"] = len(documents)
 
        with st.spinner("Splitting into chunks..."):
            chunks = chunk_documents(documents)
 
        with st.spinner("Creating embeddings..."):
            vectorstore = create_vectorstore(chunks)
            st.session_state["vectorstore"] = vectorstore
            st.session_state["chain"] = create_chain(vectorstore)
 
        st.success(f"✅ {len(documents)} documents loaded!")
 
    st.divider()
 
    # Available companies info
    st.subheader("📊 Available Companies")
    for cod_cvm, (ticker, name, sector) in EMPRESAS.items():
        st.text(f"{ticker} — {name} ({sector})")
 
    st.divider()
 
    # Example questions
    st.subheader("💡 Example Questions")
    examples = [
        "What is Petrobras's debt level?",
        "Does Azul have liquidity risk?",
        "Compare Petrobras's debt with Vale's",
        "Which company has better interest coverage?",
        "Does Magazine Luiza generate operating cash flow?",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["example_question"] = ex
 
 
# =============================================================
# Initialization: load an existing vectorstore or create one
# =============================================================
 
if "chain" not in st.session_state:
    if vectorstore_exists():
        with st.spinner("Loading database..."):
            vectorstore = load_vectorstore()
            st.session_state["vectorstore"] = vectorstore
            st.session_state["chain"] = create_chain(vectorstore)
    else:
        st.info(
            "👈 Click **Reload CVM Data** in the sidebar "
            "to download the data and build the database."
        )
        st.stop()
 
 
# =============================================================
# Chat history
# =============================================================
 
if "messages" not in st.session_state:
    st.session_state["messages"] = []
 
# Show previous messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
 
 
# =============================================================
# User input
# =============================================================
 
# Check whether an example question was selected
question = None
if "example_question" in st.session_state:
    question = st.session_state.pop("example_question")
 
# Chat input
user_input = st.chat_input("Ask a question about credit risk...")
 
# Use the example question or the user's input
if question is None:
    question = user_input
 
if question:
    # Show the user's question
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
 
    # Generate the answer
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            chain = st.session_state["chain"]
            response = ask(chain, question)
            st.write(response)
 
    # Save the answer to the history
    st.session_state["messages"].append({"role": "assistant", "content": response})
