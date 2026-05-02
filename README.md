# 🔍 CreditLens RAG

A credit risk analysis assistant for publicly traded Brazilian companies, using RAG (Retrieval-Augmented Generation) with real CVM data.

## What It Does

Answers credit risk questions about companies listed on B3, such as:
- "What is Petrobras's debt level?"
- "Does Azul have liquidity risk?"
- "Compare Petrobras's debt with Vale's"

## Stack

- **LLM:** OpenAI GPT-4o-mini
- **Framework:** LangChain
- **Vector Store:** ChromaDB
- **Embeddings:** OpenAI text-embedding-3-small
- **Data:** CVM Open Data Portal (DFP)
- **Interface:** Streamlit

## Setup

```bash
# 1. Clone
git clone https://github.com/seu-usuario/creditlens-rag.git
cd creditlens-rag

# 2. Virtual environment
python -m venv venv
source venv/bin/activate

# 3. Dependencies
pip install -r requirements.txt

# 4. API Key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Run
streamlit run app.py
```

On the first run, click "Reload CVM Data" in the sidebar to download the data and build the vector database.

## Structure

```
creditlens-rag/
├── config.py             # Constants (companies, accounts, parameters)
├── src/
│   ├── loader.py         # Downloads and processes CVM data
│   ├── chunker.py        # Splits documents into chunks
│   ├── vectorstore.py    # Creates/loads ChromaDB
│   └── rag_chain.py      # Prompt + RAG chain
├── app.py                # Streamlit interface
├── requirements.txt
└── notebooks/
    └── 01_primeiro_rag.ipynb
```

## Available Companies

| Ticker | Company | Sector |
|--------|---------|-------|
| PETR4 | Petrobras | Energy |
| VALE3 | Vale | Mining |
| AZUL4 | Azul | Aviation |
| MGLU3 | Magazine Luiza | Retail |
| WEGE3 | WEG | Industrial |

## How It Works

1. The **loader** downloads CVM financial statements (DFP)
2. Tabular data is converted into **narrative text** with calculated indicators
3. Texts are split into **chunks** and transformed into **embeddings**
4. Embeddings are stored in **ChromaDB**
5. When the user asks a question, the **retriever** fetches the most relevant chunks
6. The **LLM** generates the answer using only the retrieved context

## Next Steps

- Expand coverage to more companies, sectors, and fiscal years
- Add richer data sources such as earnings releases, risk factors, and MD&A sections
- Improve answer grounding with explicit source citations and retrieved-chunk previews in the UI
- Add evaluation and regression tests for retrieval quality and answer accuracy
- Introduce metadata filtering and reranking to improve comparisons and company-specific queries
- Automate CVM data refreshes and vector database rebuilds
