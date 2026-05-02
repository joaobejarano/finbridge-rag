"""
config.py — Centralized settings for CreditLens RAG

All constants live here. When you want to add a company
or change an accounting line item, update only this file.
"""

# =============================================================
# Brazilian companies (CVM)
# Format: cvm_code: (ticker, name, sector)
# To find the CVM code for other companies:
# https://dados.cvm.gov.br/dataset/cia_aberta-cad
# =============================================================
 
EMPRESAS = {
    9512: ("PETR4", "Petrobras", "Energy"),
    4170: ("VALE3", "Vale", "Mining"),
    24910: ("AZUL4", "Azul", "Aviation"),
    17264: ("MGLU3", "Magazine Luiza", "Retail"),
    5258: ("WEGE3", "WEG", "Industrial"),
}
 
# =============================================================
# Years to download
# =============================================================
 
ANOS = [2023, 2022]
 
# =============================================================
# CVM financial statements
# Format: key: (filename_in_zip, display_name)
# =============================================================
 
DEMONSTRATIVOS = {
    "BPA": ("dfp_cia_aberta_BPA_con_{year}.csv", "Balance Sheet — Assets"),
    "BPP": ("dfp_cia_aberta_BPP_con_{year}.csv", "Balance Sheet — Liabilities"),
    "DRE": ("dfp_cia_aberta_DRE_con_{year}.csv", "Income Statement"),
    "DFC": ("dfp_cia_aberta_DFC_MI_con_{year}.csv", "Cash Flow Statement"),
}
 
# =============================================================
# Accounting line items relevant to credit risk
# Format: account_code: description
#
# Grouped by statement type to make filtering easier
# =============================================================
 
CONTAS_POR_DEMO = {
    "BPA": {
        "1": "Total Assets",
        "1.01": "Current Assets",
        "1.01.01": "Cash and Cash Equivalents",
        "1.02": "Non-Current Assets",
    },
    "BPP": {
        "2": "Total Liabilities",
        "2.01": "Current Liabilities",
        "2.01.04": "Short-Term Loans and Financing",
        "2.02": "Non-Current Liabilities",
        "2.02.01": "Long-Term Loans and Financing",
        "2.03": "Equity",
    },
    "DRE": {
        "3.01": "Net Revenue",
        "3.05": "Operating Income (EBIT)",
        "3.06": "Financial Result",
        "3.06.02": "Financial Expenses",
        "3.11": "Net Income/Loss for the Period",
    },
    "DFC": {
        "6.01": "Cash Generated from Operations",
        "6.02": "Cash Used in Investing Activities",
        "6.03": "Cash from Financing Activities",
    },
}
 
# =============================================================
# CVM URLs
# =============================================================
 
CVM_DFP_URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/dfp_cia_aberta_{year}.zip"
 
# =============================================================
# ChromaDB
# =============================================================
 
CHROMA_DIR = "./chroma_db"
CHROMA_COLLECTION = "credit_risk"
 
# =============================================================
# OpenAI
# =============================================================
 
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0
 
# =============================================================
# RAG
# =============================================================

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVER_K = 4  # number of chunks returned per search
