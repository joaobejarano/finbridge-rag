"""
src/loader.py — Load CVM data and convert it into LangChain Documents
 
Matches Steps 1 through 4 of the notebook:
  Step 1: Download the CVM ZIP file
  Step 2: Filter company + fiscal year
  Step 3: Convert the table into narrative text
  Step 4: Create Documents with metadata
 
Usage:
    from src.loader import load_all_companies
    documents = load_all_companies()
"""
 
import zipfile
import requests
import pandas as pd
from io import BytesIO
from typing import List, Dict, Optional
from langchain_core.documents import Document
 
import sys
sys.path.append("..")
from config import (
    EMPRESAS, ANOS, DEMONSTRATIVOS,
    CONTAS_POR_DEMO, CVM_DFP_URL,
)
 
 
# =============================================================
# Step 1: Download CVM data
# =============================================================
 
# Cache: store downloaded ZIP files so they are not fetched twice
_zip_cache: Dict[int, zipfile.ZipFile] = {}
 
 
def download_dfp(year: int) -> Optional[zipfile.ZipFile]:
    """
    Download the DFP ZIP for a given year and return the ZipFile object.
    Uses caching to avoid downloading the same year twice.
    """
    if year in _zip_cache:
        print(f"  [cache] DFP {year} already downloaded")
        return _zip_cache[year]
 
    url = CVM_DFP_URL.format(year=year)
    print(f"  Downloading DFP {year}...")
 
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  ⚠️ Error downloading DFP {year}: {e}")
        return None
 
    z = zipfile.ZipFile(BytesIO(response.content))
    _zip_cache[year] = z
    print(f"  ✓ DFP {year} downloaded ({len(z.namelist())} files)")
    return z
 
 
# =============================================================
# Step 2: Filter company data
# =============================================================
 
def filter_company(
    df: pd.DataFrame,
    cod_cvm: int,
    contas: List[str],
) -> pd.DataFrame:
    """
    Filter a company's data and return only the relevant accounts.
 
    Cleanup rules:
    - ORDEM_EXERC == "ÚLTIMO" (removes prior-year comparative rows)
    - Most recent version of the document
    - Only accounts listed in CONTAS_POR_DEMO
    """
    # Filter the company
    df_filtered = df[df["CD_CVM"] == cod_cvm].copy()
 
    if df_filtered.empty:
        return df_filtered
 
    # Most recent fiscal-year ordering
    df_filtered = df_filtered[df_filtered["ORDEM_EXERC"] == "ÚLTIMO"]
 
    # Most recent version (if there were restatements)
    if "VERSAO" in df_filtered.columns and not df_filtered.empty:
        max_version = df_filtered["VERSAO"].max()
        df_filtered = df_filtered[df_filtered["VERSAO"] == max_version]
 
    # Only accounts relevant to credit analysis
    df_filtered = df_filtered[df_filtered["CD_CONTA"].isin(contas)]
 
    return df_filtered
 
 
# =============================================================
# Step 3: Convert the table into narrative text
# =============================================================
 
def to_narrative(
    df: pd.DataFrame,
    company_name: str,
    ticker: str,
    demo_name: str,
    account_labels: Dict[str, str],
    year: int,
) -> str:
    """
    Convert a filtered DataFrame into narrative text.
 
    This is the "RAG Beyond Texts" step (Chip Huyen, Ch. 6):
    tabular data becomes text that the LLM can understand.
    """
    if df.empty:
        return ""
 
    lines = [f"{company_name} ({ticker}) — {demo_name} — Fiscal Year {year}:", ""]

    for _, row in df.iterrows():
        conta = account_labels.get(row["CD_CONTA"], row["DS_CONTA"])
        valor = row["VL_CONTA"]
 
        if pd.notna(valor):
            valor_fmt = f"R$ {valor:,.0f} thousand"
        else:
            valor_fmt = "N/A"
 
        lines.append(f"  {conta}: {valor_fmt}")
 
    # Calculate credit indicators when possible
    indicators = _calculate_indicators(df)
    if indicators:
        lines.append("")
        lines.append("Calculated Credit Risk Indicators:")
        for name, value in indicators.items():
            lines.append(f"  {name}: {value}")
 
    return "\n".join(lines)
 
 
def _calculate_indicators(df: pd.DataFrame) -> Dict[str, str]:
    """
    Calculate credit indicators from the available data.
    Only computes them when the required accounts are present.
    """
    indicators = {}
 
    def get(cd_conta: str) -> Optional[float]:
        rows = df[df["CD_CONTA"] == cd_conta]
        if not rows.empty and pd.notna(rows.iloc[0]["VL_CONTA"]):
            return rows.iloc[0]["VL_CONTA"]
        return None
 
    # Current ratio
    ativo_circ = get("1.01")
    passivo_circ = get("2.01")
    if ativo_circ and passivo_circ and passivo_circ != 0:
        indicators["Current Ratio"] = f"{ativo_circ / passivo_circ:.2f}x"
 
    # Gross and net debt
    divida_cp = abs(get("2.01.04") or 0)
    divida_lp = abs(get("2.02.01") or 0)
    divida_bruta = divida_cp + divida_lp
 
    if divida_bruta > 0:
        indicators["Gross Debt"] = f"R$ {divida_bruta:,.0f} thousand"

        # Debt maturity profile
        pct_cp = (divida_cp / divida_bruta) * 100
        indicators["% Short-Term Debt"] = f"{pct_cp:.1f}%"
        indicators["% Long-Term Debt"] = f"{100 - pct_cp:.1f}%"
 
        # Net debt
        caixa = get("1.01.01")
        if caixa:
            divida_liq = divida_bruta - abs(caixa)
            indicators["Net Debt"] = f"R$ {divida_liq:,.0f} thousand"
 
        # Debt / equity
        pl = get("2.03")
        if pl and pl != 0:
            indicators["Gross Debt / Equity"] = f"{divida_bruta / abs(pl):.2f}x"
 
    # Interest coverage
    ebit = get("3.05")
    desp_fin = get("3.06.02")
    if ebit and desp_fin and desp_fin != 0:
        indicators["Interest Coverage"] = f"{abs(ebit) / abs(desp_fin):.2f}x"
 
    return indicators
 
 
# =============================================================
# Step 4: Create LangChain Documents
# =============================================================
 
def load_company(cod_cvm: int) -> List[Document]:
    """
    Load all data for a company and return Documents.
 
    Combines Steps 1 through 4 into a single function:
    download → filter → convert → package.
    """
    ticker, company_name, sector = EMPRESAS[cod_cvm]
    documents = []
 
    for year in ANOS:
        z = download_dfp(year)
        if z is None:
            continue
 
        available_files = z.namelist()
 
        # For each statement type (BPA, BPP, DRE, DFC)
        for demo_key, (filename_template, demo_name) in DEMONSTRATIVOS.items():
            filename = filename_template.format(year=year)
 
            if filename not in available_files:
                continue
 
            # Load the CSV
            df = pd.read_csv(
                z.open(filename),
                sep=";",
                encoding="latin-1",
                dtype={"CD_CONTA": str},
            )
 
            # Filter company + relevant accounts
            account_labels = CONTAS_POR_DEMO.get(demo_key, {})
            contas = list(account_labels.keys())
            df_filtered = filter_company(df, cod_cvm, contas)
 
            if df_filtered.empty:
                continue
 
            # Convert to narrative text
            narrative = to_narrative(
                df_filtered,
                company_name,
                ticker,
                demo_name,
                account_labels,
                year,
            )
 
            if not narrative:
                continue
 
            # Create the Document
            doc = Document(
                page_content=narrative,
                metadata={
                    "company": company_name,
                    "ticker": ticker,
                    "cod_cvm": cod_cvm,
                    "country": "BR",
                    "currency": "BRL",
                    "document_type": "DFP",
                    "section": demo_key.lower(),
                    "fiscal_year": year,
                    "content_type": "quantitative",
                    "sector": sector,
                    "source": "CVM",
                    "language": "en",
                },
            )
            documents.append(doc)
 
    return documents
 
 
def load_all_companies() -> List[Document]:
    """
    Load data for ALL companies configured in config.py.
    Returns a list of Documents ready for chunking.
    """
    all_docs = []
 
    for cod_cvm in EMPRESAS:
        ticker = EMPRESAS[cod_cvm][0]
        company = EMPRESAS[cod_cvm][1]
        print(f"\n{'='*50}")
        print(f"Loading {ticker} — {company}")
        print(f"{'='*50}")
 
        docs = load_company(cod_cvm)
        all_docs.extend(docs)
        print(f"  → {len(docs)} documents created")
 
    print(f"\n✅ Total: {len(all_docs)} documents from {len(EMPRESAS)} companies")
    return all_docs
 
 
# =============================================================
# Standalone test
# =============================================================
 
if __name__ == "__main__":
    # Quick test: load only Petrobras
    docs = load_company(9512)
 
    print(f"\n{len(docs)} documents created")
    for doc in docs:
        print(f"\n--- {doc.metadata['ticker']} | {doc.metadata['section']} | {doc.metadata['fiscal_year']} ---")
        print(doc.page_content[:300])
