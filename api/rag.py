# RAG module — embeds IBM HR dataset into ChromaDB for retrieval-augmented explanations
# Usage: import and call get_similar_cases(feature_dict, k=3)

import pathlib
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

_BASE       = pathlib.Path(__file__).parent.parent
_CSV        = _BASE / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
_STORE_PATH = str(_BASE / "rag_store")
_COLLECTION = "hr_employees"

# Key features to build a human-readable text representation for embedding
_TEXT_FIELDS = [
    "Age", "Department", "JobRole", "JobLevel", "MaritalStatus",
    "OverTime", "MonthlyIncome", "JobSatisfaction", "EnvironmentSatisfaction",
    "WorkLifeBalance", "YearsAtCompany", "TotalWorkingYears",
    "BusinessTravel", "DistanceFromHome", "StockOptionLevel",
]

_embed_model = SentenceTransformer("all-MiniLM-L6-v2")
_client      = chromadb.PersistentClient(path=_STORE_PATH)


def _row_to_text(row: pd.Series) -> str:
    """Convert a CSV row into a sentence for embedding."""
    ot    = row.get("OverTime", "No")
    jsat  = {1: "low", 2: "medium", 3: "high", 4: "very high"}.get(int(row.get("JobSatisfaction", 2)), "medium")
    wlb   = {1: "bad", 2: "good", 3: "better", 4: "best"}.get(int(row.get("WorkLifeBalance", 2)), "good")
    return (
        f"Age {row.get('Age')}, {row.get('Department')}, {row.get('JobRole')}, "
        f"job level {row.get('JobLevel')}, {row.get('MaritalStatus')}, "
        f"income ${int(row.get('MonthlyIncome', 0)):,}, overtime={'yes' if ot=='Yes' else 'no'}, "
        f"job satisfaction={jsat}, work-life balance={wlb}, "
        f"{row.get('YearsAtCompany')} years at company, "
        f"travel={row.get('BusinessTravel', 'Non-Travel')}"
    )


def build_index() -> int:
    """Build (or rebuild) the ChromaDB collection from the IBM HR CSV."""
    df = pd.read_csv(_CSV)

    # Delete existing collection if rebuilding
    try:
        _client.delete_collection(_COLLECTION)
    except Exception:
        pass

    col = _client.create_collection(
        name=_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    texts    = [_row_to_text(row) for _, row in df.iterrows()]
    embeddings = _embed_model.encode(texts, show_progress_bar=False).tolist()

    col.add(
        ids        = [str(row["EmployeeNumber"]) for _, row in df.iterrows()],
        embeddings = embeddings,
        documents  = texts,
        metadatas  = [
            {
                "attrition":     str(row["Attrition"]),
                "jobrole":       str(row["JobRole"]),
                "department":    str(row["Department"]),
                "income":        int(row["MonthlyIncome"]),
                "overtime":      str(row["OverTime"]),
                "maritalstatus": str(row["MaritalStatus"]),
                "yearsatcompany":int(row["YearsAtCompany"]),
            }
            for _, row in df.iterrows()
        ],
    )
    return len(df)


def _get_collection():
    """Return existing collection, building it if missing."""
    try:
        return _client.get_collection(_COLLECTION)
    except Exception:
        build_index()
        return _client.get_collection(_COLLECTION)


def get_similar_cases(feature_dict: dict, k: int = 3) -> list[dict]:
    """
    Given a raw employee feature dict (pre-preprocessing), retrieve k
    most similar historical employees with their actual attrition outcome.
    """
    # Build a text summary of the query employee
    # Map API field names → CSV-style values for text generation
    text = (
        f"Age {feature_dict.get('age')}, {feature_dict.get('department')}, "
        f"{feature_dict.get('jobrole')}, job level {feature_dict.get('joblevel')}, "
        f"{feature_dict.get('maritalstatus')}, "
        f"income ${int(feature_dict.get('monthlyincome', 0)):,}, "
        f"overtime={'yes' if feature_dict.get('overtime') == 'Yes' else 'no'}, "
        f"job satisfaction={feature_dict.get('jobsatisfaction')}, "
        f"work-life balance={feature_dict.get('worklifebalance')}, "
        f"{feature_dict.get('yearsatcompany')} years at company, "
        f"travel={feature_dict.get('businesstravel', 'Non-Travel')}"
    )

    query_emb = _embed_model.encode([text]).tolist()
    col       = _get_collection()
    results   = col.query(query_embeddings=query_emb, n_results=k)

    cases = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        cases.append({
            "similarity":    round(1 - dist, 3),        # cosine: 1-distance
            "outcome":       "LEFT" if meta["attrition"] == "Yes" else "STAYED",
            "jobrole":       meta["jobrole"],
            "department":    meta["department"],
            "income":        meta["income"],
            "overtime":      meta["overtime"],
            "maritalstatus": meta["maritalstatus"],
            "years_at_company": meta["yearsatcompany"],
            "description":   results["documents"][0][i],
        })
    return cases
