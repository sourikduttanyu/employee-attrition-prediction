# Employee Attrition Prediction API
# Run with: uvicorn api.main:app --reload

import csv
import io
import json
import os
import pathlib
import threading

import joblib
import numpy as np
import pandas as pd
import requests
import shap
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List

# ---------------------------------------------------------------------------
# Paths — resolve relative to this file so the server can be launched from
# any working directory.
# ---------------------------------------------------------------------------
_BASE = pathlib.Path(__file__).parent.parent
_ARTIFACTS = _BASE / "model_artifacts"

# ---------------------------------------------------------------------------
# Load artifacts once at startup
# ---------------------------------------------------------------------------
model = joblib.load(_ARTIFACTS / "xgb_model.joblib")
scaler = joblib.load(_ARTIFACTS / "scaler.joblib")

with open(_ARTIFACTS / "model_metadata.json") as f:
    metadata = json.load(f)

FEATURE_NAMES: list[str] = metadata["feature_names"]
METRICS: dict = metadata["metrics"]

explainer = shap.TreeExplainer(model)

# RAG — build index in background on first startup so it doesn't block requests
from api.rag import get_similar_cases, build_index as _build_rag_index

def _init_rag():
    import pathlib as _pl
    store = _pl.Path(__file__).parent.parent / "rag_store"
    if not store.exists():
        _build_rag_index()

threading.Thread(target=_init_rag, daemon=True).start()

# ---------------------------------------------------------------------------
# Categorical and ordinal columns (ordinal treated as get_dummies too)
# ---------------------------------------------------------------------------
_CATEGORICAL = [
    "businesstravel", "department", "educationfield",
    "gender", "jobrole", "maritalstatus", "overtime",
]
_ORDINAL = [
    "education", "environmentsatisfaction", "jobinvolvement", "joblevel",
    "jobsatisfaction", "numcompaniesworked", "percentsalaryhike",
    "performancerating", "relationshipsatisfaction", "stockoptionlevel",
    "trainingtimeslastyear", "worklifebalance", "yearsincurrentrole",
    "yearssincelastpromotion", "yearswithcurrmanager",
]
_DUMMIES_COLS = _CATEGORICAL + _ORDINAL

# ---------------------------------------------------------------------------
# Pydantic input model
# ---------------------------------------------------------------------------
class EmployeeInput(BaseModel):
    # Numerical
    age: int
    dailyrate: int
    distancefromhome: int
    employeenumber: int
    hourlyrate: int
    monthlyincome: int
    monthlyrate: int
    totalworkingyears: int
    yearsatcompany: int
    # Categorical
    businesstravel: str
    department: str
    educationfield: str
    gender: str
    jobrole: str
    maritalstatus: str
    overtime: str
    # Ordinal
    education: int
    environmentsatisfaction: int
    jobinvolvement: int
    joblevel: int
    jobsatisfaction: int
    numcompaniesworked: int
    percentsalaryhike: int
    performancerating: int
    relationshipsatisfaction: int
    stockoptionlevel: int
    trainingtimeslastyear: int
    worklifebalance: int
    yearsincurrentrole: int
    yearssincelastpromotion: int
    yearswithcurrmanager: int

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def _preprocess_df(df: pd.DataFrame) -> np.ndarray:
    """Vectorised preprocessing for one or many rows."""
    df = df.copy()
    for col in _ORDINAL:
        df[col] = df[col].astype(str)
    df = pd.get_dummies(df, columns=_DUMMIES_COLS)
    df = df.reindex(columns=FEATURE_NAMES, fill_value=0)
    return scaler.transform(df)

def preprocess(emp: EmployeeInput) -> np.ndarray:
    return _preprocess_df(pd.DataFrame([emp.model_dump()]))

# ---------------------------------------------------------------------------
# LLM explanation (best-effort)
# ---------------------------------------------------------------------------
def _llm_explain(risk_level: str, probability: float, top_factors: list[dict], retrieved_cases: list[dict] | None = None) -> str | None:
    factors_text = "; ".join(
        f"{f['feature']} ({f['direction'].replace('_', ' ')}, shap={f['shap_value']:.3f})"
        for f in top_factors[:3]
    )
    cases_text = ""
    if retrieved_cases:
        parts = []
        for c in retrieved_cases[:3]:
            parts.append(
                f"{c['jobrole']} in {c['department']} (income ${c['income']:,}, "
                f"overtime={c['overtime']}, {c['years_at_company']}yr) → {c['outcome']}"
            )
        cases_text = " Similar historical employees: " + "; ".join(parts) + "."
    prompt = (
        f"An employee has a {risk_level} attrition risk ({probability:.0%} probability). "
        f"Top contributing factors: {factors_text}.{cases_text} "
        "Write a 2-sentence explanation for an HR manager explaining what this means "
        "and what action to consider. Be concise and practical."
    )
    try:
        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        resp = requests.post(
            f"{ollama_host}/api/generate",
            json={"model": "llama3.2", "prompt": prompt, "stream": False},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip() or None
    except Exception:
        return None

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Employee Attrition Prediction API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": metadata["model"],
        "n_features": metadata["n_features"],
        "metrics": METRICS,
    }


@app.post("/predict")
def predict(emp: EmployeeInput):
    try:
        X = preprocess(emp)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {e}")

    proba = float(model.predict_proba(X)[0, 1])

    if proba > 0.6:
        risk_level = "HIGH"
    elif proba >= 0.3:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    prediction = "Attrition" if proba >= metadata["decision_threshold"] else "No Attrition"

    # SHAP on the preprocessed (scaled) array
    shap_values = explainer.shap_values(X)
    # For binary XGB, shap_values may be a list [neg, pos] or a single array
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    top_indices = np.argsort(np.abs(sv))[::-1][:5]
    shap_explanation = [
        {
            "feature": FEATURE_NAMES[i],
            "shap_value": float(sv[i]),
            "direction": "increases_risk" if sv[i] > 0 else "decreases_risk",
        }
        for i in top_indices
    ]

    retrieved_cases = get_similar_cases(emp.model_dump(), k=3)
    llm_explanation = _llm_explain(risk_level, proba, shap_explanation, retrieved_cases)

    return {
        "attrition_probability": round(proba, 4),
        "risk_level": risk_level,
        "prediction": prediction,
        "shap_explanation": shap_explanation,
        "similar_cases": retrieved_cases,
        "llm_explanation": llm_explanation,
    }


def _risk_label(p: float) -> str:
    return "HIGH" if p > 0.6 else "MEDIUM" if p >= 0.3 else "LOW"


@app.post("/predict/batch")
def predict_batch(employees: List[EmployeeInput]):
    """Vectorised batch prediction — no LLM (cost/latency). Up to 1000 rows."""
    if len(employees) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 employees per batch.")
    if not employees:
        raise HTTPException(status_code=400, detail="Empty batch.")

    rows = [e.model_dump() for e in employees]
    try:
        X = _preprocess_df(pd.DataFrame(rows))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {e}")

    probas = model.predict_proba(X)[:, 1]
    threshold = metadata["decision_threshold"]

    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        sv_matrix = shap_vals[1]
    else:
        sv_matrix = shap_vals

    results = []
    for i, (emp, proba) in enumerate(zip(employees, probas)):
        sv = sv_matrix[i]
        top_idx = np.argsort(np.abs(sv))[::-1][:3]
        results.append({
            "employee_number": emp.employeenumber,
            "attrition_probability": round(float(proba), 4),
            "risk_level": _risk_label(float(proba)),
            "prediction": "Attrition" if proba >= threshold else "No Attrition",
            "top_shap_factors": [
                {"feature": FEATURE_NAMES[j], "shap_value": round(float(sv[j]), 4)}
                for j in top_idx
            ],
        })

    return {"count": len(results), "predictions": results}


@app.post("/predict/batch/csv")
async def predict_batch_csv(file: UploadFile = File(...)):
    """Upload a CSV with employee columns → download CSV with predictions appended."""
    content = await file.read()
    try:
        df_in = pd.read_csv(io.BytesIO(content))
        df_in.columns = df_in.columns.str.lower().str.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV parse error: {e}")

    try:
        X = _preprocess_df(df_in)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {e}")

    probas = model.predict_proba(X)[:, 1]
    threshold = metadata["decision_threshold"]
    df_out = df_in.copy()
    df_out["attrition_probability"] = probas.round(4)
    df_out["risk_level"] = [_risk_label(float(p)) for p in probas]
    df_out["prediction"] = ["Attrition" if p >= threshold else "No Attrition" for p in probas]

    buf = io.StringIO()
    df_out.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=attrition_predictions.csv"},
    )


@app.post("/rag/rebuild")
def rag_rebuild():
    """Rebuild the ChromaDB RAG index from the IBM HR CSV."""
    try:
        n = _build_rag_index()
        return {"status": "ok", "indexed": n}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG rebuild failed: {e}")
