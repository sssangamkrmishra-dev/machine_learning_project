import streamlit as st
import numpy as np
import pandas as pd
import requests
import joblib
import json
import re
import os

# ============================================================
# LOAD ALL MODELS + FEATURE NAMES
# ============================================================

FEATURES_PATH = "models/feature_names.json"
NUMERIC_PATH = "models/numeric_cols.json"

with open(FEATURES_PATH, "r") as f:
    FEATURE_NAMES = json.load(f)

with open(NUMERIC_PATH, "r") as f:
    NUMERIC_COLS = json.load(f)

# Load individual models
models_info = []
for fname in os.listdir("models"):
    if fname.endswith(".joblib") and fname not in ["best_model.joblib"]:
        model = joblib.load(f"models/{fname}")
        models_info.append({
            "name": fname.replace(".joblib", ""),
            "model": model
        })

# Load final best model
best_model = joblib.load("models/best_model.joblib")


# ============================================================
# UTILITIES
# ============================================================

def clean_tag(text):
    """Normalize tag text for comparison."""
    return text.replace("'", "").strip().lower()


def fetch_cf_problem(contest_id, index):
    """Fetch problem from Codeforces API."""
    url = "https://codeforces.com/api/problemset.problems"
    resp = requests.get(url)
    data = resp.json()

    if data["status"] != "OK":
        return None

    for p in data["result"]["problems"]:
        if str(p.get("contestId")) == str(contest_id) and p.get("index") == index.upper():
            return p

    return None


def build_feature_vector(problem):
    """Builds 1-row DF matching training features exactly."""
    row = {c: 0 for c in FEATURE_NAMES}

    # contestId
    if "contestId" in problem:
        row["contestId"] = int(problem.get("contestId"))

    # index_num
    idx = problem.get("index", "")
    if isinstance(idx, str) and idx:
        row["index_num"] = ord(idx[0].upper()) - 64

    # points
    pts = problem.get("points", 0)
    try:
        row["points"] = float(pts) if pts else 0.0
    except:
        row["points"] = 0.0

    # clean problem tags  
    tags_clean = set(clean_tag(t) for t in problem.get("tags", []))

    # match EXACT columns
    for col in FEATURE_NAMES:
        if clean_tag(col) in tags_clean:
            row[col] = 1

    return pd.DataFrame([row])


def round_to_100(x):
    return int(((x + 50) // 100) * 100)


# ============================================================
# PREDICT WITH ALL MODELS
# ============================================================

def predict_with_all_models(row):
    results = []

    for entry in models_info:
        name = entry["name"]
        est = entry["model"]

        # Polynomial models need only numeric features
        if "Polynomial" in name or "GaussianProcess" in name:
            X = row[NUMERIC_COLS]
        else:
            X = row

        raw = est.predict(X)[0]
        rounded = round_to_100(raw)

        results.append({
            "model": name,
            "raw_prediction": float(raw),
            "rounded_prediction": rounded
        })

    return pd.DataFrame(results).sort_values("rounded_prediction").reset_index(drop=True)


def predict_best_model(row):
    """Predict using the final best model."""
    if hasattr(best_model, "named_steps") and "poly" in best_model.named_steps:
        X = row[NUMERIC_COLS]
    else:
        X = row

    raw = best_model.predict(X)[0]
    rounded = round_to_100(raw)
    return raw, rounded


# ============================================================
# STREAMLIT UI
# ============================================================

st.title(" Codeforces Rating Predictor – All Models + Best Model")

contest_id = st.number_input("Contest ID", value=1000, min_value=1)
index = st.text_input("Problem Index (A, B, C...)", "A")

if st.button("Predict Rating"):

    with st.spinner("Fetching problem from Codeforces..."):
        prob = fetch_cf_problem(contest_id, index)

    if prob is None:
        st.error(" Problem not found! Check contest ID and index.")
        st.stop()

    st.success(f"Problem Found: **{prob.get('name')}**")

    # Build feature vector
    row = build_feature_vector(prob)

    st.write("###  Extracted Features (non-zero only)")
    st.dataframe(row.loc[:, row.iloc[0] != 0])

    # Predictions from all models
    with st.spinner("Running predictions for all models..."):
        df_all = predict_with_all_models(row)

    st.subheader(" Predictions from ALL Models")
    st.dataframe(df_all)

    # Best model prediction
    raw, rounded = predict_best_model(row)

    st.subheader(" Best Model Final Prediction")
    st.write(f"**Raw:** `{raw:.2f}`")
    st.write(f"**Rounded:** `{rounded}`")
    st.info("Rounded rating matches Codeforces rating buckets (e.g., 800, 900, 1000, ...).")

