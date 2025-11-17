import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests

def get_contest_division(contestId: int):
    url = "https://codeforces.com/api/contest.list"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except:
        return None

    if data.get("status") != "OK":
        return None

    contest = next((c for c in data["result"] if c.get("id") == contestId), None)
    if not contest:
        return None

    name = contest.get("name", "")
    features = {
        'Div. 1': 1 if 'Div. 1' in name else 0,
        'Div. 2': 1 if 'Div. 2' in name else 0,
        'Div. 3': 1 if 'Div. 3' in name else 0,
        'Div. 4': 1 if 'Div. 4' in name else 0
    }
    features['unknown'] = 1 if sum(features.values()) == 0 else 0
    return features

st.title(" Codeforces User Rating Prediction")
st.write("Predict Codeforces user new rating based on contest performance using multiple ML models.")

dataset_choice = st.selectbox(
    "Select Dataset",
    ("dataset1", "dataset2")
)

MODEL_DIR = f"models_user_rating_prediction/{dataset_choice}"

if not os.path.exists(MODEL_DIR):
    st.error(f"Model directory not found: {MODEL_DIR}")
    st.stop()

model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]

if not model_files:
    st.error("No model files found. Train your models first.")
    st.stop()

st.sidebar.header("Available Models")
for f in model_files:
    st.sidebar.write("•", f)

best_model_file = None
for f in model_files:
    if f.startswith("best_model"):
        best_model_file = f
        break

if best_model_file is None:
    best_model_file = model_files[-1] 

best_model_path = os.path.join(MODEL_DIR, best_model_file)
best_model = joblib.load(best_model_path)
st.success(f"Loaded Best Model: {best_model_file}")

all_models = {}
for f in model_files:
    path = os.path.join(MODEL_DIR, f)
    try:
        all_models[f.replace(".pkl","")] = joblib.load(path)
    except:
        pass

TEMPLATE = f"data_for_user_rating_prediction/{dataset_choice.replace('dataset', 'dataset_')}.csv"
if os.path.exists(TEMPLATE):
    df_template = pd.read_csv(TEMPLATE)
    df_template = df_template.drop(columns=["contestName", "newRating", "ratingDelta"], errors="ignore")
    feature_columns = df_template.columns.tolist()
else:
    st.error("Template CSV not found. Cannot determine feature structure.")
    st.stop()


st.header(" Enter Contest Information")

old_rating = st.number_input("Old Rating", min_value=0, max_value=4000, value=1500)
contestId = st.number_input("Contest ID", min_value=1, value=1956)
rank = st.number_input("Rank", min_value=1, value=120)

if st.button("Predict Rating"):

    division = get_contest_division(int(contestId))
    if division is None:
        st.error("Invalid contest ID or API error.")
        st.stop()

    st.write("### Division Features:", division)

    row = {col: 0 for col in feature_columns}
    if "oldRating" in row:
        row["oldRating"] = old_rating
    if "contestId" in row:
        row["contestId"] = contestId
    if "rank" in row:
        row["rank"] = rank
    for k, v in division.items():
        if k in row:
            row[k] = v

    row_df = pd.DataFrame([row])

    st.write("### Feature Vector")
    st.dataframe(row_df)

    delta_best = best_model.predict(row_df)[0]
    new_rating_best = int(round(old_rating + delta_best))

    st.header(" Best Model Prediction")
    st.success(f"Predicted Rating Change: {round(delta_best)}")
    st.success(f"Predicted New Rating: {new_rating_best}")

    st.write("###  Predictions from All Models")
    table = []

    for name, model in all_models.items():
        try:
            delta = model.predict(row_df)[0]
            table.append([name, round(delta), round(old_rating + delta)])
        except:
            table.append([name, "ERR", "ERR"])

    df_results = pd.DataFrame(table, columns=["Model", "Predicted Δ", "New Rating"])
    st.dataframe(df_results)
