import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.title("Codeforces Problem Rating Predictor")

feature_names = [
    "contestId", "points", "solvedCount", "special", "2-sat", "binary search", "bitmasks",
    "brute force", "chinese remainder theorem", "combinatorics", "constructive algorithms",
    "data structures", "dfs and similar", "divide and conquer", "dp", "dsu",
    "expression parsing", "fft", "flows", "games", "geometry", "graph matchings",
    "graphs", "greedy", "hashing", "implementation", "interactive", "math",
    "matrices", "meet-in-the-middle", "number theory", "probabilities", "schedules",
    "shortest paths", "sortings", "string suffix structures", "strings",
    "ternary search", "trees", "two pointers", "index_num"
]

inputs = []
inputs.append(st.number_input("contestId", min_value=1, value=2156))
inputs.append(st.number_input("points", value=500.0))
inputs.append(st.number_input("solvedCount", value=0))
for feat in feature_names[3:]:
    inputs.append(st.number_input(feat, min_value=0, max_value=1, value=0)) 
    
model = joblib.load("models/best_model_RandomForestRegressor.joblib")

if st.button("Predict Problem Rating"):
    X = np.array(inputs).reshape(1, -1)
    pred = model.predict(X)[0]
    pred_rounded = int(np.round(pred / 100) * 100)
    st.success(f"Predicted Rating: {pred:.2f} (rounded = {pred_rounded})")
