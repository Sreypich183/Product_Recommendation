import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import os
from pathlib import Path

st.set_page_config(page_title="NCF Dashboard", layout="wide")

# Use 127.0.0.1 to avoid some Windows localhost quirks
DEFAULT_URL = "http://127.0.0.1:8501/v1/models/ncf:predict"

# Resolve paths relative to the script location (not the terminal working dir)
APP_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = str(APP_DIR / "test_set_200k_new.csv")

st.title("NCF Recommender Dashboard (TF Serving)")

with st.sidebar:
    st.header("Config")
    tf_url = st.text_input("TF Serving URL", value=DEFAULT_URL)

    # Default to CSV next to app.py
    csv_path = st.text_input("Optional CSV (for charts)", value=DEFAULT_CSV)

    # Helpful debug (leave it for now, remove later if you want)
    st.caption("Debug")
    st.write("Working dir:", os.getcwd())
    st.write("App dir:", str(APP_DIR))
    st.write("CSV path:", csv_path)
    st.write("CSV exists?:", os.path.exists(csv_path))


@st.cache_data
def load_csv(path: str, mtime: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["user_id", "item_id", "category_id"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


# Load CSV if exists
if os.path.exists(csv_path):
    df = load_csv(csv_path, os.path.getmtime(csv_path))
else:
    df = None


# -----------------------
# Charts / Summary
# -----------------------
if df is not None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Users", f"{df['user_id'].nunique():,}")
    c3.metric("Items", f"{df['item_id'].nunique():,}")

    st.subheader("Top 10 Items by Interactions")
    top_items = df["item_id"].value_counts().head(10)
    fig = plt.figure()
    top_items.plot(kind="bar")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig, clear_figure=True)
else:
    st.info("CSV not found. Dashboard will still work for scoring, but without charts.")


# -----------------------
# TF Serving helpers
# -----------------------
def _post_or_raise(url: str, payload: dict, timeout: int = 20) -> dict:
    r = requests.post(url, json=payload, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"{r.status_code} {r.reason}: {r.text}")
    return r.json()


def predict_score(url: str, user_id: str, item_id: str):
    # IMPORTANT: send SCALARS, not ["1"] lists
    payload = {"instances": [{"user_id": str(user_id), "item_id": str(item_id)}]}
    out = _post_or_raise(url, payload, timeout=20)

    preds = out.get("predictions", [])
    if not preds:
        return None, out

    # Your TF Serving response looks like: {"predictions":[-0.00180074177]}
    p0 = preds[0]
    if isinstance(p0, dict) and "score" in p0:
        s = p0["score"]
        s = s[0] if isinstance(s, list) else s
        return float(s), out

    return float(p0), out


# -----------------------
# Single scoring
# -----------------------
st.divider()
st.subheader("Score a (user_id, item_id)")

colA, colB, colC = st.columns([1, 1, 1])
user_id = colA.text_input("User ID", value=(df["user_id"].iloc[0] if df is not None else "1"))
item_id = colB.text_input("Item ID", value=(df["item_id"].iloc[0] if df is not None else "2"))
do_score = colC.button("Score", type="primary")

if do_score:
    try:
        score, raw = predict_score(tf_url, user_id, item_id)
        st.success(f"Score = {score}")
        st.json(raw)
    except Exception as e:
        st.error(f"TF Serving call failed: {e}")
        st.caption("Check TF Serving is running and the URL is correct.")


# -----------------------
# Top-K scoring (sample items)
# -----------------------
st.divider()
st.subheader("Top-K (sampled items)")

if df is None:
    st.info("Provide a CSV with Item IDs (like test_set_200k_new.csv) to enable Top-K sampling.")
else:
    col1, col2, col3 = st.columns([1, 1, 1])
    user_k = col1.text_input("User ID for Top-K", value=df["user_id"].iloc[0], key="user_k")
    k = col2.number_input("K", 1, 50, 10)
    sample_n = col3.number_input("Sample items", 100, 20000, 2000, step=100)

    if st.button("Get Top-K", type="primary"):
        try:
            items = df["item_id"].unique()
            rng = np.random.default_rng(42)
            candidates = rng.choice(items, size=min(int(sample_n), len(items)), replace=False)

            batch = 512
            all_scores = []
            all_items = []

            for start in range(0, len(candidates), batch):
                batch_items = candidates[start:start + batch]

                payload = {
                    "instances": [
                        {"user_id": str(user_k), "item_id": str(it)}
                        for it in batch_items
                    ]
                }

                out = _post_or_raise(tf_url, payload, timeout=30)
                preds = out.get("predictions", [])

                for it, p in zip(batch_items, preds):
                    if isinstance(p, dict) and "score" in p:
                        s = p["score"]
                        s = s[0] if isinstance(s, list) else s
                    else:
                        s = p
                    all_items.append(it)
                    all_scores.append(float(s))

            all_scores = np.array(all_scores)
            idx = np.argsort(all_scores)[-int(k):][::-1]

            top_df = pd.DataFrame({
                "item_id": np.array(all_items)[idx],
                "Score": all_scores[idx]
            })

            st.dataframe(top_df, use_container_width=True)
        except Exception as e:
            st.error(f"Top-K failed: {e}")
