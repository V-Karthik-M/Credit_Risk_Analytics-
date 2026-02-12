import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Credit Risk Executive Dashboard", layout="wide")

st.title("Credit Risk Executive Dashboard")

# --- Load data ---
pred_path = Path("deploy_outputs")
files = sorted(pred_path.glob("predictions_*.csv"))

if not files:
    st.error("No prediction files found in deploy_outputs/. Make sure Step 10 created predictions_*.csv")
    st.stop()

selected_file = st.sidebar.selectbox("Select prediction output file", files, index=len(files)-1)
pred_df = pd.read_csv(selected_file)

# Basic type handling
if "scoring_date" in pred_df.columns:
    pred_df["scoring_date"] = pd.to_datetime(pred_df["scoring_date"], errors="coerce")

# --- Sidebar filters ---
st.sidebar.header("Filters")
bucket_filter = st.sidebar.multiselect(
    "Risk Bucket",
    options=sorted(pred_df["risk_bucket"].dropna().unique()),
    default=sorted(pred_df["risk_bucket"].dropna().unique())
)

df = pred_df[pred_df["risk_bucket"].isin(bucket_filter)].copy()

# --- KPI Row ---
c1, c2, c3, c4 = st.columns(4)

total = len(df)
high = (df["risk_bucket"] == "High").sum()
med = (df["risk_bucket"] == "Medium").sum()
low = (df["risk_bucket"] == "Low").sum()

c1.metric("Total Customers", f"{total:,}")
c2.metric("High Risk", f"{high:,}")
c3.metric("Medium Risk", f"{med:,}")
c4.metric("Low Risk", f"{low:,}")

st.divider()

# --- Charts ---
left, right = st.columns(2)

with left:
    st.subheader("Customer Risk Segmentation")
    bucket_counts = df["risk_bucket"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)

    fig = plt.figure()
    plt.bar(bucket_counts.index, bucket_counts.values)
    plt.xlabel("Risk Bucket")
    plt.ylabel("Number of Customers")
    st.pyplot(fig)

with right:
    st.subheader("Risk Score Distribution")
    fig2 = plt.figure()
    plt.hist(df["risk_score"], bins=25)
    plt.xlabel("Risk Score")
    plt.ylabel("Count")
    st.pyplot(fig2)

st.divider()

# --- Top risky customers table ---
st.subheader("Top High-Risk Customers (by risk_score)")
top_n = st.slider("How many to show?", min_value=10, max_value=200, value=50, step=10)

cols_to_show = [c for c in ["customer_id", "risk_score", "risk_bucket", "model_version", "scoring_date"] if c in df.columns]
st.dataframe(df.sort_values("risk_score", ascending=False)[cols_to_show].head(top_n), use_container_width=True)

st.divider()

# --- Model / Monitoring section (optional if files exist) ---
st.header("Model Health & Monitoring")

monitor_path = Path("monitoring")
pred_drift_files = sorted(monitor_path.glob("prediction_drift_*.csv"))
feat_drift_files = sorted(monitor_path.glob("feature_drift_*.csv"))

m1, m2 = st.columns(2)

with m1:
    st.subheader("Prediction Drift Summary (if available)")
    if pred_drift_files:
        drift = pd.read_csv(pred_drift_files[-1], index_col=0)
        st.dataframe(drift, use_container_width=True)
    else:
        st.info("No prediction_drift_*.csv found in monitoring/. (Step 11)")

with m2:
    st.subheader("Feature Drift Summary (if available)")
    if feat_drift_files:
        fdrift = pd.read_csv(feat_drift_files[-1], index_col=0)
        st.dataframe(fdrift.sort_values("mean_delta", ascending=False), use_container_width=True)
    else:
        st.info("No feature_drift_*.csv found in monitoring/. (Step 11)")

st.caption(f"Loaded: {selected_file}")
