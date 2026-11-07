"""Streamlit app: evaluate pre-trained breast-cancer classifiers on an uploaded test CSV.

Models are loaded from model/*.joblib and are NEVER retrained on the uploaded data.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, roc_curve)

from metrics_utils import CLASS_NAMES, MODEL_FILES, TARGET_COL, score_model

BASE = Path(__file__).parent
st.set_page_config(page_title="Breast Cancer Model Evaluator", page_icon="🩺", layout="wide")


@st.cache_resource
def load_models():
    return {n: joblib.load(BASE / "model" / f) for n, f in MODEL_FILES.items()}


@st.cache_data
def load_features():
    return json.loads((BASE / "model" / "feature_names.json").read_text())


models, features = load_models(), load_features()

# ---- header ---------------------------------------------------------------
st.title("🩺 Breast Cancer Diagnosis - Classifier Comparison")
st.caption("UCI Breast Cancer Wisconsin (Diagnostic) · 30 numeric features · "
           "target `diagnosis` (1 = malignant, 0 = benign). Pre-trained models are evaluated, not retrained.")

# ---- sidebar --------------------------------------------------------------
with st.sidebar:
    st.header("1 · Upload test data")
    upload = st.file_uploader("CSV with all 30 features + `diagnosis`", type="csv")
    use_sample = st.checkbox("Use bundled test_data.csv", value=upload is None, disabled=upload is not None)
    st.header("2 · Choose model")
    choice = st.selectbox("Model", list(models))

if upload is not None:
    df = pd.read_csv(upload)
elif use_sample:
    df = pd.read_csv(BASE / "test_data.csv")
else:
    st.info("Upload a test CSV (or tick the bundled sample) to begin.")
    st.stop()

missing = [c for c in features + [TARGET_COL] if c not in df.columns]
if missing:
    st.error(f"Uploaded file is missing required columns: {missing}")
    st.stop()
if not set(df[TARGET_COL].unique()) <= {0, 1} or df[TARGET_COL].nunique() < 2:
    st.error(f"`{TARGET_COL}` must contain both classes 0 and 1 (needed for AUC).")
    st.stop()

df = df.dropna(subset=features + [TARGET_COL])
X, y = df[features], df[TARGET_COL].astype(int)
st.write(f"**{len(df)} rows** loaded · {y.sum()} malignant / {(y == 0).sum()} benign")
with st.expander("Preview data"):
    st.dataframe(df.head(20), width="stretch")

# ---- selected model -------------------------------------------------------
model = models[choice]
m = score_model(model, X, y)
pred = model.predict(X)

st.subheader(f"Evaluation metrics - {choice}")
cols = st.columns(6)
for col, (k, v) in zip(cols, m.items()):
    col.metric(k, f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y, pred), display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Blues", colorbar=False)
    st.pyplot(fig)
with right:
    st.markdown("**ROC curve (all models)**")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    for n, mdl in models.items():
        fpr, tpr, _ = roc_curve(y, mdl.predict_proba(X)[:, 1])
        ax.plot(fpr, tpr, lw=3 if n == choice else 1, label=n)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(fontsize=7)
    st.pyplot(fig)

st.markdown("**Classification report**")
report = classification_report(y, pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")

# ---- all-model comparison --------------------------------------------------
st.subheader("All models on this data")
table = pd.DataFrame([{"ML Model Name": n, **score_model(mdl, X, y)} for n, mdl in models.items()]).set_index("ML Model Name")
st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#cfe8cf"), width="stretch")
fig, ax = plt.subplots(figsize=(9, 3.5))
table.plot.bar(ax=ax, rot=15); ax.set_ylim(0.8, 1.01); ax.legend(ncol=6, fontsize=7, loc="lower center")
st.pyplot(fig)
