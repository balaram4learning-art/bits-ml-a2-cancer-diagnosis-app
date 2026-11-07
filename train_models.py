"""Train the five classifiers, save them to model/, and write test_data.csv.

Dataset: UCI Breast Cancer Wisconsin (Diagnostic) - 569 rows, 30 numeric features.
Run:  python train_models.py
"""
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

from metrics_utils import TARGET_COL, MODEL_FILES, score_model

SEED = 42

# ---- data -----------------------------------------------------------------
raw = load_breast_cancer(as_frame=True)
X = raw.data
y = 1 - raw.target            # sklearn: 0 = malignant; flip so malignant = 1 (positive class)
y.name = TARGET_COL

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)

# Full-feature test set the Streamlit app will upload / evaluate
pd.concat([X_test, y_test], axis=1).to_csv("test_data.csv", index=False)

# ---- models ---------------------------------------------------------------
# Scale-sensitive models get the scaler inside a Pipeline so the fitted scaler
# is saved together with the classifier (no separate preprocessing artifact).
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=SEED),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": make_pipeline(StandardScaler(), CalibratedClassifierCV(SVC(kernel="rbf", C=1.0, random_state=SEED), ensemble=False)),
}

rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"model/{MODEL_FILES[name]}")
    rows.append({"ML Model Name": name, **score_model(model, X_test, y_test)})

results = pd.DataFrame(rows).round(4)
results.to_csv("model/metrics.csv", index=False)
Path("model/feature_names.json").write_text(json.dumps(list(X.columns)))
print(f"train={len(X_train)} test={len(X_test)} features={X.shape[1]}")
print(results.to_string(index=False))
