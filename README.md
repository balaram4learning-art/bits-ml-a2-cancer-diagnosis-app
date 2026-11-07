# Breast Cancer Diagnosis - Classification Model Comparison

## a. Problem statement
Predict whether a breast mass is **malignant (1)** or **benign (0)** from 30 numeric measurements of cell-nucleus images (digitised fine-needle-aspirate). Five classifiers - Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes and SVM - are trained on the same data, compared on Accuracy, AUC, Precision, Recall, F1 and MCC, and exposed through a Streamlit app that evaluates the **saved** models on an uploaded test CSV (no retraining).
In this domain a missed malignancy (false negative) is the costly error, so Recall and MCC are read alongside accuracy.

## b. Dataset description
| Item | Value |
|---|---|
| Source | UCI ML Repository - Breast Cancer Wisconsin (Diagnostic), loaded via `sklearn.datasets.load_breast_cancer` |
| Instances | 569 (212 malignant, 357 benign) |
| Features | 30 numeric (mean / standard error / worst of 10 nucleus properties: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension) |
| Target | `diagnosis` - 1 = malignant (positive class), 0 = benign (note: sklearn's native coding is reversed; flipped here) |
| Missing values | None |
| Split | 80/20 stratified, `random_state=42` -> 455 train / 114 test (`test_data.csv`) |
| Preprocessing | `StandardScaler` inside a Pipeline for Logistic Regression, KNN and SVM (saved with the model); none for Decision Tree / Naive Bayes |

## c. GitHub repository link
`<ADD YOUR GITHUB REPO URL HERE>`  ·  Live app: `<ADD YOUR STREAMLIT COMMUNITY CLOUD URL HERE>`

```
app.py                 Streamlit app
train_models.py        trains + saves the 5 models, writes test_data.csv
ML_Assignment2.ipynb   notebook version (EDA, training, evaluation)
metrics_utils.py       shared metric helper
requirements.txt
test_data.csv          114-row hold-out set (30 features + diagnosis)
model/                 5 x .joblib, feature_names.json, metrics.csv
```
Run locally: `pip install -r requirements.txt && python train_models.py && streamlit run app.py`

## d. Models used
Metrics on the 114-row hold-out test set (`test_data.csv`):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9035 | 0.9471 | 0.9429 | 0.7857 | 0.8571 | 0.7926 |
| KNN | 0.9561 | 0.9825 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | 0.9737 | 0.9947 | 0.9756 | 0.9524 | 0.9639 | 0.9433 |

### Observations
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very strong for a linear model (AUC 0.996, the highest of all five): the classes are nearly linearly separable once features are standardised. Misses 3 of 42 malignant cases (Recall 0.929). Fast, interpretable and a hard baseline to beat. |
| Decision Tree | Weakest model on every metric (Accuracy 0.904, MCC 0.793). Recall of 0.786 means 9 of 42 malignancies are missed - unacceptable for screening. A depth-limited single tree has high variance and cannot match smooth boundaries across 30 correlated features; an ensemble would likely help. |
| KNN | Good (Accuracy 0.956, MCC 0.906) once features are scaled; without scaling, large-valued features such as area would dominate the distance. Slightly behind LR/SVM on recall (0.905) and slower at prediction time as it stores all training data. |
| Naive Bayes | Perfect precision (1.00, no false alarms) but low recall (0.833; 7 malignancies missed), and its AUC (0.993) is high, so ranking is good but the 0.5 threshold is poorly calibrated. The independence assumption is violated by the many highly correlated size features (radius/perimeter/area). |
| Support Vector Machine | Best on Accuracy (0.974), Recall (0.952), F1 (0.964) and MCC (0.943); only 2 malignant cases missed and 1 false alarm. The RBF kernel with scaled inputs captures the non-linear boundary while staying regularised. AUC (0.995) is essentially tied with Logistic Regression. |
| **Overall winner** | **Support Vector Machine** - highest Accuracy, F1, Recall and MCC, with Logistic Regression a very close second (best AUC). Caveat: with only 114 test rows, one case changes accuracy by ~0.9 points, so the SVM/LR gap is within noise; LR would be a defensible choice for its simplicity. |

## Streamlit app features
- CSV upload (test data only; the bundled `test_data.csv` can be used instead)
- Model selection dropdown (5 models)
- Accuracy / AUC / Precision / Recall / F1 / MCC for the selected model
- Confusion matrix, classification report, ROC curves for all models
- Side-by-side metric table and bar chart for all five models on the uploaded data

The uploaded CSV must contain the 30 feature columns (same names as `test_data.csv`) and the `diagnosis` target column with both classes present.
