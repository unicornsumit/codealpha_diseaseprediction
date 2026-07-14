"""
TASK 4: Disease Prediction from Medical Data
=============================================
Objective : Predict the possibility of disease based on patient data.
Approach  : Classification techniques applied to structured medical datasets.

Datasets used:
    1. Breast Cancer (Wisconsin) - loaded via sklearn (mirrors UCI ML Repository version)
    2. Heart Disease (UCI Cleveland)
    3. Diabetes (Pima Indians)

Algorithms compared:
    - Logistic Regression
    - Support Vector Machine (SVM)
    - Random Forest
    - XGBoost

For each dataset the script:
    1. Loads and cleans the data
    2. Splits into train/test sets (stratified)
    3. Scales features
    4. Trains all four models
    5. Evaluates with Accuracy, Precision, Recall, F1-score, ROC-AUC
    6. Plots confusion matrices + a model-comparison bar chart
    7. Saves the best model for each dataset to disk (.pkl)

Run:
    python disease_prediction.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

sns.set_style("whitegrid")
RANDOM_STATE = 42


# ----------------------------------------------------------------------
# 1. Data loading functions (one per dataset)
# ----------------------------------------------------------------------
def load_breast_cancer_data():
    """Breast Cancer Wisconsin dataset (569 samples, 30 features, binary target)."""
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    X = df.drop(columns=["target"])
    y = df["target"]  # 0 = malignant, 1 = benign
    return X, y, "Breast Cancer"


def load_heart_data():
    """UCI Heart Disease (Cleveland) dataset (303 samples, 13 features, binary target)."""
    path = os.path.join(DATA_DIR, "heart.csv")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    X = df.drop(columns=["target"])
    y = df["target"]  # 0 = no disease, 1 = disease
    return X, y, "Heart Disease"


def load_diabetes_data():
    """Pima Indians Diabetes dataset (768 samples, 8 features, binary target)."""
    path = os.path.join(DATA_DIR, "diabetes.csv")
    cols = ["pregnancies", "glucose", "blood_pressure", "skin_thickness",
             "insulin", "bmi", "diabetes_pedigree", "age", "outcome"]
    df = pd.read_csv(path, names=cols)

    # Several columns use 0 as a placeholder for "missing" -- replace with median
    zero_as_missing = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]
    for col in zero_as_missing:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())

    X = df.drop(columns=["outcome"])
    y = df["outcome"]  # 0 = no diabetes, 1 = diabetes
    return X, y, "Diabetes"


# ----------------------------------------------------------------------
# 2. Model zoo
# ----------------------------------------------------------------------
def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=300, use_label_encoder=False,
            eval_metric="logloss", random_state=RANDOM_STATE
        ),
    }


# ----------------------------------------------------------------------
# 3. Train + evaluate one dataset across all models
# ----------------------------------------------------------------------
def run_pipeline(loader_fn):
    X, y, name = loader_fn()
    print(f"\n{'='*60}\nDATASET: {name}   (samples={X.shape[0]}, features={X.shape[1]})\n{'='*60}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = get_models()
    results = []
    fitted_models = {}

    for model_name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        results.append({
            "Model": model_name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1-Score": f1, "ROC-AUC": auc
        })
        fitted_models[model_name] = model

        print(f"\n--- {model_name} ---")
        print(classification_report(y_test, y_pred, digits=3))

    results_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False).reset_index(drop=True)
    print(f"\nSummary for {name}:\n", results_df.round(3).to_string(index=False))

    # Save results table
    results_df.to_csv(os.path.join(OUT_DIR, f"{name.replace(' ', '_').lower()}_results.csv"), index=False)

    # Confusion matrices (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    fig.suptitle(f"Confusion Matrices — {name}", fontsize=14, fontweight="bold")
    for ax, (model_name, model) in zip(axes.ravel(), fitted_models.items()):
        y_pred = model.predict(X_test_s)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
        ax.set_title(model_name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{name.replace(' ', '_').lower()}_confusion_matrices.png"), dpi=150)
    plt.close()

    # Model comparison bar chart
    plt.figure(figsize=(9, 5))
    plot_df = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric")
    plt.title(f"Model Comparison — {name}")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{name.replace(' ', '_').lower()}_comparison.png"), dpi=150)
    plt.close()

    # Save best model + scaler
    best_model_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_model_name]
    joblib.dump(best_model, os.path.join(MODEL_DIR, f"{name.replace(' ', '_').lower()}_best_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"{name.replace(' ', '_').lower()}_scaler.pkl"))
    print(f"Best model for {name}: {best_model_name}  (saved to models/)")

    return results_df.assign(Dataset=name)


# ----------------------------------------------------------------------
# 4. Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_results = []
    for loader in [load_breast_cancer_data, load_heart_data, load_diabetes_data]:
        all_results.append(run_pipeline(loader))

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(os.path.join(OUT_DIR, "all_results_summary.csv"), index=False)

    print(f"\n{'='*60}\nALL DATASETS — FULL SUMMARY\n{'='*60}")
    print(final_df.round(3).to_string(index=False))
    print(f"\nAll outputs (plots + CSVs) saved in: {OUT_DIR}")
    print(f"All trained models saved in: {MODEL_DIR}")
