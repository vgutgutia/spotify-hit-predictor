"""Train LR, RF, XGBoost. Pick best by ROC-AUC. Save model + metrics + feature importance plot."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import prepare_data

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORT_DIR = ROOT / "report"
RANDOM_STATE = 42


def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def feature_importance_plot(name, model, feature_columns, out_path):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.abs(model.coef_[0])

    series = (
        pd.Series(importances, index=feature_columns)
        .sort_values(ascending=True)
        .tail(15)
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    series.plot.barh(ax=ax, color="#5cb85c")
    ax.set_title(f"Top-15 feature importance ({name})")
    ax.set_xlabel("importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def main():
    prep = prepare_data()
    X_train, X_test = prep.X_train, prep.X_test
    y_train, y_test = prep.y_train, prep.y_test

    print("X_train:", X_train.shape, "X_test:", X_test.shape)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.1, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }

    results = []
    fitted = {}
    for name, model in models.items():
        print("fitting", name)
        model.fit(X_train, y_train)
        results.append(evaluate(name, model, X_test, y_test))
        fitted[name] = model

    df = pd.DataFrame(results).set_index("model")
    print(df.round(4).to_string())

    winner_name = df["roc_auc"].idxmax()
    winner_model = fitted[winner_name]
    print("winner:", winner_name)

    MODELS_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    joblib.dump(winner_model, MODELS_DIR / "best_model.pkl")
    (MODELS_DIR / "metrics.json").write_text(json.dumps({
        "winner": winner_name,
        "selection_metric": "roc_auc",
        "models": results,
    }, indent=2))
    feature_importance_plot(
        winner_name, winner_model, prep.feature_columns,
        REPORT_DIR / "feature_importance.png",
    )


if __name__ == "__main__":
    main()
