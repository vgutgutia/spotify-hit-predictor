"""Train Logistic Regression, Random Forest, and XGBoost; pick the winner by
ROC-AUC; save the winning model, all-model metrics, and a feature-importance
plot of the winner.
"""
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
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import prepare_data  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORT_DIR = ROOT / "report"
RANDOM_STATE = 42


def evaluate(name: str, model, X_test, y_test) -> dict:
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


def feature_importance_plot(
    name: str, model, feature_columns: list[str], out_path: Path
) -> None:
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        kind = "feature_importances_"
    else:
        # LogisticRegression: shape (1, n_features) for binary
        importances = np.abs(model.coef_[0])
        kind = "|coef_|"

    series = (
        pd.Series(importances, index=feature_columns)
        .sort_values(ascending=True)
        .tail(15)
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    series.plot.barh(ax=ax, color="#5cb85c")
    ax.set_title(f"Top-15 feature importance — {name} ({kind})")
    ax.set_xlabel("importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def main() -> None:
    prep = prepare_data()
    X_train, X_test = prep.X_train, prep.X_test
    y_train, y_test = prep.y_train, prep.y_test

    print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")
    print(f"Training 3 models (random_state={RANDOM_STATE})...\n")

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = []
    fitted = {}
    for name, model in models.items():
        print(f"  fitting {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate(name, model, X_test, y_test)
        results.append(metrics)
        fitted[name] = model

    df = pd.DataFrame(results).set_index("model")
    print("\nResults:")
    print(df.round(4).to_string())

    winner_name = df["roc_auc"].idxmax()
    winner_model = fitted[winner_name]
    print(
        f"\nWinner by ROC-AUC: {winner_name} "
        f"({df.loc[winner_name, 'roc_auc']:.4f})"
    )

    MODELS_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    joblib.dump(winner_model, MODELS_DIR / "best_model.pkl")
    (MODELS_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "winner": winner_name,
                "selection_metric": "roc_auc",
                "models": results,
            },
            indent=2,
        )
    )
    feature_importance_plot(
        winner_name,
        winner_model,
        prep.feature_columns,
        REPORT_DIR / "feature_importance.png",
    )
    print(
        "\nSaved: models/best_model.pkl, models/metrics.json, "
        "report/feature_importance.png"
    )


if __name__ == "__main__":
    main()
