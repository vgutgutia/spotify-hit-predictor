"""Train LR, RF, XGBoost, and a 2-hidden-layer MLP. Pick best tree model for deployment."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import prepare_data

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORT_DIR = ROOT / "report"
RANDOM_STATE = 42


class MLP(nn.Module):
    """input(32) -> hidden(64) -> hidden(32) -> output(1)."""
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def score_sklearn(name, model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return _metrics_row(name, y_test, pred, proba)


def _metrics_row(name, y_test, pred, proba):
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def train_mlp(X_train, y_train, X_test, y_test, n_epochs=40, batch_size=256):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test.values, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size, shuffle=True,
    )

    model = MLP(n_features=X_train.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:>2}/{n_epochs}  loss={total_loss/len(X_train_t):.4f}")

    model.train(False)
    with torch.no_grad():
        proba = torch.sigmoid(model(X_test_t)).numpy().flatten()
    pred = (proba >= 0.5).astype(int)
    return model, _metrics_row("MLP", y_test, pred, proba)


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

    tree_models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.1, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }

    results = []
    fitted = {}
    for name, model in tree_models.items():
        print("fitting", name)
        model.fit(X_train, y_train)
        results.append(score_sklearn(name, model, X_test, y_test))
        fitted[name] = model

    print("fitting MLP")
    mlp_model, mlp_row = train_mlp(X_train, y_train.astype(np.float32), X_test, y_test)
    results.append(mlp_row)

    df = pd.DataFrame(results).set_index("model")
    print(df.round(4).to_string())

    # deployed model = best tree model (SHAP TreeExplainer needs a tree)
    tree_df = df.drop("MLP", errors="ignore")
    winner_name = tree_df["roc_auc"].idxmax()
    winner_model = fitted[winner_name]
    print("deployed winner (best tree by roc-auc):", winner_name)
    print("overall best by roc-auc:", df["roc_auc"].idxmax())

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
