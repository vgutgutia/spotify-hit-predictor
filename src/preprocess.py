"""Combine decade CSVs, run EDA, split + scale for training."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "report"
MODELS_DIR = ROOT / "models"

RANDOM_STATE = 42

DECADES = ["60s", "70s", "80s", "90s", "00s", "10s"]
ID_COLS = ["track", "artist", "uri"]
TARGET = "target"

NUMERIC_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence",
    "tempo", "duration_ms", "chorus_hit", "sections",
    "mode", "time_signature",
]

CATEGORICAL_FEATURES = ["key", "decade"]


def load_and_combine():
    frames = []
    for d in DECADES:
        df = pd.read_csv(DATA_DIR / f"dataset-of-{d}.csv")
        df["decade"] = d
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def eda(df):
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df["target"].value_counts().sort_index().plot.bar(
        ax=axes[0], color=["#d9534f", "#5cb85c"]
    )
    axes[0].set_xticklabels(["flop (0)", "hit (1)"], rotation=0)
    axes[0].set_title("Overall class balance")
    axes[0].set_ylabel("count")

    by_decade = (
        df.groupby(["decade", "target"]).size().unstack(fill_value=0).reindex(DECADES)
    )
    by_decade.plot.bar(ax=axes[1], color=["#d9534f", "#5cb85c"], stacked=False)
    axes[1].set_title("Class balance by decade")
    axes[1].set_ylabel("count")
    axes[1].legend(["flop", "hit"])

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "class_balance.png", dpi=130)
    plt.close()

    numeric_for_plots = [
        "danceability", "energy", "loudness", "speechiness",
        "acousticness", "instrumentalness", "liveness", "valence",
        "tempo", "duration_ms", "chorus_hit", "sections",
    ]
    n = len(numeric_for_plots)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3))
    for ax, feat in zip(axes.flatten(), numeric_for_plots):
        sns.kdeplot(
            data=df, x=feat, hue="target", ax=ax,
            palette={0: "#d9534f", 1: "#5cb85c"},
            common_norm=False, fill=True, alpha=0.35,
        )
        ax.set_title(feat)
        ax.set_xlabel("")
    for ax in axes.flatten()[n:]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "feature_distributions.png", dpi=130)
    plt.close()


@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    feature_columns: list
    scaler: StandardScaler


def prepare_data(persist=True):
    df = pd.read_csv(DATA_DIR / "combined.csv")
    if df.isna().sum().sum():
        df = df.dropna()

    df = df.drop(columns=ID_COLS)
    df = pd.get_dummies(
        df, columns=CATEGORICAL_FEATURES, prefix=CATEGORICAL_FEATURES, dtype=int
    )

    y = df[TARGET].values.astype(int)
    X = df.drop(columns=[TARGET])
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    # fit scaler on train only to avoid leakage
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[NUMERIC_FEATURES] = scaler.fit_transform(X_train[NUMERIC_FEATURES])
    X_test[NUMERIC_FEATURES] = scaler.transform(X_test[NUMERIC_FEATURES])

    if persist:
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
        (MODELS_DIR / "feature_columns.json").write_text(
            json.dumps(feature_columns, indent=2)
        )

    return PreparedData(X_train, X_test, y_train, y_test, feature_columns, scaler)


def main():
    REPORT_DIR.mkdir(exist_ok=True)

    df = load_and_combine()
    print("shape:", df.shape)
    print("class balance:", df["target"].value_counts().to_dict())
    print("nulls:", int(df.isna().sum().sum()))

    eda(df)
    df.to_csv(DATA_DIR / "combined.csv", index=False)

    prep = prepare_data()
    print("X_train:", prep.X_train.shape, "X_test:", prep.X_test.shape)
    print("features:", len(prep.feature_columns))


if __name__ == "__main__":
    main()
