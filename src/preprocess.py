"""EDA + preprocessing for the Spotify Hit Predictor.

Phase 1 (step 2): load decade CSVs, combine, run EDA, write combined.csv.
Phase 2 (step 3): drop IDs, one-hot encode key + decade, stratified 80/20
                  split, then StandardScale numerics fit on the train fold
                  only (avoids train/test leakage). Persists the scaler and
                  the feature column order to models/ so train.py and the
                  Streamlit app can reuse them.
"""
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


def load_and_combine() -> pd.DataFrame:
    frames = []
    for d in DECADES:
        df = pd.read_csv(DATA_DIR / f"dataset-of-{d}.csv")
        df["decade"] = d
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def eda(df: pd.DataFrame) -> None:
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
    n_features = len(numeric_for_plots)
    cols = 3
    rows = (n_features + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3))
    for ax, feat in zip(axes.flatten(), numeric_for_plots):
        sns.kdeplot(
            data=df, x=feat, hue="target", ax=ax,
            palette={0: "#d9534f", 1: "#5cb85c"},
            common_norm=False, fill=True, alpha=0.35,
        )
        ax.set_title(feat)
        ax.set_xlabel("")
    for ax in axes.flatten()[n_features:]:
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
    feature_columns: list[str]
    scaler: StandardScaler


def _build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=ID_COLS)
    df = pd.get_dummies(
        df, columns=CATEGORICAL_FEATURES, prefix=CATEGORICAL_FEATURES, dtype=int
    )
    return df


def prepare_data(persist: bool = True) -> PreparedData:
    df = pd.read_csv(DATA_DIR / "combined.csv")

    null_count = int(df.isna().sum().sum())
    if null_count:
        print(f"[warn] {null_count} nulls in combined.csv — dropping rows")
        df = df.dropna()

    df = _build_feature_frame(df)

    y = df[TARGET].values.astype(int)
    X = df.drop(columns=[TARGET])
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

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

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_columns=feature_columns,
        scaler=scaler,
    )


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)

    df = load_and_combine()
    print(f"Combined shape: {df.shape}")
    print(f"Class balance: {df['target'].value_counts().to_dict()}")
    print(f"Nulls total: {int(df.isna().sum().sum())}")
    print(f"Decade counts: {df['decade'].value_counts().reindex(DECADES).to_dict()}")

    eda(df)
    combined_path = DATA_DIR / "combined.csv"
    df.to_csv(combined_path, index=False)
    print(f"\nSaved {combined_path} and EDA plots to {REPORT_DIR}/")

    prep = prepare_data()
    print(f"\n[prep] X_train: {prep.X_train.shape}  X_test: {prep.X_test.shape}")
    print(f"[prep] features ({len(prep.feature_columns)}): {prep.feature_columns}")
    print(f"[prep] saved scaler.pkl + feature_columns.json to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
