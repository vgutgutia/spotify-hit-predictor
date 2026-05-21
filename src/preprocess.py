"""EDA + preprocessing for the Spotify Hit Predictor.

Phase 1 (this file, step 2): combine decade CSVs and run EDA.
Phase 2 (step 3): scaling / encoding / split — added after EDA review.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "report"
REPORT_DIR.mkdir(exist_ok=True)

DECADES = ["60s", "70s", "80s", "90s", "00s", "10s"]


def load_and_combine() -> pd.DataFrame:
    frames = []
    for d in DECADES:
        df = pd.read_csv(DATA_DIR / f"dataset-of-{d}.csv")
        df["decade"] = d
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return combined


def eda(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    # 1. Class balance overall + by decade
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

    # 2. Feature distributions: hits vs flops
    numeric_features = [
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms",
        "chorus_hit",
        "sections",
    ]
    n_features = len(numeric_features)
    cols = 3
    rows = (n_features + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3))
    for ax, feat in zip(axes.flatten(), numeric_features):
        sns.kdeplot(
            data=df, x=feat, hue="target", ax=ax,
            palette={0: "#d9534f", 1: "#5cb85c"}, common_norm=False, fill=True, alpha=0.35,
        )
        ax.set_title(feat)
        ax.set_xlabel("")
    for ax in axes.flatten()[n_features:]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "feature_distributions.png", dpi=130)
    plt.close()


def main() -> None:
    df = load_and_combine()
    print(f"Combined shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Nulls per column: {df.isna().sum().to_dict()}")
    print(f"Class balance: {df['target'].value_counts().to_dict()}")
    print(f"Class balance %: {(df['target'].value_counts(normalize=True) * 100).round(2).to_dict()}")
    print(f"Decade counts: {df['decade'].value_counts().reindex(DECADES).to_dict()}")
    print(f"Numeric describe:\n{df.describe().T[['mean', 'std', 'min', 'max']].round(3)}")

    eda(df)
    out_path = DATA_DIR / "combined.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} and EDA plots to {REPORT_DIR}/")


if __name__ == "__main__":
    main()
