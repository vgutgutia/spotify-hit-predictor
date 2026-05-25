"""Streamlit app for the Spotify Hit Predictor.

Two tabs:
  1. Sliders  — pick an example song from the dataset (or design one from
                scratch), tweak the audio-feature sliders, and predict.
  2. About    — model summary + global feature-importance chart.

Each prediction shows a verdict (hit/flop), the hit probability, and a SHAP
breakdown of which features pushed THIS song toward / away from being a hit.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORT_DIR = ROOT / "report"

NUMERIC_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence",
    "tempo", "duration_ms", "chorus_hit", "sections",
    "mode", "time_signature",
]

# (min, max, default) — bounds inferred from training data with a little headroom
SLIDER_RANGES: dict[str, tuple[float, float, float]] = {
    "danceability":      (0.0, 1.0, 0.55),
    "energy":            (0.0, 1.0, 0.62),
    "loudness":          (-60.0, 5.0, -8.0),
    "speechiness":       (0.0, 1.0, 0.06),
    "acousticness":      (0.0, 1.0, 0.25),
    "instrumentalness":  (0.0, 1.0, 0.02),
    "liveness":          (0.0, 1.0, 0.18),
    "valence":           (0.0, 1.0, 0.55),
    "tempo":             (40.0, 220.0, 120.0),
    "duration_ms":       (30000, 900000, 230000),
    "chorus_hit":        (0.0, 300.0, 40.0),
    "sections":          (1, 50, 10),
    "mode":              (0, 1, 1),
    "time_signature":    (0, 7, 4),
}

DECADES = ["60s", "70s", "80s", "90s", "00s", "10s"]
KEY_NAMES = ["C", "C♯/D♭", "D", "D♯/E♭", "E", "F", "F♯/G♭",
             "G", "G♯/A♭", "A", "A♯/B♭", "B"]

# Real songs from the dataset (preserves actual feature values). Picking
# mostly famous artists so the demo lands — the actual hit/flop label is
# shown in parentheses, and the model's prediction should match.
EXAMPLE_SONGS: dict[str, dict] = {
    "🎵 Can't Buy Me Love — The Beatles (60s, hit)": {
        "decade": "60s", "key": 0,
        "numeric": {"danceability": 0.844, "energy": 0.299, "loudness": -12.645,
                    "speechiness": 0.0621, "acousticness": 0.848,
                    "instrumentalness": 0.844, "liveness": 0.102, "valence": 0.566,
                    "tempo": 102.088, "duration_ms": 131176, "chorus_hit": 20.009,
                    "sections": 6, "mode": 1, "time_signature": 4},
    },
    "🎵 Miss You — The Rolling Stones (70s, hit)": {
        "decade": "70s", "key": 9,
        "numeric": {"danceability": 0.795, "energy": 0.710, "loudness": -4.746,
                    "speechiness": 0.0392, "acousticness": 0.443,
                    "instrumentalness": 0.0215, "liveness": 0.344, "valence": 0.845,
                    "tempo": 109.689, "duration_ms": 288667, "chorus_hit": 45.452,
                    "sections": 15, "mode": 0, "time_signature": 4},
    },
    "🎵 Billie Jean — Michael Jackson (80s, hit)": {
        "decade": "80s", "key": 11,
        "numeric": {"danceability": 0.920, "energy": 0.654, "loudness": -3.051,
                    "speechiness": 0.0401, "acousticness": 0.0236,
                    "instrumentalness": 0.0158, "liveness": 0.0359, "valence": 0.847,
                    "tempo": 117.046, "duration_ms": 293827, "chorus_hit": 29.591,
                    "sections": 14, "mode": 0, "time_signature": 4},
    },
    "🎵 My Name Is — Eminem (90s, hit)": {
        "decade": "90s", "key": 1,
        "numeric": {"danceability": 0.869, "energy": 0.680, "loudness": -6.233,
                    "speechiness": 0.318, "acousticness": 0.0416,
                    "instrumentalness": 1.12e-06, "liveness": 0.0914, "valence": 0.815,
                    "tempo": 85.519, "duration_ms": 268400, "chorus_hit": 101.419,
                    "sections": 8, "mode": 1, "time_signature": 4},
    },
    "🎵 Hotline Bling — Drake (10s, hit)": {
        "decade": "10s", "key": 2,
        "numeric": {"danceability": 0.891, "energy": 0.625, "loudness": -7.861,
                    "speechiness": 0.0558, "acousticness": 0.00261,
                    "instrumentalness": 0.000176, "liveness": 0.0504,
                    "valence": 0.548, "tempo": 134.967, "duration_ms": 267067,
                    "chorus_hit": 69.390, "sections": 8, "mode": 1, "time_signature": 4},
    },
    "💀 River Man — Nick Drake (60s, flop)": {
        "decade": "60s", "key": 8,
        "numeric": {"danceability": 0.474, "energy": 0.118, "loudness": -20.098,
                    "speechiness": 0.0339, "acousticness": 0.852,
                    "instrumentalness": 0.735, "liveness": 0.112, "valence": 0.0921,
                    "tempo": 114.520, "duration_ms": 258680, "chorus_hit": 39.473,
                    "sections": 18, "mode": 1, "time_signature": 5},
    },
    "💀 3's & 7's — Queens of the Stone Age (00s, flop)": {
        "decade": "00s", "key": 10,
        "numeric": {"danceability": 0.438, "energy": 0.990, "loudness": -3.890,
                    "speechiness": 0.108, "acousticness": 0.0651,
                    "instrumentalness": 0.0222, "liveness": 0.359, "valence": 0.452,
                    "tempo": 131.306, "duration_ms": 214067, "chorus_hit": 40.384,
                    "sections": 9, "mode": 0, "time_signature": 4},
    },
}

START_FRESH = "— start fresh —"


# ---------------------------------------------------------------- artifacts --

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / "best_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    feature_columns = json.loads((MODELS_DIR / "feature_columns.json").read_text())
    metrics = json.loads((MODELS_DIR / "metrics.json").read_text())
    explainer = shap.TreeExplainer(model)
    return model, scaler, feature_columns, metrics, explainer


# -------------------------------------------------------------- prediction --

def build_feature_row(
    numeric: dict, key: int, decade: str, feature_columns: list[str]
) -> pd.DataFrame:
    """Assemble a one-row DataFrame matching the trained model's column order."""
    row = {col: 0 for col in feature_columns}
    for k, v in numeric.items():
        if k in row:
            row[k] = v
    row[f"key_{key}"] = 1
    row[f"decade_{decade}"] = 1
    return pd.DataFrame([row], columns=feature_columns)


def predict_and_explain(
    raw_df: pd.DataFrame, model, scaler, feature_columns, explainer
):
    """Scale numerics, predict probability, compute SHAP values."""
    scaled = raw_df.copy()
    scaled[NUMERIC_FEATURES] = scaler.transform(scaled[NUMERIC_FEATURES])
    proba = float(model.predict_proba(scaled)[0, 1])
    shap_values = explainer.shap_values(scaled)
    contribs = pd.Series(np.asarray(shap_values)[0], index=feature_columns)
    return proba, contribs


def render_prediction(proba: float, contribs: pd.Series) -> None:
    is_hit = proba >= 0.5
    label = "HIT" if is_hit else "FLOP"
    color = "#1DB954" if is_hit else "#E22134"

    st.markdown(
        f"<h2 style='color:{color}; margin-bottom:0.2em'>"
        f"{label} &mdash; {proba * 100:.1f}% hit probability"
        f"</h2>",
        unsafe_allow_html=True,
    )
    st.progress(proba)

    st.markdown("#### Top 5 contributing features (SHAP)")
    top_idx = contribs.abs().sort_values(ascending=False).head(5).index
    top = contribs.loc[top_idx]

    fig, ax = plt.subplots(figsize=(7, 3))
    colors = ["#1DB954" if v > 0 else "#E22134" for v in top.values]
    ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
    ax.axvline(0, color="#333", lw=0.6)
    ax.set_xlabel("SHAP value  —  positive pushes toward HIT, negative toward FLOP")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ----------------------------------------------------------------- layout --

st.set_page_config(page_title="Spotify Hit Predictor", page_icon="🎵", layout="wide")
st.title("🎵 Spotify Hit Predictor")
st.caption(
    "Predicts whether a song would have landed on the Billboard Hot-100, "
    "based on Spotify audio features. Trained on ~41k songs (50/50 hit/flop, "
    "1960s–2010s). Model: XGBoost (ROC-AUC 0.89)."
)

model, scaler, feature_columns, metrics, explainer = load_artifacts()

tab_sliders, tab_about = st.tabs(["🎚️  Sliders", "ℹ️  About"])

# -------- TAB 1: Sliders --------------------------------------------------
with tab_sliders:
    st.markdown(
        "Pick a famous song from the dataset to pre-fill the sliders, then "
        "tweak whatever you want and hit **Predict**. Or pick *start fresh* "
        "and design a hypothetical song from scratch."
    )

    # Example selector lives OUTSIDE the form so it triggers a rerun and
    # prefills the slider session_state before the form re-renders.
    example_choice = st.selectbox(
        "Load an example song",
        options=[START_FRESH] + list(EXAMPLE_SONGS.keys()),
        key="example_choice",
    )

    if (
        example_choice != START_FRESH
        and st.session_state.get("_last_loaded") != example_choice
    ):
        song = EXAMPLE_SONGS[example_choice]
        for feat, val in song["numeric"].items():
            st.session_state[f"s_{feat}"] = val
        st.session_state["sliders_key"] = song["key"]
        st.session_state["sliders_decade"] = song["decade"]
        st.session_state["_last_loaded"] = example_choice
        st.rerun()

    # Wrapping the widgets in a form means Streamlit only reruns the script
    # (and the SHAP + matplotlib pipeline) when "Predict" is clicked, not on
    # every slider micro-movement. Without this the page visibly flickers.
    with st.form("sliders_form"):
        col_left, col_right = st.columns(2)
        numeric_values: dict[str, float] = {}

        half = len(NUMERIC_FEATURES) // 2 + len(NUMERIC_FEATURES) % 2
        left_feats, right_feats = NUMERIC_FEATURES[:half], NUMERIC_FEATURES[half:]

        for column, feats in ((col_left, left_feats), (col_right, right_feats)):
            with column:
                for feat in feats:
                    lo, hi, default = SLIDER_RANGES[feat]
                    if feat in ("mode", "time_signature", "sections", "duration_ms"):
                        numeric_values[feat] = st.slider(
                            feat, int(lo), int(hi), int(default), key=f"s_{feat}"
                        )
                    else:
                        numeric_values[feat] = st.slider(
                            feat, float(lo), float(hi), float(default), key=f"s_{feat}"
                        )

        col_k, col_d = st.columns(2)
        with col_k:
            key_choice = st.selectbox(
                "key",
                options=list(range(12)),
                key="sliders_key",
                format_func=lambda i: f"{i} — {KEY_NAMES[i]}",
            )
        with col_d:
            decade_choice = st.selectbox(
                "decade", options=DECADES, key="sliders_decade"
            )

        submitted = st.form_submit_button(
            "🎯  Predict", type="primary", use_container_width=True
        )

    if submitted:
        raw_df = build_feature_row(
            numeric_values, key_choice, decade_choice, feature_columns
        )
        proba, contribs = predict_and_explain(
            raw_df, model, scaler, feature_columns, explainer
        )
        st.divider()
        render_prediction(proba, contribs)
    else:
        st.info("Adjust the sliders above (or load an example), then click **Predict**.")


# -------- TAB 2: About ----------------------------------------------------
with tab_about:
    st.markdown(
        """
        ### How it works

        The model is an **XGBoost** classifier trained on the
        [Spotify Hit Predictor](https://www.kaggle.com/datasets/theoverman/the-spotify-hit-predictor-dataset)
        dataset — ~41,000 songs from the 1960s through the 2010s, each labeled
        *hit* (made the Billboard Hot-100) or *flop* (didn't).

        Each track is represented by 14 numeric audio features that Spotify
        computes for every song (danceability, energy, loudness, valence,
        acousticness, instrumentalness, liveness, speechiness, tempo,
        duration, chorus hit, sections, mode, time-signature), plus one-hot
        encodings of musical key and decade. That's **32 features in total**
        after preprocessing.

        Three models were compared on a held-out 20% test set:
        """
    )

    metric_rows = pd.DataFrame(metrics["models"]).set_index("model").round(4)
    st.dataframe(metric_rows, use_container_width=True)

    st.markdown(
        f"**Winner**: `{metrics['winner']}` "
        f"(selected by `{metrics['selection_metric']}`)."
    )

    st.markdown("### Global feature importance")
    fi_path = REPORT_DIR / "feature_importance.png"
    if fi_path.exists():
        st.image(str(fi_path), caption="Top-15 features from the winning model.")

    st.markdown(
        """
        ### Limitations

        - The "hit" label is **Billboard Hot-100 only** — US-centric, ignores
          songs that blew up in other countries or on TikTok.
        - Spotify's audio features describe the *finished master* — not the
          artist's prior fame, marketing budget, music video, or release timing,
          all of which arguably matter more than the audio itself.
        - The dataset stops at 2019; tastes since then aren't represented.
        - Originally this app also had a "paste a Spotify URL" mode that fetched
          features directly from the Spotify Web API, but Spotify deprecated
          the `audio-features` endpoint for new developer apps in November 2024.
          The example-song selector above replaces that functionality using real
          rows from the training dataset.
        """
    )
