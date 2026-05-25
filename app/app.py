"""Streamlit app for the Spotify Hit Predictor."""
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

SLIDER_RANGES = {
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
KEY_NAMES = ["C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb",
             "G", "G#/Ab", "A", "A#/Bb", "B"]

EXAMPLE_SONGS = {
    "Can't Buy Me Love - The Beatles (60s, hit)": {
        "decade": "60s", "key": 0,
        "numeric": {"danceability": 0.844, "energy": 0.299, "loudness": -12.645,
                    "speechiness": 0.0621, "acousticness": 0.848,
                    "instrumentalness": 0.844, "liveness": 0.102, "valence": 0.566,
                    "tempo": 102.088, "duration_ms": 131176, "chorus_hit": 20.009,
                    "sections": 6, "mode": 1, "time_signature": 4},
    },
    "Miss You - The Rolling Stones (70s, hit)": {
        "decade": "70s", "key": 9,
        "numeric": {"danceability": 0.795, "energy": 0.710, "loudness": -4.746,
                    "speechiness": 0.0392, "acousticness": 0.443,
                    "instrumentalness": 0.0215, "liveness": 0.344, "valence": 0.845,
                    "tempo": 109.689, "duration_ms": 288667, "chorus_hit": 45.452,
                    "sections": 15, "mode": 0, "time_signature": 4},
    },
    "Billie Jean - Michael Jackson (80s, hit)": {
        "decade": "80s", "key": 11,
        "numeric": {"danceability": 0.920, "energy": 0.654, "loudness": -3.051,
                    "speechiness": 0.0401, "acousticness": 0.0236,
                    "instrumentalness": 0.0158, "liveness": 0.0359, "valence": 0.847,
                    "tempo": 117.046, "duration_ms": 293827, "chorus_hit": 29.591,
                    "sections": 14, "mode": 0, "time_signature": 4},
    },
    "My Name Is - Eminem (90s, hit)": {
        "decade": "90s", "key": 1,
        "numeric": {"danceability": 0.869, "energy": 0.680, "loudness": -6.233,
                    "speechiness": 0.318, "acousticness": 0.0416,
                    "instrumentalness": 1.12e-06, "liveness": 0.0914, "valence": 0.815,
                    "tempo": 85.519, "duration_ms": 268400, "chorus_hit": 101.419,
                    "sections": 8, "mode": 1, "time_signature": 4},
    },
    "Hotline Bling - Drake (10s, hit)": {
        "decade": "10s", "key": 2,
        "numeric": {"danceability": 0.891, "energy": 0.625, "loudness": -7.861,
                    "speechiness": 0.0558, "acousticness": 0.00261,
                    "instrumentalness": 0.000176, "liveness": 0.0504,
                    "valence": 0.548, "tempo": 134.967, "duration_ms": 267067,
                    "chorus_hit": 69.390, "sections": 8, "mode": 1, "time_signature": 4},
    },
    "River Man - Nick Drake (60s, flop)": {
        "decade": "60s", "key": 8,
        "numeric": {"danceability": 0.474, "energy": 0.118, "loudness": -20.098,
                    "speechiness": 0.0339, "acousticness": 0.852,
                    "instrumentalness": 0.735, "liveness": 0.112, "valence": 0.0921,
                    "tempo": 114.520, "duration_ms": 258680, "chorus_hit": 39.473,
                    "sections": 18, "mode": 1, "time_signature": 5},
    },
    "3's & 7's - Queens of the Stone Age (00s, flop)": {
        "decade": "00s", "key": 10,
        "numeric": {"danceability": 0.438, "energy": 0.990, "loudness": -3.890,
                    "speechiness": 0.108, "acousticness": 0.0651,
                    "instrumentalness": 0.0222, "liveness": 0.359, "valence": 0.452,
                    "tempo": 131.306, "duration_ms": 214067, "chorus_hit": 40.384,
                    "sections": 9, "mode": 0, "time_signature": 4},
    },
}

START_FRESH = "Start fresh"


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / "best_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    feature_columns = json.loads((MODELS_DIR / "feature_columns.json").read_text())
    metrics = json.loads((MODELS_DIR / "metrics.json").read_text())
    explainer = shap.TreeExplainer(model)
    return model, scaler, feature_columns, metrics, explainer


def build_feature_row(numeric, key, decade, feature_columns):
    row = {col: 0 for col in feature_columns}
    for k, v in numeric.items():
        if k in row:
            row[k] = v
    row[f"key_{key}"] = 1
    row[f"decade_{decade}"] = 1
    return pd.DataFrame([row], columns=feature_columns)


def predict_and_explain(raw_df, model, scaler, feature_columns, explainer):
    scaled = raw_df.copy()
    scaled[NUMERIC_FEATURES] = scaler.transform(scaled[NUMERIC_FEATURES])
    proba = float(model.predict_proba(scaled)[0, 1])
    shap_values = explainer.shap_values(scaled)
    contribs = pd.Series(np.asarray(shap_values)[0], index=feature_columns)
    return proba, contribs


st.title("Spotify Hit Predictor")
st.write(
    "Predicts whether a song would have made the Billboard Hot-100 based on its "
    "Spotify audio features. Trained on ~41,000 songs from the 1960s to the 2010s."
)

model, scaler, feature_columns, metrics, explainer = load_artifacts()

tab1, tab2 = st.tabs(["Predict", "About"])

with tab1:
    example_choice = st.selectbox(
        "Load an example song",
        [START_FRESH] + list(EXAMPLE_SONGS.keys()),
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

    with st.form("sliders_form"):
        col1, col2 = st.columns(2)
        numeric_values = {}
        half = len(NUMERIC_FEATURES) // 2 + len(NUMERIC_FEATURES) % 2
        left_feats = NUMERIC_FEATURES[:half]
        right_feats = NUMERIC_FEATURES[half:]

        for column, feats in [(col1, left_feats), (col2, right_feats)]:
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
                list(range(12)),
                key="sliders_key",
                format_func=lambda i: KEY_NAMES[i],
            )
        with col_d:
            decade_choice = st.selectbox("decade", DECADES, key="sliders_decade")

        submitted = st.form_submit_button("Predict")

    if submitted:
        raw_df = build_feature_row(
            numeric_values, key_choice, decade_choice, feature_columns
        )
        proba, contribs = predict_and_explain(
            raw_df, model, scaler, feature_columns, explainer
        )

        if proba >= 0.5:
            st.success(f"HIT - {proba * 100:.1f}% hit probability")
        else:
            st.error(f"FLOP - {proba * 100:.1f}% hit probability")
        st.progress(proba)

        st.subheader("Top contributing features (SHAP)")
        top_idx = contribs.abs().sort_values(ascending=False).head(8).index
        top = contribs.loc[top_idx]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(top.index[::-1], top.values[::-1])
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("SHAP value (positive = toward hit)")
        st.pyplot(fig)
        plt.close(fig)

with tab2:
    st.header("About")
    st.write(
        "XGBoost classifier trained on the Spotify Hit Predictor dataset "
        "(~41,000 songs from the 1960s through the 2010s, perfectly balanced "
        "hit vs flop). 14 numeric audio features plus one-hot encodings of "
        "key and decade = 32 features after preprocessing."
    )

    st.subheader("Model comparison (test set)")
    metric_rows = pd.DataFrame(metrics["models"]).set_index("model").round(4)
    st.dataframe(metric_rows)
    st.write(f"Winner: {metrics['winner']} (selected by {metrics['selection_metric']})")

    st.subheader("Feature importance")
    fi_path = REPORT_DIR / "feature_importance.png"
    if fi_path.exists():
        st.image(str(fi_path))

    st.subheader("Limitations")
    st.write(
        "- Billboard Hot-100 only, US-centric\n"
        "- Audio features don't capture artist fame, marketing, music videos, etc.\n"
        "- Dataset cuts off at 2019\n"
        "- Originally had a Spotify URL mode but Spotify deprecated the "
        "audio-features API for new dev apps in Nov 2024"
    )
