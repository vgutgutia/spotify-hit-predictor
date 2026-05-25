"""Streamlit app for the Spotify Hit Predictor."""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import joblib
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

# Grouped semantically rather than alphabetically — left column is "what the
# song feels like," right column is "how it's built."
LEFT_GROUP = {
    "Feel": ["danceability", "energy", "valence", "mode"],
    "Texture": ["acousticness", "instrumentalness", "liveness", "speechiness"],
}
RIGHT_GROUP = {
    "Production": ["loudness", "tempo"],
    "Structure": ["duration_ms", "chorus_hit", "sections", "time_signature"],
}

DECADES = ["60s", "70s", "80s", "90s", "00s", "10s"]
KEY_NAMES = ["C", "C♯/D♭", "D", "D♯/E♭", "E", "F", "F♯/G♭",
             "G", "G♯/A♭", "A", "A♯/B♭", "B"]

EXAMPLE_SONGS: dict[str, dict] = {
    "Can't Buy Me Love — The Beatles (1960s) · hit": {
        "decade": "60s", "key": 0,
        "numeric": {"danceability": 0.844, "energy": 0.299, "loudness": -12.645,
                    "speechiness": 0.0621, "acousticness": 0.848,
                    "instrumentalness": 0.844, "liveness": 0.102, "valence": 0.566,
                    "tempo": 102.088, "duration_ms": 131176, "chorus_hit": 20.009,
                    "sections": 6, "mode": 1, "time_signature": 4},
    },
    "Miss You — The Rolling Stones (1970s) · hit": {
        "decade": "70s", "key": 9,
        "numeric": {"danceability": 0.795, "energy": 0.710, "loudness": -4.746,
                    "speechiness": 0.0392, "acousticness": 0.443,
                    "instrumentalness": 0.0215, "liveness": 0.344, "valence": 0.845,
                    "tempo": 109.689, "duration_ms": 288667, "chorus_hit": 45.452,
                    "sections": 15, "mode": 0, "time_signature": 4},
    },
    "Billie Jean — Michael Jackson (1980s) · hit": {
        "decade": "80s", "key": 11,
        "numeric": {"danceability": 0.920, "energy": 0.654, "loudness": -3.051,
                    "speechiness": 0.0401, "acousticness": 0.0236,
                    "instrumentalness": 0.0158, "liveness": 0.0359, "valence": 0.847,
                    "tempo": 117.046, "duration_ms": 293827, "chorus_hit": 29.591,
                    "sections": 14, "mode": 0, "time_signature": 4},
    },
    "My Name Is — Eminem (1990s) · hit": {
        "decade": "90s", "key": 1,
        "numeric": {"danceability": 0.869, "energy": 0.680, "loudness": -6.233,
                    "speechiness": 0.318, "acousticness": 0.0416,
                    "instrumentalness": 1.12e-06, "liveness": 0.0914, "valence": 0.815,
                    "tempo": 85.519, "duration_ms": 268400, "chorus_hit": 101.419,
                    "sections": 8, "mode": 1, "time_signature": 4},
    },
    "Hotline Bling — Drake (2010s) · hit": {
        "decade": "10s", "key": 2,
        "numeric": {"danceability": 0.891, "energy": 0.625, "loudness": -7.861,
                    "speechiness": 0.0558, "acousticness": 0.00261,
                    "instrumentalness": 0.000176, "liveness": 0.0504,
                    "valence": 0.548, "tempo": 134.967, "duration_ms": 267067,
                    "chorus_hit": 69.390, "sections": 8, "mode": 1, "time_signature": 4},
    },
    "River Man — Nick Drake (1960s) · flop": {
        "decade": "60s", "key": 8,
        "numeric": {"danceability": 0.474, "energy": 0.118, "loudness": -20.098,
                    "speechiness": 0.0339, "acousticness": 0.852,
                    "instrumentalness": 0.735, "liveness": 0.112, "valence": 0.0921,
                    "tempo": 114.520, "duration_ms": 258680, "chorus_hit": 39.473,
                    "sections": 18, "mode": 1, "time_signature": 5},
    },
    "3's & 7's — Queens of the Stone Age (2000s) · flop": {
        "decade": "00s", "key": 10,
        "numeric": {"danceability": 0.438, "energy": 0.990, "loudness": -3.890,
                    "speechiness": 0.108, "acousticness": 0.0651,
                    "instrumentalness": 0.0222, "liveness": 0.359, "valence": 0.452,
                    "tempo": 131.306, "duration_ms": 214067, "chorus_hit": 40.384,
                    "sections": 9, "mode": 0, "time_signature": 4},
    },
}

START_FRESH = "Start fresh"
HIT_GREEN = "#1DB954"
FLOP_RED = "#FF4D5E"


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


def render_verdict(proba: float) -> None:
    is_hit = proba >= 0.5
    label = "HIT" if is_hit else "FLOP"
    color = HIT_GREEN if is_hit else FLOP_RED
    confidence = proba if is_hit else (1 - proba)

    st.markdown(
        f"""
        <div class="verdict-card">
          <div class="verdict-label" style="color:{color};">{label}</div>
          <div class="verdict-prob">{proba * 100:.1f}% hit probability</div>
          <div class="verdict-bar-wrap">
            <div class="verdict-bar" style="width:{proba * 100:.1f}%; background:{color};"></div>
          </div>
          <div class="verdict-conf">model confidence: {confidence * 100:.0f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_shap(contribs: pd.Series) -> None:
    top_idx = contribs.abs().sort_values(ascending=False).head(8).index
    top = contribs.loc[top_idx].reset_index()
    top.columns = ["feature", "shap"]
    top["direction"] = np.where(top["shap"] > 0, "Pushed toward HIT", "Pushed toward FLOP")
    top["abs_shap"] = top["shap"].abs()

    chart = (
        alt.Chart(top)
        .mark_bar(cornerRadiusEnd=3, height=22)
        .encode(
            x=alt.X("shap:Q", title=None,
                    axis=alt.Axis(grid=True, gridColor="#222", domain=False, tickColor="#444")),
            y=alt.Y("feature:N",
                    sort=alt.SortField("abs_shap", order="descending"),
                    title=None, axis=alt.Axis(domain=False, ticks=False)),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=["Pushed toward HIT", "Pushed toward FLOP"],
                    range=[HIT_GREEN, FLOP_RED],
                ),
                legend=alt.Legend(title=None, orient="top", labelColor="#aaa"),
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("shap:Q", title="SHAP value", format="+.3f"),
            ],
        )
        .properties(height=260, padding=0)
        .configure_view(strokeWidth=0)
        .configure_axis(labelColor="#bbb", titleColor="#bbb")
    )
    st.altair_chart(chart, use_container_width=True)


# ----------------------------------------------------------------- layout --

st.set_page_config(
    page_title="Hit Predictor",
    page_icon="●",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom styling — keeps the rest of the file logic-only.
st.markdown(
    """
<style>
:root { --hit:#1DB954; --flop:#FF4D5E; --line:#22222B; --muted:#8E8E94; }

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

.block-container {
    padding-top: 2.5rem; padding-bottom: 4rem;
    max-width: 1120px;
}

.hero { padding-bottom: 1.5rem; border-bottom: 1px solid var(--line); margin-bottom: 1.75rem; }
.hero .eyebrow {
    color: var(--hit); text-transform: uppercase; letter-spacing: 0.14em;
    font-size: 0.72rem; font-weight: 700; margin: 0;
}
.hero h1 {
    font-size: 2.6rem; font-weight: 800; letter-spacing: -0.035em;
    margin: 0.35rem 0 0.4rem 0; line-height: 1.05;
}
.hero p { color: var(--muted); font-size: 1.02rem; margin: 0; max-width: 56ch; line-height: 1.5; }

div[data-testid="stMetric"] { background: #12121A; border: 1px solid var(--line); border-radius: 10px; padding: 0.9rem 1rem; }
div[data-testid="stMetricLabel"] { color: var(--muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; }
div[data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 700; letter-spacing: -0.01em; }

.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; border-bottom: 1px solid var(--line); padding-bottom: 0; }
.stTabs [data-baseweb="tab"] {
    height: 42px; padding: 0 1.1rem; font-weight: 500; color: var(--muted);
    background: transparent; border-radius: 0; border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] { color: #fff; border-bottom-color: var(--hit); background: transparent; }

.slider-group-title {
    color: var(--muted); text-transform: uppercase; letter-spacing: 0.12em;
    font-size: 0.7rem; font-weight: 700; margin: 1rem 0 0.3rem 0;
}

.stSlider [data-baseweb="slider"] > div > div > div { background: #2A2A33 !important; }

.stFormSubmitButton button {
    width: 100%; height: 52px; border-radius: 26px;
    background: var(--hit); border: 0; color: #000; font-weight: 700;
    font-size: 0.95rem; letter-spacing: 0.04em; text-transform: uppercase;
    transition: transform 0.08s ease, filter 0.15s ease;
}
.stFormSubmitButton button:hover { filter: brightness(1.08); }
.stFormSubmitButton button:active { transform: scale(0.99); }

.verdict-card {
    background: linear-gradient(160deg, #14141C 0%, #0E0E14 100%);
    border: 1px solid var(--line); border-radius: 14px;
    padding: 1.6rem 1.9rem; margin: 1.25rem 0 1.5rem 0;
}
.verdict-label { font-size: 2.6rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1; }
.verdict-prob { color: #DDD; font-size: 1.05rem; margin-top: 0.35rem; font-weight: 500; }
.verdict-bar-wrap {
    margin-top: 1rem; height: 6px; background: #1E1E26;
    border-radius: 3px; overflow: hidden;
}
.verdict-bar { height: 100%; border-radius: 3px; transition: width 0.3s ease; }
.verdict-conf { color: var(--muted); font-size: 0.78rem; margin-top: 0.55rem; letter-spacing: 0.02em; }

.section-title {
    font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.16em;
    color: var(--muted); font-weight: 700; margin-top: 1.25rem;
}

.about-card {
    background: #12121A; border: 1px solid var(--line); border-radius: 12px;
    padding: 1.3rem 1.5rem; margin-bottom: 1rem;
}
</style>
    """,
    unsafe_allow_html=True,
)

model, scaler, feature_columns, metrics, explainer = load_artifacts()

# ---- HEADER -------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <p class="eyebrow">XGBoost · 41,106 songs · 1960s–2010s</p>
  <h1>Spotify Hit Predictor</h1>
  <p>Would this song have charted on the Billboard Hot-100? Drag the sliders, load a famous track, and see what the model thinks of it.</p>
</div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
xgb = next(m for m in metrics["models"] if m["model"] == metrics["winner"])
m1.metric("ROC-AUC", f"{xgb['roc_auc']:.3f}")
m2.metric("F1", f"{xgb['f1']:.3f}")
m3.metric("Accuracy", f"{xgb['accuracy'] * 100:.1f}%")
m4.metric("Test songs", "8,222")

st.write("")

tab_predict, tab_about = st.tabs(["Predict", "About"])

# -------- TAB 1: Predict -------------------------------------------------
with tab_predict:

    example_choice = st.selectbox(
        "Load an example song from the dataset",
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

    with st.form("sliders_form"):
        col_left, col_right = st.columns(2, gap="large")
        numeric_values: dict[str, float] = {}

        def render_slider(feat: str) -> None:
            lo, hi, default = SLIDER_RANGES[feat]
            if feat in ("mode", "time_signature", "sections", "duration_ms"):
                numeric_values[feat] = st.slider(
                    feat, int(lo), int(hi), int(default), key=f"s_{feat}"
                )
            else:
                numeric_values[feat] = st.slider(
                    feat, float(lo), float(hi), float(default), key=f"s_{feat}"
                )

        with col_left:
            for group_name, feats in LEFT_GROUP.items():
                st.markdown(f'<div class="slider-group-title">{group_name}</div>',
                            unsafe_allow_html=True)
                for feat in feats:
                    render_slider(feat)

        with col_right:
            for group_name, feats in RIGHT_GROUP.items():
                st.markdown(f'<div class="slider-group-title">{group_name}</div>',
                            unsafe_allow_html=True)
                for feat in feats:
                    render_slider(feat)

            st.markdown('<div class="slider-group-title">Musical context</div>',
                        unsafe_allow_html=True)
            key_choice = st.selectbox(
                "key", options=list(range(12)),
                key="sliders_key",
                format_func=lambda i: f"{KEY_NAMES[i]}",
            )
            decade_choice = st.selectbox(
                "decade", options=DECADES, key="sliders_decade"
            )

        st.write("")
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        raw_df = build_feature_row(
            numeric_values, key_choice, decade_choice, feature_columns
        )
        proba, contribs = predict_and_explain(
            raw_df, model, scaler, feature_columns, explainer
        )
        render_verdict(proba)
        st.markdown('<div class="section-title">Why the model decided this</div>',
                    unsafe_allow_html=True)
        render_shap(contribs)
    else:
        st.caption("Adjust the sliders (or load an example), then click Predict.")


# -------- TAB 2: About ---------------------------------------------------
with tab_about:
    st.markdown(
        """
        <div class="about-card">
        <h3 style="margin-top:0">The model</h3>
        <p style="color:#CCC; line-height:1.6; margin:0;">
        An XGBoost classifier trained on the
        <a href="https://www.kaggle.com/datasets/theoverman/the-spotify-hit-predictor-dataset" style="color:#1DB954;">Spotify Hit Predictor</a>
        dataset — ~41,000 songs from the 1960s through the 2010s, each labeled
        a Billboard Hot-100 hit or a flop, with 14 audio features plus one-hot
        encodings of musical key and decade (32 features after preprocessing).
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Model comparison (test set)</div>',
                unsafe_allow_html=True)
    metric_rows = pd.DataFrame(metrics["models"]).set_index("model").round(4)
    metric_rows.columns = [c.replace("_", " ").title() for c in metric_rows.columns]
    st.dataframe(metric_rows, use_container_width=True)
    st.caption(
        f"Winner: **{metrics['winner']}** — selected by `{metrics['selection_metric']}`."
    )

    st.markdown('<div class="section-title">Global feature importance</div>',
                unsafe_allow_html=True)
    fi_path = REPORT_DIR / "feature_importance.png"
    if fi_path.exists():
        st.image(str(fi_path), caption="Top-15 features from the winning XGBoost model.")

    st.markdown('<div class="section-title">Limitations</div>',
                unsafe_allow_html=True)
    st.markdown(
        """
        - The hit label is **Billboard Hot-100 only** — US-centric, ignores
          songs that blew up in other countries or on TikTok.
        - Spotify's audio features describe the **finished master** — they say
          nothing about artist fame, label marketing, music video, or release
          timing, all of which arguably matter more.
        - The dataset stops at **2019**; tastes since then aren't represented.
        - Originally this app had a "paste a Spotify URL" mode that fetched
          features live from the Spotify Web API. Spotify deprecated
          `/v1/audio-features` for new developer apps in November 2024, so it
          was replaced with the in-dataset example selector above.
        """
    )
