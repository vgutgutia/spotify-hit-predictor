"""Streamlit app for the Spotify Hit Predictor.

Three tabs:
  1. Sliders        — manually set audio features → live prediction
  2. Spotify URL    — paste a track URL, app fetches features via spotipy
  3. About          — model summary + global feature-importance chart

Both prediction modes render a verdict (hit/flop), the hit probability, and
a SHAP-based breakdown of which features pushed THIS song toward / away from
being a hit.
"""
from __future__ import annotations

import json
import re
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
    # XGBoost binary: shap_values shape (1, n_features)
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


# ----------------------------------------------------------- spotify utils --

TRACK_ID_RE = re.compile(
    r"(?:spotify:track:|open\.spotify\.com/(?:intl-[a-z]+/)?track/)([A-Za-z0-9]{22})"
)


def extract_track_id(s: str) -> str | None:
    s = s.strip()
    m = TRACK_ID_RE.search(s)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9]{22}", s):
        return s
    return None


def year_to_decade(year: int) -> str:
    if year < 1970:
        return "60s"
    if year < 1980:
        return "70s"
    if year < 1990:
        return "80s"
    if year < 2000:
        return "90s"
    if year < 2010:
        return "00s"
    return "10s"


@st.cache_data(show_spinner=False)
def fetch_spotify_features(track_id: str, _client_id: str, _client_secret: str):
    """Pull audio features + audio analysis + release year for a Spotify track.

    Returns a dict with the same keys the model expects, plus track/artist names.
    """
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials

    sp = spotipy.Spotify(
        auth_manager=SpotifyClientCredentials(
            client_id=_client_id, client_secret=_client_secret
        ),
        requests_timeout=15,
    )

    features = sp.audio_features([track_id])[0]
    if features is None:
        raise RuntimeError("Spotify returned no audio features for this track.")

    track = sp.track(track_id)
    release_date = track["album"]["release_date"]
    year = int(release_date.split("-")[0])

    try:
        analysis = sp.audio_analysis(track_id)
        sections = analysis.get("sections", [])
        sections_count = len(sections)
        chorus_hit = float(sections[1]["start"]) if len(sections) > 1 else 0.0
    except Exception:
        sections_count = 0
        chorus_hit = 0.0

    return {
        "track_name": track["name"],
        "artist_name": ", ".join(a["name"] for a in track["artists"]),
        "release_year": year,
        "decade": year_to_decade(year),
        "key": int(features["key"]),
        "numeric": {
            "danceability": float(features["danceability"]),
            "energy": float(features["energy"]),
            "loudness": float(features["loudness"]),
            "speechiness": float(features["speechiness"]),
            "acousticness": float(features["acousticness"]),
            "instrumentalness": float(features["instrumentalness"]),
            "liveness": float(features["liveness"]),
            "valence": float(features["valence"]),
            "tempo": float(features["tempo"]),
            "duration_ms": int(features["duration_ms"]),
            "chorus_hit": float(chorus_hit),
            "sections": int(sections_count),
            "mode": int(features["mode"]),
            "time_signature": int(features["time_signature"]),
        },
    }


# ----------------------------------------------------------------- layout --

st.set_page_config(page_title="Spotify Hit Predictor", page_icon="🎵", layout="wide")
st.title("🎵 Spotify Hit Predictor")
st.caption(
    "Predicts whether a song would have landed on the Billboard Hot-100, "
    "based on Spotify audio features. Trained on ~41k songs (50/50 hit/flop, "
    "1960s–2010s). Model: XGBoost (ROC-AUC 0.89)."
)

model, scaler, feature_columns, metrics, explainer = load_artifacts()

tab_sliders, tab_url, tab_about = st.tabs(
    ["🎚️  Sliders", "🔗  Spotify URL", "ℹ️  About"]
)

# -------- TAB 1: Sliders --------------------------------------------------
with tab_sliders:
    st.markdown(
        "Drag the sliders to design a song, then see if our model thinks it'd hit."
    )

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
                index=0,
                format_func=lambda i: f"{i} — {KEY_NAMES[i]}",
            )
        with col_d:
            decade_choice = st.selectbox("decade", options=DECADES, index=5)

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
        st.info("Adjust the sliders above, then click **Predict**.")


# -------- TAB 2: Spotify URL ---------------------------------------------
with tab_url:
    st.markdown(
        "Paste a Spotify track URL (or `spotify:track:…` URI, or just the 22-char ID). "
        "The app pulls the song's actual audio features from Spotify, then runs the "
        "same model on them."
    )
    st.warning(
        "⚠️  **Heads up**: Spotify deprecated the `audio-features` endpoint for "
        "developer apps registered after November 2024. If your Spotify app isn't "
        "in extended-quota mode, this tab will return a 403 — that's a Spotify "
        "policy issue, not a bug in the app. The **Sliders** tab works without "
        "any external API."
    )

    # st.secrets raises if no secrets.toml exists at all. Fall back to env
    # vars so HF Spaces' Docker SDK (which exposes secrets as env vars rather
    # than a TOML file) works too.
    import os
    def _secret(key: str) -> str:
        try:
            v = st.secrets.get(key, "")
            if v:
                return v
        except Exception:
            pass
        return os.environ.get(key, "")
    client_id = _secret("SPOTIFY_CLIENT_ID")
    client_secret = _secret("SPOTIFY_CLIENT_SECRET")

    if not client_id or not client_secret:
        st.warning(
            "⚠️  Spotify credentials not configured. Add `SPOTIFY_CLIENT_ID` and "
            "`SPOTIFY_CLIENT_SECRET` to `.streamlit/secrets.toml` locally, or to "
            "the HF Space's secrets. (See README for the schema.) "
            "The Sliders tab still works without credentials."
        )
    else:
        url = st.text_input(
            "Spotify track URL",
            placeholder="https://open.spotify.com/track/6rqhFgbbKwnb9MLmUQDhG6",
        )
        go = st.button("Predict", type="primary")

        if go and url:
            track_id = extract_track_id(url)
            if not track_id:
                st.error("Couldn't parse a Spotify track ID out of that input.")
            else:
                try:
                    with st.spinner("Fetching features from Spotify…"):
                        info = fetch_spotify_features(track_id, client_id, client_secret)
                except Exception as e:
                    msg = str(e)
                    if "403" in msg:
                        st.error(
                            "Spotify returned **403 Forbidden** on the "
                            "`audio-features` endpoint. This means your "
                            "Spotify developer app doesn't have access to "
                            "that endpoint — almost certainly because Spotify "
                            "deprecated it for new apps in Nov 2024. There's "
                            "nothing wrong with your credentials. Use the "
                            "**Sliders** tab to design a song manually instead."
                        )
                    else:
                        st.error(f"Spotify API error: {e}")
                else:
                    st.subheader(f"🎶  {info['track_name']}")
                    st.caption(
                        f"{info['artist_name']}  •  released {info['release_year']}  "
                        f"•  decade bucket: **{info['decade']}**  "
                        f"•  key: **{KEY_NAMES[info['key']]}**"
                    )

                    with st.expander("Fetched audio features"):
                        st.json(info["numeric"])

                    raw_df = build_feature_row(
                        info["numeric"], info["key"], info["decade"], feature_columns
                    )
                    proba, contribs = predict_and_explain(
                        raw_df, model, scaler, feature_columns, explainer
                    )
                    st.divider()
                    render_prediction(proba, contribs)


# -------- TAB 3: About ----------------------------------------------------
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
        """
    )
