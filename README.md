# Spotify Hit Predictor

Predict whether a song will land on the Billboard Hot-100 from its Spotify audio features (danceability, energy, valence, tempo, etc.).

Trained on the [Spotify Hit Predictor dataset](https://www.kaggle.com/datasets/theoverman/the-spotify-hit-predictor-dataset) — ~40k songs labeled hit/flop, spanning the 1960s through the 2010s.

## Structure

```
spotify-hit-predictor/
  data/           # per-decade CSVs + combined.csv (gitignored)
  src/            # preprocess.py, train.py, predict.py
  app/            # Streamlit app + requirements
  models/         # trained model, scaler, feature order (gitignored)
  report/         # REPORT.md + EDA plots
```

## Quickstart

```bash
# 1. Pull data (requires ~/.kaggle/kaggle.json)
kaggle datasets download -d theoverman/the-spotify-hit-predictor-dataset -p data/ --unzip

# 2. EDA + preprocess + train
python src/preprocess.py
python src/train.py

# 3. Run app locally
streamlit run app/app.py
```

## Live demo

Deployed on Hugging Face Spaces — link added after step 6.
