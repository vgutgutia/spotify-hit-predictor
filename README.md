---
title: Spotify Hit Predictor
emoji: 🎵
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Predict Billboard Hot-100 hits from Spotify audio features
---

# Spotify Hit Predictor

Final project for Advanced AI Models and Applications.

Predicts whether a song would have made the Billboard Hot-100 from its Spotify audio features. XGBoost, ROC-AUC ~0.89, trained on ~41k songs from the 60s through the 10s.

- Live demo: https://huggingface.co/spaces/vgutgutia/spotify-hit-predictor
- Dataset: https://www.kaggle.com/datasets/theoverman/the-spotify-hit-predictor-dataset
- Writeup: `report/REPORT.md`

## Run locally

```
pip install -r requirements.txt
kaggle datasets download -d theoverman/the-spotify-hit-predictor-dataset -p data/ --unzip
python src/preprocess.py
python src/train.py
streamlit run app/app.py
```
