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

# spotify hit predictor

final project for advanced ai models and applications.

predicts whether a song would have made the billboard hot-100 based on its spotify audio features. xgboost classifier, ~0.89 roc-auc on the test set, trained on ~41k songs from the 60s through the 10s.

live demo: https://huggingface.co/spaces/vgutgutia/spotify-hit-predictor

writeup with the metrics, limitations, and reflection is in `report/REPORT.md`.

## run it

```
pip install -r requirements.txt
pip install torch  # only needed if you're going to retrain the mlp
kaggle datasets download -d theoverman/the-spotify-hit-predictor-dataset -p data/ --unzip
python src/preprocess.py
python src/train.py
streamlit run app/app.py
```
