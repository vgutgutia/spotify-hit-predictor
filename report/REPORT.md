# Spotify Hit Predictor - Project Report

Author: Vansh Gutgutia
Course: Advanced AI Models and Applications

- GitHub: https://github.com/vgutgutia/spotify-hit-predictor
- Live demo: https://huggingface.co/spaces/vgutgutia/spotify-hit-predictor

## Project Overview

I trained a classifier that predicts whether a song would have made the Billboard Hot-100 based on its Spotify audio features (danceability, energy, valence, tempo, etc.). It's a binary classification problem.

The pipeline goes from the raw Kaggle CSVs to a Streamlit app deployed on Hugging Face Spaces. The user picks a famous song from a dropdown (or just plays with the sliders), clicks Predict, and gets a hit/flop verdict with a probability and a SHAP explanation of which features pushed that specific song toward or away from being a hit.

I picked this dataset because hit prediction is a problem where the audio features clearly aren't the whole story (artist fame, marketing, timing, music videos all matter a lot too), so I wanted to see how well a model could do with just the audio.

## Dataset

- Source: The Spotify Hit Predictor Dataset on Kaggle (theoverman)
- Link: https://www.kaggle.com/datasets/theoverman/the-spotify-hit-predictor-dataset
- Size: 41,106 songs, split across six per-decade CSVs (1960s-2010s)
- Decade counts: 60s = 8,642, 70s = 7,766, 80s = 6,908, 90s = 5,520, 00s = 5,872, 10s = 6,398
- Target: binary column `target`, 1 if the song made the Hot-100, 0 otherwise
- Class balance: 50/50 by construction (the dataset author paired each hit with a non-hit), so no resampling needed
- Nulls: zero

Each track has Spotify's standard 14 audio features:

| Feature | Description |
|---|---|
| danceability | how suitable for dancing (0-1) |
| energy | perceived intensity (0-1) |
| loudness | overall loudness in dB |
| speechiness | presence of spoken words (0-1) |
| acousticness | confidence the track is acoustic (0-1) |
| instrumentalness | likelihood of no vocals (0-1) |
| liveness | presence of audience (0-1) |
| valence | musical positivity (0-1) |
| tempo | BPM |
| duration_ms | track length in ms |
| key | 0-11 (C through B) |
| mode | major (1) or minor (0) |
| time_signature | meter (3-7) |
| chorus_hit | start time of the chorus in seconds |
| sections | number of sections in the track |

Plus `decade` which I derived from which CSV the row came from. The ID columns (`track`, `artist`, `uri`) got dropped before training.

## Data Cleanup

The dataset was already clean (no nulls, no duplicates), so most of the work was feature engineering for the model:

1. Combined the six per-decade CSVs into one DataFrame and added a `decade` column.
2. Dropped the ID columns (`track`, `artist`, `uri`).
3. One-hot encoded the two categorical features. `key` became 12 binary columns (`key_0` through `key_11`), `decade` became 6. I used one-hot instead of treating them as numeric because neither is ordinal - C isn't "less than" B in any meaningful way, and the 60s isn't "less than" the 70s for the model's purposes.
4. Stratified 80/20 train/test split with `random_state=42` to keep both folds balanced.
5. StandardScaled the 14 numeric features, fit on the training fold only and then applied to test. Doing it the other way (fit on the whole dataset before splitting) leaks test statistics into training. Scaling matters for Logistic Regression and doesn't hurt the tree models.

After preprocessing, every song is a 32-dimensional vector: 14 standardized numerics + 12 key dummies + 6 decade dummies. The `StandardScaler` object and the column order get saved to `models/` so the Streamlit app can reproduce the exact same transformation at inference time.

## Model Info

I trained four models from scratch on the prepared 32,884-row training fold. Nothing pretrained.

| Model | Hyperparameters | Approximate parameter count |
|---|---|---|
| Logistic Regression | `max_iter=1000`, L2 penalty (default), `random_state=42` | 33 (32 weights + 1 bias) |
| Random Forest | `n_estimators=300`, default depth, `random_state=42`, `n_jobs=-1` | ~300 trees of a few thousand nodes each, so ~10^6 split decisions |
| XGBoost | `n_estimators=300`, `learning_rate=0.1`, `eval_metric="logloss"`, `random_state=42`, `n_jobs=-1` | ~300 boosted trees of depth 6 (default), also ~10^5 to 10^6 effective parameters |
| MLP (PyTorch) | 2 hidden layers (64, 32 units), ReLU + Dropout(0.2), Adam lr=1e-3, batch=256, 40 epochs, BCE-with-logits loss, `random_state=42` | 4,225 trainable parameters |

No hyperparameter tuning - I wanted to compare the four algorithms on sensible defaults, not on how much effort I spent tuning each one. `random_state=42` everywhere for reproducibility.

## Architecture

The overall pipeline:

1. Load and concatenate the six decade CSVs into `combined.csv` (41,106 rows, 20 columns).
2. Drop the three ID columns.
3. One-hot encode `key` and `decade`, giving 32 features plus the target.
4. Stratified 80/20 train/test split with `random_state=42`.
5. Fit `StandardScaler` on the train fold only, transform both folds.
6. Fit LR, RF, XGBoost, and the MLP on the training fold, evaluate each on the test fold.
7. Pick the best tree-based model by ROC-AUC, save `best_model.pkl`, `scaler.pkl`, `feature_columns.json`, and `metrics.json` to `models/`.
8. The Streamlit app loads the saved artifacts, takes the user's slider values, rebuilds the feature vector in the same column order the model was trained on, scales it with the saved scaler, calls `predict_proba`, and explains the result with `shap.TreeExplainer`.

### MLP architecture

The MLP is the one model in the comparison that has a real neural architecture worth describing. I built it in PyTorch as a small feedforward network:

```
Input  (32 features)
   |
   v
Linear (32 -> 64)   --> ReLU --> Dropout(0.2)     [hidden layer 1]
   |
   v
Linear (64 -> 32)   --> ReLU --> Dropout(0.2)     [hidden layer 2]
   |
   v
Linear (32 -> 1)    --> sigmoid (at inference)    [output]
```

Total: 4,225 trainable parameters. Trained with Adam (lr=1e-3), `BCEWithLogitsLoss` (numerically stable form of binary cross-entropy), batch size 256, for 40 epochs. Dropout helps regularize since 32k training rows isn't a huge amount of data for a network with this many parameters.

I picked 2 hidden layers because the input is only 32 features and the target is binary - deeper networks would just be overkill on a problem this size and would risk overfitting. The 64 -> 32 funnel shape is a standard "compress toward the decision" choice.

### Other decisions I made along the way:

- Fit the scaler on the train fold only, to avoid leaking test statistics into training.
- One-hot encode `key` and `decade` instead of treating them as numeric, since neither is ordinal.
- Save the scaler and the feature column order to disk alongside the model. Without them, the Streamlit app would silently produce wrong predictions instead of an error.
- Use SHAP for per-prediction explanations rather than just showing the global feature importance. Global tells you what matters on average; SHAP tells you what mattered for the specific song the user is looking at.

## Metrics

I evaluated all three models on the held-out 8,222-song test fold and reported five metrics:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.7414 | 0.7103 | 0.8154 | 0.7592 | 0.8170 |
| MLP (2 hidden layers) | 0.8059 | 0.7816 | 0.8489 | 0.8139 | 0.8824 |
| Random Forest | 0.8075 | 0.7810 | 0.8545 | 0.8161 | 0.8878 |
| XGBoost (winner) | 0.8149 | 0.7881 | 0.8613 | 0.8231 | 0.8899 |

Why these five:

- Accuracy is the obvious metric. It works here because the classes are perfectly balanced (50/50). If they weren't, it'd be misleading.
- Precision and recall let you see how the model trades off false positives vs false negatives. If a record label uses this to pick songs to sign, they probably care about precision (don't waste money on flops the model called hits). If a music recommender uses it, they probably care more about recall (don't miss any actual hits).
- F1 is a single-number balance of precision and recall.
- ROC-AUC measures how well the model separates the two classes regardless of where you set the decision threshold. It's the most informative single number for binary classification, so I used it to pick the winner.

XGBoost won by ROC-AUC (0.8899 vs RF 0.8878 vs MLP 0.8824 vs LR 0.8170). XGBoost is the model that gets saved to `best_model.pkl` and used by the deployed Streamlit app.

## Performance Analysis

The tree models beat Logistic Regression by about 7 points of ROC-AUC. That makes sense - the relationships between audio features aren't linear (a super danceable song that's also 12 minutes long is probably not a hit), and linear models can't pick up on those interactions without explicit feature crosses. XGBoost edges out Random Forest by a tiny margin (0.0021), which suggests the boosting step is helping a little but not by much on this dataset.

The MLP came in between LR and the tree models at 0.8824 ROC-AUC. About 6.5 points better than the linear baseline, but slightly worse than RF and XGBoost. That matches the general pattern I'd read about: on small dense tabular datasets like this, gradient-boosted trees usually beat neural networks. Trees can pick out individual feature thresholds cleanly; the MLP has to learn those same thresholds implicitly through smooth nonlinearities, which is harder when you don't have huge amounts of data or complex high-dimensional interactions to exploit. If the dataset were 10x bigger or had richer features (like raw audio waveforms instead of pre-extracted summary stats), the MLP would probably have a better shot.

All three models have higher recall than precision. XGBoost has precision 0.788 and recall 0.861 - it's slightly trigger-happy about calling songs hits. If I wanted to balance them out, I could push the decision threshold above 0.5, trading some recall for precision.

The feature importance chart (in `report/feature_importance.png`) tells an interesting story:

- `instrumentalness` is by far the most important feature at 17.6%, almost 3x the next highest. Songs without vocals very rarely become Billboard hits, since the chart is mostly vocal pop and hip-hop.
- The next tier is decade buckets and `acousticness` at around 5% each. Decade matters because what makes a 60s hit is different from what makes a 2010s hit.
- `danceability`, `mode`, and `duration_ms` are in the next group at 4-5%.
- The rest of the features each contribute less than 4%, so the signal is reasonably spread out rather than concentrated.

The fact that the decade features show up so high in feature importance confirms what I expected: a single time-blind model would underfit.

## Limitations and Ethics

1. The "hit" label is Billboard Hot-100 only. That's a US chart. A song that was huge in the UK, Brazil, Korea, or on TikTok but didn't crack the Hot-100 counts as a "flop" here. The model is really predicting "would this have charted in the US," not "is this a good song" or "is this song popular globally."

2. The dataset stops at 2019. Anything that's happened in pop music since (TikTok-driven viral hits, K-pop in the US, streaming-era changes) is invisible to the model.

3. Spotify's audio features describe the finished master. They say nothing about the artist's existing fame, the label's marketing budget, the music video, sync placements, or release timing. Industry research consistently shows those factors matter more than pure audio in determining whether a song becomes a hit. So the model is closer to "which audio profiles fit the pattern of past hits" than "which songs will be hits."

4. The "flop" pool has survivorship bias. The non-hits in the dataset still made it onto Spotify and got their features measured. Truly obscure tracks (most music ever made) aren't in there. So "flop" here means "released but didn't chart," not "ignored by everyone."

5. Ethical considerations. A model like this used by labels to deprioritize artists whose music doesn't match "hit" patterns would entrench existing tastes and disadvantage music from underrepresented styles, languages, or eras. The model output should be one signal among many, not a gatekeeper.

6. Spotify deprecated the `audio-features` endpoint for new developer apps in November 2024. That blocked the "paste a Spotify URL" mode I originally wanted to include - new dev apps now get a 403 from that endpoint. I replaced it with a dropdown of example songs pulled directly from the training data, so users can still see the model react to real music. If I rebuilt this today, I'd probably use `librosa` to compute the audio features locally from each song's 30-second preview MP3.

## Reflection

The biggest thing I learned is that the model is the easy part. Training was maybe an hour of scikit-learn and xgboost on defaults, and it got to ~89% ROC-AUC on the first try with no tuning. The hard part was everything around it: making sure the feature vector at inference time matches the column order the model was trained on, fixing the Streamlit form so the page didn't flicker every time someone dragged a slider, getting the Hugging Face Docker deploy to accept binary files (it wanted LFS), and dealing with Spotify deprecating the API my whole URL mode depended on right after I'd built it.

Two concrete things I'll take with me:

1. Always save the scaler and the feature column order alongside the model. Without them, the model is technically usable but will silently produce wrong predictions because the input vector won't match what it was trained on.

2. Fit the scaler on the training fold only. It's a one-line difference but fitting on the full dataset before splitting leaks test statistics into training, and inflates the metrics by an amount you'll never catch unless you know to look for it.

If I had more time I'd tune the XGBoost hyperparameters with `RandomizedSearchCV` (probably worth another point or two of ROC-AUC), try a small neural network for comparison, and calibrate the probabilities so that "80% hit probability" actually corresponds to about 80% of songs at that score being hits.

But the lesson I want to remember is that a working 0.89 ROC-AUC classifier deployed where anyone can use it is more useful than a 0.95 ROC-AUC notebook sitting on my laptop.
