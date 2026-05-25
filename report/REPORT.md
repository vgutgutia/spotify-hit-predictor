# Spotify Hit Predictor — Project Report

**Author**: Vansh Gutgutia
**Course**: Advanced AI Models and Applications
**Repository**: https://github.com/vgutgutia/spotify-hit-predictor
**Live demo**: https://huggingface.co/spaces/vgutgutia/spotify-hit-predictor

---

## Project Overview

This project takes a song's measurable audio characteristics — danceability, energy, tempo, valence, and so on — and predicts whether that song would have made the Billboard Hot-100. It's framed as binary supervised classification.

The full pipeline goes from raw Kaggle CSVs to a deployed, interactive Streamlit web app on Hugging Face Spaces. A user picks a famous song from a dropdown of pre-loaded examples (Beatles, Michael Jackson, Drake, etc.) to instantly pre-fill the audio-feature sliders, then either hits Predict to see how the model scores that real track or tweaks the sliders to design a hypothetical song. The app shows a hit/flop verdict, a hit probability, and a SHAP-based breakdown of which features pushed *this specific song* toward or away from being a hit.

Originally the app was going to include a second "paste a Spotify URL" mode that fetched a song's audio features live from the Spotify Web API. I built and deployed it, but Spotify deprecated the `/v1/audio-features` endpoint for new developer apps in November 2024 — the endpoint now returns 403 unless the app was grandfathered into extended-quota mode. Rather than ship a feature that would silently fail for anyone reviewing the project, I replaced it with the example-song selector, which pulls real rows directly from the training dataset. The UX trade-off is small (you can't try arbitrary songs, but you can try a handpicked seven that span the dataset) and the demo is now self-contained — no external API dependencies, no secrets needed, nothing can break at runtime.

I chose this dataset because hit prediction sits in a genuinely interesting place — the audio features capture a lot, but they obviously don't capture everything (artist fame, marketing, timing, music videos), so there's a real ceiling on what any model can do here. That tension was part of what I wanted to explore.

## Dataset

- **Source**: [The Spotify Hit Predictor Dataset](https://www.kaggle.com/datasets/theoverman/the-spotify-hit-predictor-dataset) on Kaggle (theoverman)
- **Size**: 41,106 songs total, split across six per-decade CSVs (1960s through 2010s)
- **Decade counts**: 60s = 8,642, 70s = 7,766, 80s = 6,908, 90s = 5,520, 00s = 5,872, 10s = 6,398
- **Target**: binary `target` column — `1` if the song made the Billboard Hot-100, `0` otherwise
- **Class balance**: **perfectly 50/50** by construction (the dataset's author paired each hit with a non-hit), so no class weighting or resampling was needed
- **Nulls**: zero, in every column

**Input features** — each track comes with Spotify's standard audio features plus a couple of derived ones:

| Feature | What it measures |
|---|---|
| `danceability` | how suitable for dancing (0–1) |
| `energy` | perceived intensity (0–1) |
| `loudness` | overall loudness in dB |
| `speechiness` | presence of spoken words (0–1) |
| `acousticness` | confidence the track is acoustic (0–1) |
| `instrumentalness` | likelihood of no vocals (0–1) |
| `liveness` | presence of audience (0–1) |
| `valence` | musical positivity (0–1) |
| `tempo` | BPM |
| `duration_ms` | track length in ms |
| `key` | 0–11 (C through B) |
| `mode` | major (1) / minor (0) |
| `time_signature` | meter (3–7) |
| `chorus_hit` | start time of the chorus in seconds |
| `sections` | number of sections in the track |
| `decade` | derived from which CSV the row came from |

Identifiers (`track`, `artist`, `uri`) were dropped before training — they're useful only for human inspection.

## Data Cleanup

The dataset arrived clean (no missing values, consistent dtypes, no duplicates), so most "cleanup" was actually feature engineering for the model:

1. **Combined the six per-decade CSVs** into a single 41,106-row DataFrame, adding a `decade` column derived from the source filename (60s, 70s, ..., 10s).
2. **Dropped identifier columns** (`track`, `artist`, `uri`) — they leak nothing useful to a classifier.
3. **One-hot encoded** the two categorical features:
   - `key` → 12 binary columns (`key_0` through `key_11`) — musical key has no natural ordering, so treating 0–11 as numeric would imply C is "less than" B in some meaningful way, which is wrong.
   - `decade` → 6 binary columns. Same reason — decade is categorical, and we want the model to learn separate effects per era.
4. **Stratified 80/20 train/test split** (`random_state=42`) on the target so both folds keep the 50/50 balance.
5. **StandardScaled the 14 numeric features** — fit on the training fold only, then applied to test. Doing it in the other order (fit on the full dataset, then split) would leak test-set statistics into training. Standardization matters for Logistic Regression (which is scale-sensitive) and is harmless for the tree models.

After preprocessing, every song is a 32-dimensional vector: 14 standardized numerics + 12 key dummies + 6 decade dummies. The `StandardScaler` object and the column order were persisted to `models/` so the Streamlit app can reproduce the exact same transformation on any new song at inference time.

## Model Info

I trained **three models from scratch** on the prepared data and picked a winner. Nothing pretrained — these are all classical models trained on the 32,884-row training fold.

| Model | Hyperparameters | Parameter count (approx.) |
|---|---|---|
| Logistic Regression | `max_iter=1000`, L2 penalty (default), `random_state=42` | 33 (32 weights + 1 bias) |
| Random Forest | `n_estimators=300`, default depth, `random_state=42`, `n_jobs=-1` | ~300 trees, each typically a few thousand nodes — call it ~10⁶ split decisions total |
| XGBoost | `n_estimators=300`, `learning_rate=0.1`, `eval_metric="logloss"`, `random_state=42`, `n_jobs=-1` | ~300 boosted trees of depth 6 (default) — also ~10⁵–10⁶ effective parameters |

No hyperparameter tuning — I deliberately kept the comparison apples-to-apples on sensible defaults. Defaults are also more honest as a baseline: they show what each algorithm gets you without the noise of tuning effort. Adding `RandomizedSearchCV` on the winner was on the table but felt like over-engineering for a dataset this size.

`random_state=42` everywhere for reproducibility.

## Architecture

The "architecture" picture for this project is the **pipeline**, not a neural network diagram:

```
six decade CSVs
      │
      ▼
load_and_combine()  →  combined.csv (41,106 × 20)
      │
      ▼
drop {track, artist, uri}  →  17 cols + target
      │
      ▼
one-hot encode {key, decade}  →  33 cols + target
      │
      ▼
stratified 80/20 split (random_state=42)
      │
      ├──────── train fold ────────┐
      │                             ▼
      │                       fit StandardScaler on 14 numerics
      │                             │
      ▼                             ▼
   test fold ──────► transform with same scaler
      │
      ▼
fit {LR, RF, XGBoost} on train, evaluate on test
      │
      ▼
pick winner by ROC-AUC → save best_model.pkl + scaler + feature_columns
      │
      ▼
Streamlit app loads artifacts, accepts sliders OR a Spotify URL,
re-applies the saved transformation, calls model.predict_proba,
and explains the result with shap.TreeExplainer
```

The deliberate architectural decisions were:
- **Fit the scaler on the train fold only.** Avoids subtle leakage that would inflate test metrics.
- **One-hot rather than ordinal encoding for `key` and `decade`.** Both are nominal — there's no meaningful "C < C#" or "60s < 70s" ordering.
- **Persist the scaler + the exact feature column order to disk.** The Streamlit app rebuilds each input vector in the same column order the model was trained on; otherwise predictions silently corrupt.
- **`shap.TreeExplainer` for explanations rather than global feature importance.** Global importance tells you what matters *on average*. SHAP tells you what mattered *for this specific song* — which is what a user actually wants when they're staring at a prediction.

## Metrics

Five metrics were computed on the held-out 8,222-song test fold:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.7414 | 0.7103 | 0.8154 | 0.7592 | 0.8170 |
| Random Forest       | 0.8075 | 0.7810 | 0.8545 | 0.8161 | 0.8878 |
| **XGBoost (winner)** | **0.8149** | **0.7881** | **0.8613** | **0.8231** | **0.8899** |

**Why these five:**
- **Accuracy** is the obvious one and is meaningful here *because the classes are perfectly balanced* — with skewed classes it would mislead, but at 50/50 it's a clean read on overall correctness.
- **Precision and recall** capture the asymmetric costs you might care about in practice. A hit-predictor labeling a flop as a hit (low precision) wastes A&R money. Labeling a hit as a flop (low recall) means missing a winner. Reporting both lets a downstream consumer pick the trade-off.
- **F1** is the harmonic mean of precision and recall — a single number that penalizes lopsided models. Useful as the "balanced" headline.
- **ROC-AUC** measures discrimination ability *threshold-free*. It's the most informative single metric for binary classification because it doesn't depend on where you set the cutoff (the default 0.5 is arbitrary). I used it as the **selection metric** for the winner.

The winner was selected by ROC-AUC: **XGBoost edges out Random Forest by 0.0021** (0.8899 vs 0.8878), with Logistic Regression a clear step behind at 0.8170.

## Performance Analysis

**XGBoost won, but it's a photo finish with Random Forest.** That tells me the additional regularization and stagewise refitting of gradient boosting are worth ~0.2 points of ROC-AUC over bagged trees on this data — real but small. Both tree models are about 7 points of ROC-AUC ahead of Logistic Regression, which is consistent with the hypothesis I started with: hit prediction is non-linear (e.g. "danceable but also 12 minutes long" is not a hit), and linear models can't capture interactions like that without explicit feature crosses.

**All three models have recall > precision.** Looking at XGBoost: precision 0.788, recall 0.861. The model is slightly *over-eager* to label songs as hits. In practical terms it catches more real hits at the cost of more false positives. Adjusting the classification threshold above 0.5 would trade some recall for precision; the right place to put it depends on the application (an A&R analyst wants high precision; a music recommender might prefer high recall).

**The feature importance story is striking.** Looking at the top-15 chart in `feature_importance.png`:

- `instrumentalness` dominates at **17.6%** — roughly 3× the next-most-important feature. Songs with vocals are massively more likely to be hits. Pure instrumental tracks (jazz interludes, classical pieces, lounge music) almost never crack the Hot-100, which is mostly vocal pop/hip-hop.
- The next tier is decade buckets and `acousticness` (~5% each). Era matters — the model learns that what makes a 60s hit is different from what makes a 2010s hit.
- `danceability`, `mode`, and `duration_ms` round out the top group at ~4–5% each.
- Most other features sit below 4%, meaning the signal is reasonably *distributed* across many features rather than concentrated in a few.

The fact that `decade` features land in the top 10 is itself interesting — it confirms my suspicion that "what's a hit" drifts meaningfully across eras, and a single time-blind model would underfit.

## Limitations and Ethics

The headline number (89% ROC-AUC) is real but should be read carefully:

1. **"Hit" is defined as Billboard Hot-100 only.** That's a US-centric chart from a single country's music industry. A song that was huge in the UK, Brazil, Korea, or on TikTok but didn't crack the Hot-100 is labeled a "flop" by this model. The model is really predicting "would this song have charted in the US," not "is this song good" or "is this song popular globally."
2. **The dataset cuts off at 2019.** Anything that's happened in pop music since — the TikTok-driven viral cycle, the rise of K-pop in the US, shifts in streaming economics — is invisible to the model. Predictions on 2020+ songs are out-of-distribution.
3. **Spotify's audio features describe the finished master, not the song's path to a hit.** The model can't see the artist's existing fame, the label's marketing budget, the music video, the timing of the release, sync placements, or playlist editorial decisions. Industry research consistently shows those factors swamp pure audio characteristics in determining commercial success. The model is essentially predicting *which audio profiles are consistent with hits*, not *which songs will become hits* — a subtler distinction than the UI implies.
4. **Survivorship and selection bias in the "flop" pool.** The non-hit songs in the dataset were songs that made it onto Spotify at all and got their audio features measured. Truly obscure tracks are absent. So "flop" here means "released-but-not-charting," not "made by a random person and ignored."
5. **Ethical use.** A model like this could be used by labels to deprioritize artists whose music doesn't fit "hit" patterns, which would entrench existing tastes and disadvantage music from underrepresented styles, languages, or eras. Treating model output as gospel rather than as one signal among many would be a mistake.
6. **The Spotify audio-features API was deprecated for new developer apps in November 2024.** That blocked the original "paste a Spotify URL" mode I'd planned (and built — the code is still in git history). The deployed app side-steps this entirely by shipping seven real example songs from the training data so users can still see the model react to *actual* music rather than only hand-crafted slider configurations. If I rebuilt this today and wanted the URL mode back, I'd either compute the features locally from the song's 30-second preview MP3 using `librosa`, or use a community alternative like ReccoBeats that reverse-engineers comparable features.

I tried to be honest about these limitations in the app itself — the About tab calls out the Billboard-only label and the dataset cutoff so users see them before drawing conclusions.

## Reflection

The thing that surprised me most was how much harder the "wrap it as a real product" half of the project was than the "train a model" half. Training was a quiet hour of `scikit-learn` and `xgboost` defaults that produced ~89% ROC-AUC on the first try with no tuning. The wrap-it half — making sure the feature vector at inference time matches the column order the model was trained on, getting the Streamlit form to stop flickering on every slider drag (the fix is `st.form`, batches widget updates), discovering halfway through deployment that Spotify had deprecated the API my whole URL mode depended on, then pivoting to example songs that ship inside the repo — took longer than everything else combined.

Two specific lessons I'll take with me:

1. **Persisting *exactly* what's needed for inference matters more than I thought.** The model file alone is useless without the scaler and the feature column order. Forgetting any of the three would silently produce wrong predictions instead of an error. Saving them as a coordinated bundle from the training script is non-optional.
2. **Don't fit the scaler on the full dataset before splitting.** It's a one-line difference but it leaks test statistics into training and inflates metrics by an amount you'll never catch unless you know to look for it. The textbook order (split → fit on train → transform both) is textbook for a reason.

If I had more time I'd tune XGBoost hyperparameters with `RandomizedSearchCV` (probably gets another 1–2 ROC-AUC points), try a small MLP for comparison, and look at calibrating the probabilities — right now `0.8` from the model doesn't necessarily mean "80% likely to be a hit" in a frequentist sense, and a Platt or isotonic calibration step would make the probability bar in the UI more honest.

But the core lesson is: a 0.89 ROC-AUC classifier deployed as a working web app that anyone can paste a Spotify URL into is more *useful* than a 0.95 ROC-AUC notebook nobody can run.
