# VolatiLiX — AAPL Volatility XAI Dashboard

Dual-model explainable AI framework for Apple stock analysis.

---

## Models

| Model | Task | Explainer |
|-------|------|-----------|
| Random Forest (200 trees) | Next-20-day volatility regime (High / Low) | LIME + DiCE |
| MLP Regressor (256-128-64) | Next-day closing price | SHAP KernelExplainer + BETA Surrogate + Sensitivity/LRP + Ablation + Saliency |

---

## File Structure

```
volatillix/
├── train_model.py          ← Run once to train all models
├── app.py                  ← Flask server (10 endpoints)
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
├── static/
│   ├── css/style.css
│   └── js/main.js
└── model/                  ← Auto-created by train_model.py
    ├── rf_pipeline.pkl     ← RF sklearn pipeline
    ├── rf_data.csv         ← Feature-engineered dataset
    ├── mlp_model.pkl       ← Trained MLP Regressor
    ├── beta_tree.pkl       ← BETA surrogate decision tree
    ├── ohlcv_scaler.pkl    ← MinMaxScaler for OHLCV
    ├── shap_background.npy ← 100 background samples for SHAP
    ├── scaled_ohlcv.csv    ← Scaled OHLCV (for MLP windows)
    ├── raw_ohlcv.csv       ← Raw prices (for display)
    ├── mlp_dates.json      ← Date → index mapping
    ├── global_grads.npy    ← Pre-computed global gradients
    ├── global_xai.json     ← Global sensitivity + LRP scores
    └── metadata.json       ← Accuracy, RMSE, feature names
```

---

## Setup

```bash
## Setup

# 1 — Install dependencies
pip install -r requirements.txt

# 2 — (Optional) Re-train models from scratch — ~5-10 min
#     Skip this if you want to use the pre-trained models included in /model
python train_model.py

# 3 — Start the web app
python app.py
# → Open http://localhost:5000
```

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET  | `/api/dates` | All available trading dates + metadata |
| POST | `/api/predict` | RF volatility prediction |
| POST | `/api/mlp/predict` | MLP next-day close price + sparkline data |
| POST | `/api/explain/lime` | LIME local explanation weights (RF) |
| POST | `/api/explain/dice` | DiCE counterfactual scenarios (RF) |
| POST | `/api/explain/shap` | SHAP feature + temporal importance (MLP) |
| POST | `/api/explain/beta` | BETA surrogate tree explanation (MLP) |
| POST | `/api/explain/sensitivity` | Sensitivity analysis + LRP proxy (MLP) |
| POST | `/api/explain/ablation` | Ablation + permutation importance (MLP) |
| POST | `/api/explain/saliency` | Saliency map — time × feature heatmap (MLP) |

All POST endpoints accept `{ "date": "YYYY-MM-DD" }`.

---

## XAI Methods

**LIME** — Fits a local linear surrogate around the RF's prediction for one specific date.
Shows which of the 12 technical features pushed the model toward High or Low volatility.

**DiCE** — Generates 3 diverse counterfactual scenarios for the RF prediction.
Answers: *"What minimal feature changes would flip this volatility forecast?"*
Note: DiCE uses random search and takes 20–40 seconds per query.

**SHAP** — KernelExplainer on the MLP's 60-day OHLCV input.
Shows (1) which raw features (Open/High/Low/Close/Volume) contributed most to the price prediction,
and (2) which time-steps in the 60-day window were most influential.
Note: SHAP computation takes 30–60 seconds per query.

**BETA Surrogate Tree** — A shallow decision tree (depth 3) trained to mimic the MLP's outputs.
Provides a globally interpretable proxy with R² fidelity score, node path, and feature importances.

**Sensitivity + LRP** — Numerical gradients of the MLP output with respect to each input.
Sensitivity = mean |∂ŷ/∂x| per feature; LRP proxy = mean |grad × input| per feature.

**Ablation** — Zeroes out each OHLCV feature across all 60 time steps and measures prediction drop.
Permutation importance measures MSE increase when a feature is shuffled on the test set.

**Saliency Map** — Returns the full 60×5 gradient matrix as a normalised heatmap,
showing exactly which (time-step, feature) combinations had the most influence.

---

## Notable Dates

| Date | Event |
|------|-------|
| `2020-03-16` | COVID-19 crash — maximum volatility |
| `2018-12-24` | Christmas Eve dip |
| `2015-08-24` | China-driven flash crash |
| `2021-11-19` | Peak bull run |
| `2022-06-16` | Bear market trough |
| `2020-08-31` | Post stock-split calm |

---

*For academic and research purposes only. Not financial advice.*
