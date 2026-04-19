"""
app.py  —  VolatiLiX Flask Backend  (Complete XAI Suite)
==========================================================
100% scikit-learn — NO TensorFlow, NO PyTorch, NO DLL issues.

Endpoints:
  GET  /api/dates                  metadata + available dates
  POST /api/predict                RF volatility classification
  POST /api/mlp/predict            MLP next-day close price + sparkline
  POST /api/explain/lime           LIME local explanation   (RF)
  POST /api/explain/dice           DiCE counterfactuals     (RF)
  POST /api/explain/shap           SHAP KernelExplainer     (MLP)
  POST /api/explain/beta           BETA surrogate tree      (MLP) [CNS_Phase_3]
  POST /api/explain/sensitivity    Sensitivity + LRP Proxy  (MLP) [CNS_Phase_3]
  POST /api/explain/ablation       Ablation + Permutation   (MLP) [Untitled3]
  POST /api/explain/saliency       Saliency Map time×feat   (MLP) [Untitled3]
"""

import json, pickle, warnings, traceback
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

import lime, lime.lime_tabular
import dice_ml
import shap
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Load artefacts
# ═══════════════════════════════════════════════════════════════════════════

with open("model/metadata.json")    as f: meta         = json.load(f)
with open("model/rf_pipeline.pkl",  "rb") as f: rf_pipeline  = pickle.load(f)
with open("model/mlp_model.pkl",    "rb") as f: mlp_model    = pickle.load(f)
with open("model/ohlcv_scaler.pkl", "rb") as f: ohlcv_scaler = pickle.load(f)
with open("model/beta_tree.pkl",    "rb") as f: beta_tree    = pickle.load(f)
with open("model/mlp_dates.json")   as f: mlp_date_map = json.load(f)
with open("model/global_xai.json")  as f: global_xai   = json.load(f)

rf_df        = pd.read_csv("model/rf_data.csv",      index_col=0)
scaled_ohlcv = pd.read_csv("model/scaled_ohlcv.csv", index_col=0)
raw_ohlcv    = pd.read_csv("model/raw_ohlcv.csv",    index_col=0)

shap_background = np.load("model/shap_background.npy")
X_test_mlp      = np.load("model/X_test_mlp.npy")
y_test_mlp      = np.load("model/y_test_mlp.npy")

FEATURE_COLS    = meta["feature_cols"]
FEATURE_DISPLAY = meta["feature_display"]
OHLCV_COLS      = meta["ohlcv_cols"]
TIME_STEP       = meta["time_step"]
CLOSE_IDX       = OHLCV_COLS.index("Close")

# ── LIME (initialised once) ───────────────────────────────────────────────────
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data = rf_df[FEATURE_COLS].values,
    feature_names = [FEATURE_DISPLAY[f] for f in FEATURE_COLS],
    class_names   = ["Low Volatility", "High Volatility"],
    mode          = "classification",
    random_state  = 42,
)

# ── SHAP (initialised once) ───────────────────────────────────────────────────
def _mlp_predict(flat_data):
    return mlp_model.predict(flat_data).reshape(-1, 1)

shap_explainer = shap.KernelExplainer(_mlp_predict, shap_background)

# ═══════════════════════════════════════════════════════════════════════════
# XAI helpers
# ═══════════════════════════════════════════════════════════════════════════

def _numerical_gradient(flat_x, eps=1e-4):
    """Central-difference numerical gradient of MLP for one sample (1×300)."""
    grad = np.zeros(flat_x.shape[1])
    for j in range(flat_x.shape[1]):
        xp = flat_x.copy(); xp[0, j] += eps
        xm = flat_x.copy(); xm[0, j] -= eps
        grad[j] = (mlp_model.predict(xp)[0] - mlp_model.predict(xm)[0]) / (2 * eps)
    return grad  # (300,)

def _inverse_close(val):
    tmp = np.zeros((1, len(OHLCV_COLS)))
    tmp[0, CLOSE_IDX] = val
    return float(ohlcv_scaler.inverse_transform(tmp)[0, CLOSE_IDX])

# ═══════════════════════════════════════════════════════════════════════════
# Routing helpers
# ═══════════════════════════════════════════════════════════════════════════

def _get_rf_row(date):
    if date in rf_df.index:
        return rf_df.loc[date], date
    for d in range(1, 6):
        for s in (1, -1):
            c = (pd.Timestamp(date) + pd.Timedelta(days=d*s)).strftime("%Y-%m-%d")
            if c in rf_df.index:
                return rf_df.loc[c], c
    raise KeyError(f"No trading data near {date}.")

def _get_mlp_window(date):
    actual = date if date in mlp_date_map else None
    if not actual:
        for d in range(1, 6):
            for s in (1, -1):
                c = (pd.Timestamp(date) + pd.Timedelta(days=d*s)).strftime("%Y-%m-%d")
                if c in mlp_date_map:
                    actual = c; break
            if actual: break
    if not actual:
        raise KeyError(f"No MLP window near {date}.")
    idx       = mlp_date_map[actual]
    window    = scaled_ohlcv.values[idx : idx + TIME_STEP]   # (60,5)
    flat      = window.flatten().reshape(1, -1)               # (1,300)
    raw_close = float(raw_ohlcv.loc[actual, "Close"]) if actual in raw_ohlcv.index else None
    return flat, window, actual, raw_close

# ═══════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html", meta=meta)

@app.route("/api/dates")
def api_dates():
    return jsonify({"dates": rf_df.index.tolist(), "meta": meta})

# ── RF Prediction ─────────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    body = request.get_json(force=True)
    try: row, actual_date = _get_rf_row(body.get("date", ""))
    except KeyError as e: return jsonify({"error": str(e)}), 404

    X_q   = row[FEATURE_COLS].values.reshape(1, -1)
    pred  = int(rf_pipeline.predict(X_q)[0])
    proba = rf_pipeline.predict_proba(X_q)[0]
    imps  = rf_pipeline.named_steps["clf"].feature_importances_

    return jsonify({
        "date":             actual_date,
        "prediction":       pred,
        "prediction_label": "High Volatility" if pred == 1 else "Low Volatility",
        "confidence":       round(float(max(proba)) * 100, 2),
        "prob_low":         round(float(proba[0]) * 100, 2),
        "prob_high":        round(float(proba[1]) * 100, 2),
        "actual":           int(row["Target"]),
        "feature_importances": {
            FEATURE_DISPLAY[f]: round(float(imps[i]), 4)
            for i, f in enumerate(FEATURE_COLS)
        },
        "model_meta": {"rf_train_acc": meta["rf_train_acc"], "rf_test_acc": meta["rf_test_acc"]},
    })

# ── MLP Price Prediction ──────────────────────────────────────────────────────
@app.route("/api/mlp/predict", methods=["POST"])
def api_mlp_predict():
    body = request.get_json(force=True)
    try: flat, _, actual_date, raw_close = _get_mlp_window(body.get("date", ""))
    except KeyError as e: return jsonify({"error": str(e)}), 404

    pred_p = _inverse_close(float(mlp_model.predict(flat)[0]))

    all_dates  = list(raw_ohlcv.index)
    try: idx = all_dates.index(actual_date)
    except ValueError: idx = len(all_dates) - 1
    start       = max(0, idx - 29)
    hist_dates  = all_dates[start : idx + 1]
    hist_closes = [round(float(raw_ohlcv.loc[d, "Close"]), 2) for d in hist_dates]

    return jsonify({
        "date":            actual_date,
        "predicted_close": round(pred_p, 2),
        "actual_close":    round(raw_close, 2) if raw_close else None,
        "error_usd":       round(abs(pred_p - raw_close), 2) if raw_close else None,
        "hist_dates":      hist_dates,
        "hist_closes":     hist_closes,
        "model_meta":      {"mlp_rmse": meta["mlp_rmse"], "mlp_mae": meta["mlp_mae"]},
    })

# ── LIME ──────────────────────────────────────────────────────────────────────
@app.route("/api/explain/lime", methods=["POST"])
def api_lime():
    body = request.get_json(force=True)
    try: row, actual_date = _get_rf_row(body.get("date", ""))
    except KeyError as e: return jsonify({"error": str(e)}), 404

    exp = lime_explainer.explain_instance(
        data_row     = row[FEATURE_COLS].values,
        predict_fn   = lambda x: rf_pipeline.predict_proba(x),
        num_features = len(FEATURE_COLS),
        num_samples  = 2000,
    )
    return jsonify({
        "date": actual_date,
        "explanations": [
            {"feature": f, "weight": round(float(w), 6),
             "direction": "positive" if w > 0 else "negative"}
            for f, w in exp.as_list()
        ],
    })

# ── DiCE ──────────────────────────────────────────────────────────────────────
@app.route("/api/explain/dice", methods=["POST"])
def api_dice():
    body = request.get_json(force=True)
    try: row, actual_date = _get_rf_row(body.get("date", ""))
    except KeyError as e: return jsonify({"error": str(e)}), 404

    curr_pred = int(rf_pipeline.predict(row[FEATURE_COLS].values.reshape(1,-1))[0])
    try:
        train_data = rf_df[FEATURE_COLS + ["Target"]].copy()
        train_data["Target"] = train_data["Target"].astype(int)
        d   = dice_ml.Data(dataframe=train_data, continuous_features=FEATURE_COLS, outcome_name="Target")
        m   = dice_ml.Model(model=rf_pipeline, backend="sklearn", model_type="classifier")
        exp = dice_ml.Dice(d, m, method="random")
        res = exp.generate_counterfactuals(
            pd.DataFrame([{f: float(row[f]) for f in FEATURE_COLS}]),
            total_CFs=3, desired_class="opposite", random_seed=42
        )
        cf_examples = res.cf_examples_list[0]
        cf_df = None
        for attr in ("final_cfs_df","final_cfs_df_sparse","final_haml_df","haml_df","test_pred_df"):
            c = getattr(cf_examples, attr, None)
            if isinstance(c, pd.DataFrame) and len(c) > 0:
                cf_df = c.copy(); break
        if cf_df is None:
            for attr in vars(cf_examples):
                obj = getattr(cf_examples, attr, None)
                if isinstance(obj, pd.DataFrame) and len(obj) > 0 and any(f in obj.columns for f in FEATURE_COLS):
                    cf_df = obj.copy(); break
        if cf_df is None: return jsonify({"error": "DiCE could not generate counterfactuals."}), 500
        for dc in ["Target","target","outcome",d.outcome_name]:
            if dc in cf_df.columns: cf_df = cf_df.drop(columns=[dc])

        counterfactuals = []
        for _, cf_row in cf_df.iterrows():
            changes = {}
            for f in FEATURE_COLS:
                if f not in cf_row.index: continue
                try: cv, ov = float(cf_row[f]), float(row[f])
                except: continue
                if abs(cv - ov) > 1e-6:
                    changes[FEATURE_DISPLAY[f]] = {
                        "original": round(ov,4), "counterfactual": round(cv,4),
                        "change": round(cv-ov,4), "pct_change": round((cv-ov)/(abs(ov)+1e-10)*100,2),
                    }
            counterfactuals.append({"changes": changes})
        if not counterfactuals: return jsonify({"error":"DiCE produced no changes."}), 500
        return jsonify({
            "date": actual_date, "original_prediction": curr_pred,
            "original_label": "High Volatility" if curr_pred==1 else "Low Volatility",
            "target_label":   "Low Volatility"  if curr_pred==1 else "High Volatility",
            "counterfactuals": counterfactuals,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"DiCE error: {str(e)}"}), 500

# ── SHAP ──────────────────────────────────────────────────────────────────────
@app.route("/api/explain/shap", methods=["POST"])
def api_shap():
    body = request.get_json(force=True)
    try: flat, window, actual_date, _ = _get_mlp_window(body.get("date",""))
    except KeyError as e: return jsonify({"error": str(e)}), 404
    try:
        sv = shap_explainer.shap_values(flat.astype(np.float64), nsamples=100)
        if isinstance(sv, list): sv = sv[0]
        sv3d  = np.array(sv).reshape(1, TIME_STEP, len(OHLCV_COLS))
        f_imp = np.abs(sv3d[0]).mean(axis=0)
        t_imp = np.abs(sv3d[0]).mean(axis=1)
        return jsonify({
            "date": actual_date,
            "feature_importance": {OHLCV_COLS[i]: round(float(f_imp[i]),6) for i in range(5)},
            "temporal_importance": [
                {"step": f"t-{TIME_STEP-i}", "importance": round(float(t_imp[i]),6)}
                for i in range(TIME_STEP)
            ],
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"SHAP error: {str(e)}"}), 500

# ── BETA Surrogate Tree  (CNS_Phase_3) ────────────────────────────────────────
@app.route("/api/explain/beta", methods=["POST"])
def api_beta():
    body = request.get_json(force=True)
    try: flat, window, actual_date, _ = _get_mlp_window(body.get("date",""))
    except KeyError as e: return jsonify({"error": str(e)}), 404
    try:
        mlp_pred  = float(mlp_model.predict(flat)[0])
        beta_pred = float(beta_tree.predict(flat)[0])

        # Feature importances from the surrogate tree (aggregated by OHLCV feature)
        tree_imps = beta_tree.feature_importances_  # (300,)
        imps_3d   = tree_imps.reshape(TIME_STEP, len(OHLCV_COLS))
        feat_imps = imps_3d.mean(axis=0)  # (5,) — average across time

        # Decision path depth
        decision_path  = beta_tree.decision_path(flat)
        n_nodes_visited = int(decision_path.nnz)

        # Global surrogate importances from training
        global_imps_3d = beta_tree.feature_importances_.reshape(TIME_STEP, len(OHLCV_COLS))
        global_feat    = global_imps_3d.mean(axis=0)

        return jsonify({
            "date":          actual_date,
            "mlp_pred_scaled":   round(mlp_pred, 6),
            "beta_pred_scaled":  round(beta_pred, 6),
            "agreement_pct":     round(100 - abs(mlp_pred-beta_pred)/max(abs(mlp_pred),1e-10)*100, 2),
            "beta_r2":           meta["beta_r2"],
            "tree_depth":        int(beta_tree.get_depth()),
            "tree_leaves":       int(beta_tree.get_n_leaves()),
            "nodes_visited":     n_nodes_visited,
            "feature_importance": {OHLCV_COLS[i]: round(float(global_feat[i]),6) for i in range(5)},
            "local_importance":   {OHLCV_COLS[i]: round(float(feat_imps[i]),6)   for i in range(5)},
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"BETA error: {str(e)}"}), 500

# ── Sensitivity Analysis + LRP Proxy  (CNS_Phase_3) ──────────────────────────
@app.route("/api/explain/sensitivity", methods=["POST"])
def api_sensitivity():
    body = request.get_json(force=True)
    try: flat, window, actual_date, _ = _get_mlp_window(body.get("date",""))
    except KeyError as e: return jsonify({"error": str(e)}), 404
    try:
        grad  = _numerical_gradient(flat, eps=1e-4)         # (300,)
        g3d   = grad.reshape(TIME_STEP, len(OHLCV_COLS))    # (60,5)
        w3d   = flat.reshape(TIME_STEP, len(OHLCV_COLS))    # (60,5)

        # Sensitivity per feature = mean |grad| over time steps
        sensitivity = np.abs(g3d).mean(axis=0)   # (5,)
        # LRP proxy = mean |grad × input| over time steps
        lrp         = np.abs(g3d * w3d).mean(axis=0)  # (5,)

        # Temporal sensitivity = mean |grad| over features per time step
        temporal_sens = np.abs(g3d).mean(axis=1)  # (60,)

        return jsonify({
            "date": actual_date,
            "sensitivity": {OHLCV_COLS[i]: round(float(sensitivity[i]),8) for i in range(5)},
            "lrp_scores":  {OHLCV_COLS[i]: round(float(lrp[i]),8)         for i in range(5)},
            "temporal_sensitivity": [
                {"step": f"t-{TIME_STEP-i}", "value": round(float(temporal_sens[i]),8)}
                for i in range(TIME_STEP)
            ],
            # Also return global (pre-computed) for reference
            "global_sensitivity": {OHLCV_COLS[i]: round(float(global_xai["global_sensitivity"][i]),8) for i in range(5)},
            "global_lrp":         {OHLCV_COLS[i]: round(float(global_xai["global_lrp"][i]),8)         for i in range(5)},
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Sensitivity error: {str(e)}"}), 500

# ── Ablation Study + Permutation Importance  (Untitled3) ─────────────────────
@app.route("/api/explain/ablation", methods=["POST"])
def api_ablation():
    body = request.get_json(force=True)
    try: flat, window, actual_date, _ = _get_mlp_window(body.get("date",""))
    except KeyError as e: return jsonify({"error": str(e)}), 404
    try:
        baseline_pred = float(mlp_model.predict(flat)[0])

        # ── Ablation Study (zero out each feature across all 60 time steps)
        ablation = {}
        for i, feat in enumerate(OHLCV_COLS):
            x_ablated = flat.copy()
            # Zero out columns i, i+5, i+10 … (every 5th element = this feature)
            x_ablated[0, i::len(OHLCV_COLS)] = 0.0
            ablated_pred = float(mlp_model.predict(x_ablated)[0])
            ablation[feat] = round(abs(baseline_pred - ablated_pred), 8)

        # ── Permutation Importance (shuffle feature across all time steps)
        # Use saved test subset for meaningful MSE
        baseline_mse = mean_squared_error(y_test_mlp, mlp_model.predict(X_test_mlp))
        perm = {}
        for i, feat in enumerate(OHLCV_COLS):
            X_perm = X_test_mlp.copy()
            # Shuffle all time-step values for this feature
            col_indices = list(range(i, X_perm.shape[1], len(OHLCV_COLS)))
            rng = np.random.default_rng(42)
            rng.shuffle(X_perm[:, col_indices])
            perm_mse = mean_squared_error(y_test_mlp, mlp_model.predict(X_perm))
            perm[feat] = round(float(perm_mse - baseline_mse), 8)

        return jsonify({
            "date":                   actual_date,
            "baseline_pred_scaled":   round(baseline_pred, 6),
            "ablation_importance":    ablation,
            "permutation_importance": perm,
            "features":               OHLCV_COLS,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Ablation error: {str(e)}"}), 500

# ── Saliency Map  (Untitled3 — time × features heatmap) ──────────────────────
@app.route("/api/explain/saliency", methods=["POST"])
def api_saliency():
    body = request.get_json(force=True)
    try: flat, window, actual_date, _ = _get_mlp_window(body.get("date",""))
    except KeyError as e: return jsonify({"error": str(e)}), 404
    try:
        grad  = _numerical_gradient(flat, eps=1e-4)       # (300,)
        g3d   = grad.reshape(TIME_STEP, len(OHLCV_COLS))  # (60,5)
        # Return absolute gradient as 2-D saliency matrix
        saliency = np.abs(g3d).tolist()  # list[60][5]
        # Normalise to [0,1] for display
        s_arr = np.abs(g3d)
        s_max = s_arr.max() + 1e-10
        saliency_norm = (s_arr / s_max).tolist()

        return jsonify({
            "date":          actual_date,
            "saliency":      saliency_norm,
            "saliency_raw":  saliency,
            "features":      OHLCV_COLS,
            "time_steps":    TIME_STEP,
            "time_labels":   [f"t-{TIME_STEP-i}" for i in range(TIME_STEP)],
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Saliency error: {str(e)}"}), 500

# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n╔═════════════════════════════════════════════════════╗")
    print("║   VolatiLiX — Complete XAI Flask Server            ║")
    print("║   RF + MLP + BETA · LIME DiCE SHAP Grad LRP Sal   ║")
    print("║   http://localhost:5000                            ║")
    print("╚═════════════════════════════════════════════════════╝\n")
    app.run(debug=True, port=5000, host="0.0.0.0")