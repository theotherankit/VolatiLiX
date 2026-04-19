"""
train_model.py  —  VolatiLiX Complete Training Pipeline

Models trained:
  A) Random Forest Classifier → volatility regime (High/Low)
     XAI: LIME, DiCE

  B) MLP Regressor → next-day Close price
     XAI: SHAP, Ablation, Permutation Importance, Saliency Map

  C) BETA Surrogate Decision Tree → mimics MLP 
     XAI: BETA tree, Sensitivity Analysis, LRP Proxy
"""

import os, json, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
)

TIME_STEP  = 60
OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]
CLOSE_IDX  = OHLCV_COLS.index("Close")

MLP_FEATURE_NAMES = OHLCV_COLS  # 5 features, each over 60 time steps

RF_FEATURE_COLS = [
    "Vol_5", "Vol_10", "Vol_20",
    "RSI", "MACD", "MACD_Hist",
    "BB_Width", "BB_Position",
    "Volume_Ratio",
    "Momentum_5", "Momentum_10", "Momentum_20",
]
FEATURE_DISPLAY = {
    "Vol_5":        "5-Day Volatility",
    "Vol_10":       "10-Day Volatility",
    "Vol_20":       "20-Day Volatility",
    "RSI":          "RSI (14)",
    "MACD":         "MACD Line",
    "MACD_Hist":    "MACD Histogram",
    "BB_Width":     "Bollinger Width",
    "BB_Position":  "BB Position",
    "Volume_Ratio": "Volume Ratio",
    "Momentum_5":   "Momentum 5D",
    "Momentum_10":  "Momentum 10D",
    "Momentum_20":  "Momentum 20D",
}

def calc_rsi(prices, window=14):
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta).clip(lower=0).rolling(window).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-10))


def engineer_rf_features(raw):
    d = raw.copy()
    d["Log_Returns"] = np.log(d["Close"] / d["Close"].shift(1))
    for w in (5, 10, 20):
        d[f"Vol_{w}"]      = d["Log_Returns"].rolling(w).std() * np.sqrt(252)
        d[f"Momentum_{w}"] = d["Close"].pct_change(w)
    d["RSI"] = calc_rsi(d["Close"])
    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"]        = ema12 - ema26
    d["MACD_Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_Hist"]   = d["MACD"] - d["MACD_Signal"]
    bb_mid           = d["Close"].rolling(20).mean()
    bb_std           = d["Close"].rolling(20).std()
    d["BB_Width"]    = 2 * bb_std / (bb_mid + 1e-10)
    d["BB_Position"] = (d["Close"] - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-10)
    d["Volume_Ratio"]= d["Volume"] / (d["Volume"].rolling(20).mean() + 1e-10)
    future_vol       = d["Log_Returns"].shift(-20).rolling(20).std() * np.sqrt(252)
    median_vol       = float(future_vol.dropna().median())
    d["Target"]      = (future_vol > median_vol).astype(int)
    return d, median_vol


def create_mlp_dataset(scaled, time_step=60):
    X, y = [], []
    for i in range(len(scaled) - time_step):
        X.append(scaled[i : i + time_step].flatten())
        y.append(scaled[i + time_step, CLOSE_IDX])
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)


def mlp_numerical_gradient(model, flat_x, eps=1e-4):
    grad = np.zeros(flat_x.shape[1])
    for j in range(flat_x.shape[1]):
        x_p = flat_x.copy(); x_p[0, j] += eps
        x_m = flat_x.copy(); x_m[0, j] -= eps
        grad[j] = (model.predict(x_p)[0] - model.predict(x_m)[0]) / (2 * eps)
    return grad  



def train_and_save():
    print("\n╔══════════════════════════════════════════════════╗")
    print("║   VolatiLiX — Full Training Pipeline             ║")
    print("║   RF + MLP + BETA · LIME DiCE SHAP Grad LRP     ║")
    print("╚══════════════════════════════════════════════════╝\n")
    os.makedirs("model", exist_ok=True)

    print("📥  Downloading AAPL 2010-01-01 → 2022-12-31 …")
    raw = yf.download("AAPL", start="2010-01-01", end="2022-12-31", progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw[OHLCV_COLS].dropna()
    print(f"    {len(raw):,} trading days\n")

    print("── MODEL A: Random Forest Volatility Classifier ──────────────")
    rf_df, vol_median = engineer_rf_features(raw)
    rf_df.dropna(inplace=True)

    X_rf, y_rf = rf_df[RF_FEATURE_COLS], rf_df["Target"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_rf, y_rf, test_size=0.2, shuffle=False, random_state=42
    )

    rf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=10,
            min_samples_split=5, min_samples_leaf=2,
            random_state=42, n_jobs=-1,
        )),
    ])
    rf_pipeline.fit(X_tr, y_tr)

    train_acc = accuracy_score(y_tr, rf_pipeline.predict(X_tr))
    test_acc  = accuracy_score(y_te, rf_pipeline.predict(X_te))
    print(f"    Train acc: {train_acc*100:.1f}%   Test acc: {test_acc*100:.1f}%")
    print(classification_report(y_te, rf_pipeline.predict(X_te),
                                 target_names=["Low Vol", "High Vol"]))

    with open("model/rf_pipeline.pkl", "wb") as f:
        pickle.dump(rf_pipeline, f)

    rf_save = rf_df[RF_FEATURE_COLS + ["Target"]].copy()
    rf_save.index = rf_save.index.strftime("%Y-%m-%d")
    rf_save.to_csv("model/rf_data.csv")
    print("    ✓ RF pipeline saved\n")

    print("── MODEL B: MLP Neural Network Price Predictor ───────────────")
    ohlcv  = raw[OHLCV_COLS].values.astype(np.float64)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(ohlcv)

    print("    Building 60-day windows …")
    X_mlp, y_mlp = create_mlp_dataset(scaled, TIME_STEP)

    split = int(len(X_mlp) * 0.8)
    X_tr_m, X_te_m = X_mlp[:split], X_mlp[split:]
    y_tr_m, y_te_m = y_mlp[:split], y_mlp[split:]

    print(f"    Train: {len(X_tr_m):,}  |  Test: {len(X_te_m):,}  |  Features: {X_mlp.shape[1]}")

    mlp = MLPRegressor(
        hidden_layer_sizes  = (256, 128, 64),
        activation          = "relu",
        solver              = "adam",
        learning_rate_init  = 0.001,
        max_iter            = 300,
        early_stopping      = True,
        validation_fraction = 0.1,
        n_iter_no_change    = 15,
        random_state        = 42,
        verbose             = True,
    )
    print("    Training MLP (~2-5 min) …")
    mlp.fit(X_tr_m, y_tr_m)
    print(f"    Converged in {mlp.n_iter_} iterations")

    preds_s  = mlp.predict(X_te_m).reshape(-1, 1)
    tmp_p    = np.zeros((len(preds_s), len(OHLCV_COLS)))
    tmp_p[:, CLOSE_IDX] = preds_s[:, 0]
    preds_p  = scaler.inverse_transform(tmp_p)[:, CLOSE_IDX]

    tmp_y    = np.zeros((len(y_te_m), len(OHLCV_COLS)))
    tmp_y[:, CLOSE_IDX] = y_te_m
    actual_p = scaler.inverse_transform(tmp_y)[:, CLOSE_IDX]

    rmse = float(np.sqrt(mean_squared_error(actual_p, preds_p)))
    mae  = float(mean_absolute_error(actual_p, preds_p))
    print(f"\n    RMSE: ${rmse:.2f}   MAE: ${mae:.2f}")

    with open("model/mlp_model.pkl", "wb") as f:
        pickle.dump(mlp, f)
    with open("model/ohlcv_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    n_save = min(200, len(X_te_m))
    np.save("model/X_test_mlp.npy",  X_te_m[:n_save])
    np.save("model/y_test_mlp.npy",  y_te_m[:n_save])

    np.save("model/shap_background.npy", X_tr_m[:100])

    mlp_dates    = raw.index[TIME_STEP:].strftime("%Y-%m-%d").tolist()
    mlp_date_map = {d: i for i, d in enumerate(mlp_dates)}
    with open("model/mlp_dates.json", "w") as f:
        json.dump(mlp_date_map, f)

    pd.DataFrame(scaled, index=raw.index.strftime("%Y-%m-%d"),
                 columns=OHLCV_COLS).to_csv("model/scaled_ohlcv.csv")
    pd.DataFrame(ohlcv, index=raw.index.strftime("%Y-%m-%d"),
                 columns=OHLCV_COLS).to_csv("model/raw_ohlcv.csv")
    print("    ✓ MLP model + artefacts saved\n")

    print("── MODEL C: BETA Surrogate Decision Tree (CNS_Phase_3) ───────")
    print("    Using MLP predictions on test set as surrogate targets …")

    mlp_test_preds = mlp.predict(X_te_m[:n_save])

    beta_tree = DecisionTreeRegressor(max_depth=3, random_state=42)
    beta_tree.fit(X_te_m[:n_save], mlp_test_preds)

    beta_r2    = float(r2_score(mlp_test_preds, beta_tree.predict(X_te_m[:n_save])))
    beta_rmse  = float(np.sqrt(mean_squared_error(
        mlp_test_preds, beta_tree.predict(X_te_m[:n_save])
    )))
    print(f"    BETA surrogate R²: {beta_r2:.4f}   RMSE vs MLP: {beta_rmse:.6f}")
    print(f"    Tree leaves: {beta_tree.get_n_leaves()}   Max depth: {beta_tree.get_depth()}")

    with open("model/beta_tree.pkl", "wb") as f:
        pickle.dump(beta_tree, f)

    print("    Pre-computing global sensitivity (numerical gradients on 50 samples) …")
    n_grad    = min(50, len(X_te_m))
    all_grads = []
    for idx in range(n_grad):
        g = mlp_numerical_gradient(mlp, X_te_m[idx:idx+1], eps=1e-4)
        all_grads.append(g)
        if (idx + 1) % 10 == 0:
            print(f"      {idx+1}/{n_grad} done …")
    all_grads = np.array(all_grads)   
    
    grads_3d  = all_grads.reshape(n_grad, TIME_STEP, len(OHLCV_COLS))

    global_sensitivity = np.abs(grads_3d).mean(axis=(0, 1)).tolist()   
    
    X_te_3d   = X_te_m[:n_grad].reshape(n_grad, TIME_STEP, len(OHLCV_COLS))
    grad_x_in = np.abs(grads_3d * X_te_3d).mean(axis=(0, 1)).tolist() 

    np.save("model/global_grads.npy", all_grads)
    global_xai = {
        "global_sensitivity": global_sensitivity,
        "global_lrp":         grad_x_in,
        "features":           OHLCV_COLS,
    }
    with open("model/global_xai.json", "w") as f:
        json.dump(global_xai, f, indent=2)

    print("    ✓ BETA tree + global XAI saved\n")

    meta = {
        "feature_cols":    RF_FEATURE_COLS,
        "feature_display": FEATURE_DISPLAY,
        "ohlcv_cols":      OHLCV_COLS,
        "time_step":       TIME_STEP,
        "vol_median":      vol_median,
        "rf_train_acc":    round(train_acc, 4),
        "rf_test_acc":     round(test_acc,  4),
        "mlp_rmse":        round(rmse, 4),
        "mlp_mae":         round(mae,  4),
        "beta_r2":         round(beta_r2, 4),
        "n_rf_samples":    len(X_rf),
        "n_mlp_train":     split,
        "date_start":      rf_df.index[0].strftime("%Y-%m-%d"),
        "date_end":        rf_df.index[-1].strftime("%Y-%m-%d"),
    }
    with open("model/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("   All artefacts saved to ./model/")
    print(f"    RF   — test acc {test_acc*100:.1f}%")
    print(f"    MLP  — RMSE ${rmse:.2f}")
    print(f"    BETA — R² {beta_r2:.4f}")
    print("\n    Now run:  python App.py\n")


if __name__ == "__main__":
    train_and_save()