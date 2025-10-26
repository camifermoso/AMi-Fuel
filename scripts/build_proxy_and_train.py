import pandas as pd, numpy as np, joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from pathlib import Path

def fuel_proxy(df: pd.DataFrame) -> pd.Series:
    # scale/clip inputs safely
    rpm = df["avg_rpm"].astype(float)
    thr = df["avg_throttle"].astype(float)
    ers = df["avg_ers_mode"].astype(float) if "avg_ers_mode" in df.columns else pd.Series(0.0, index=df.index)

    rpm = rpm.fillna(rpm.mean())
    thr = thr.fillna(thr.mean())
    ers = ers.fillna(0.0)

    rpm_scaled = np.clip(rpm / 12000.0, 0, 1.2)  # allow slight >1 due to outliers
    thr_scaled = np.clip(thr / 100.0, 0, 1.0)
    ers_scaled = np.clip(ers / 4.0, 0, 1.0)      # rough normalization for deploy modes

    return 0.48 * rpm_scaled + 0.32 * thr_scaled + 0.20 * ers_scaled

def load_and_prepare(path):
    df = pd.read_csv(path)
    df["fuel_proxy"] = fuel_proxy(df)
    # Drop lines missing the essentials
    df = df.dropna(subset=["avg_throttle","avg_rpm","avg_speed"])
    # ensure numeric
    for col in ["avg_rpm","avg_throttle","avg_speed","avg_gear","avg_drs","avg_ers_mode"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df

if __name__ == "__main__":
    train = load_and_prepare("data/train_highfuel.csv")
    test  = load_and_prepare("data/test_highfuel.csv")

    features = ["avg_rpm","avg_throttle","avg_speed","avg_gear","avg_drs","avg_ers_mode"]
    Xtr, ytr = train[features], train["fuel_proxy"]
    Xte, yte = test[features],  test["fuel_proxy"]

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(Xtr, ytr)
    yhat = model.predict(Xte)

    r2  = r2_score(yte, yhat)
    mae = mean_absolute_error(yte, yhat)
    print(f"Test R²: {r2:.3f} | MAE: {mae:.4f} on held-out year/circuit set")

    Path("outputs").mkdir(exist_ok=True, parents=True)
    joblib.dump(model, "outputs/fuel_model_real.pkl")
    pd.DataFrame({"y_true": yte, "y_pred": yhat}).to_csv("outputs/test_preds.csv", index=False)
    print("Saved model -> outputs/fuel_model_real.pkl, predictions -> outputs/test_preds.csv")
