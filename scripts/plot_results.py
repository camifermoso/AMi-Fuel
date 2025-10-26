import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

REAL_FEATS  = ["avg_rpm","avg_throttle","avg_speed","avg_gear","avg_drs","avg_ers_mode"]
SYN_FEATS   = ["rpm","throttle","ers","speed","gear"]

DATA  = "data/test_highfuel.csv"      
MODEL = "outputs/fuel_model_real.pkl"
STRAT = "outputs/optimized_strategy.csv"

def pick_schema(df):
    if set(REAL_FEATS).issubset(df.columns):
        return "real"
    elif set(SYN_FEATS).issubset(df.columns):
        return "synth"
    else:
        raise ValueError("Input data doesn't match known schemas.")

# --- load ---
df = pd.read_csv(DATA)
model = joblib.load(MODEL)
opt = pd.read_csv(STRAT)
t_scale = float(opt.loc[0, "throttle_scale"])
e_shift = float(opt.loc[0, "ers_shift"])

schema = pick_schema(df)

if schema == "real":
    feats = REAL_FEATS
    fuel_base = model.predict(df[feats])

    df_opt = df.copy()
    df_opt["avg_throttle"] = (df_opt["avg_throttle"] * t_scale).clip(0, 100)
    df_opt["avg_ers_mode"] = (df_opt["avg_ers_mode"].fillna(0.0) + e_shift*4.0).clip(0, 4)
    fuel_opt = model.predict(df_opt[feats])

    # lap axis (if LapNumber exists)
    if "LapNumber" in df.columns:
        lap_axis = df["LapNumber"]
    else:
        lap_axis = range(1, len(df) + 1)

else:
    feats = SYN_FEATS
    fuel_base = model.predict(df[feats])

    df_opt = df.copy()
    df_opt["throttle"] = (df_opt["throttle"] * t_scale).clip(0.6, 1.0)
    df_opt["ers"] = (df_opt["ers"] + e_shift).clip(0.0, 1.0)
    fuel_opt = model.predict(df_opt[feats])

    lap_axis = df.get("lap", range(1, len(df) + 1))

# ---- Figure 1: fuel per lap (baseline vs optimized)
plt.figure()
plt.plot(lap_axis, fuel_base, label="Baseline")
plt.plot(lap_axis, fuel_opt, label="Optimized")
plt.xlabel("Lap")
plt.ylabel("Predicted fuel (proxy)")
plt.title("AMi Fuel — Baseline vs Optimized Fuel Usage")
plt.legend(); plt.tight_layout()
plt.savefig("outputs/fuel_comparison.png"); plt.close()

# ---- Figure 2: fuel vs lap-time trade-off (simple proxy)
# If we have lap_time, use it; else fake a small delta from t_scale/e_shift
if "lap_time" in df.columns:
    lap_time_base = df["lap_time"].values
else:
    lap_time_base = np.full(len(df), 95.0)

lap_time_opt = lap_time_base + (1.0 - t_scale)*0.5 - (e_shift)*0.2
plt.figure()
plt.scatter(lap_time_base, fuel_base, s=18, label="Baseline")
plt.scatter(lap_time_opt, fuel_opt, s=18, label="Optimized")
plt.xlabel("Lap time (s)"); plt.ylabel("Predicted fuel (proxy)")
plt.title("Fuel–Lap Time Trade-off")
plt.legend(); plt.tight_layout()
plt.savefig("outputs/fuel_time_tradeoff.png"); plt.close()

print("Saved: outputs/fuel_comparison.png, outputs/fuel_time_tradeoff.png")
print(f"Baseline total: {fuel_base.sum():.3f} | Optimized total: {fuel_opt.sum():.3f}")
