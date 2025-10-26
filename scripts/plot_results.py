import pandas as pd
import joblib
import matplotlib.pyplot as plt

DATA = "data/synth_telemetry.csv"
MODEL = "outputs/fuel_model.pkl"
STRAT = "outputs/optimized_strategy.csv"

df = pd.read_csv(DATA)
model = joblib.load(MODEL)
opt = pd.read_csv(STRAT)

t_scale = float(opt.loc[0, "throttle_scale"])
e_shift = float(opt.loc[0, "ers_shift"])

features = ["rpm","throttle","ers","speed","gear"]

# baseline predictions
fuel_base = model.predict(df[features])

# simulate optimized inputs and predict fuel
df_opt = df.copy()
df_opt["throttle"] = (df_opt["throttle"] * t_scale).clip(0.6, 1.0)
df_opt["ers"] = (df_opt["ers"] + e_shift).clip(0.0, 1.0)
fuel_opt = model.predict(df_opt[features])

# ---- Figure 1: fuel per lap (baseline vs optimized)
plt.figure()
plt.plot(df["lap"], fuel_base, label="Baseline")
plt.plot(df_opt["lap"], fuel_opt, label="Optimized")
plt.xlabel("Lap"); plt.ylabel("Predicted fuel (proxy)")
plt.title("AMi Fuel — Baseline vs Optimized Fuel Usage")
plt.legend(); plt.tight_layout()
plt.savefig("outputs/fuel_comparison.png"); plt.close()

# ---- Figure 2: fuel vs lap-time trade-off
lap_time_base = df["lap_time"]
lap_time_opt = df["lap_time"] + (1.0 - t_scale)*0.5 - (e_shift)*0.2
plt.figure()
plt.scatter(lap_time_base, fuel_base, s=18, label="Baseline")
plt.scatter(lap_time_opt, fuel_opt, s=18, label="Optimized")
plt.xlabel("Lap time (s)"); plt.ylabel("Predicted fuel (proxy)")
plt.title("Fuel–Lap Time Trade-off")
plt.legend(); plt.tight_layout()
plt.savefig("outputs/fuel_time_tradeoff.png"); plt.close()

print("Saved: outputs/fuel_comparison.png")
print("Saved: outputs/fuel_time_tradeoff.png")
print(f"Baseline total fuel proxy: {fuel_base.sum():.3f}")
print(f"Optimized total fuel proxy: {fuel_opt.sum():.3f}")
