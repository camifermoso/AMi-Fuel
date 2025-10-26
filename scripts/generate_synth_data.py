
import numpy as np, pandas as pd, argparse

def generate(laps=50, seed=42):
    rng = np.random.default_rng(seed)
    rpm = rng.normal(11000, 450, laps).clip(8000, 12500)
    throttle = rng.uniform(0.75, 1.0, laps)
    ers = rng.uniform(0.2, 0.9, laps)
    speed = rng.normal(230, 10, laps).clip(180, 320)
    gear = rng.integers(5, 8, laps)
    fuel_burn_proxy = 0.48*(rpm/12000) + 0.32*throttle + 0.20*ers + rng.normal(0, 0.01, laps)
    lap_time = 95 - 2.8*throttle - 1.8*ers + rng.normal(0, 0.2, laps)
    return pd.DataFrame(dict(lap=range(1, laps+1), rpm=rpm, throttle=throttle, ers=ers, speed=speed, gear=gear, fuel_burn_proxy=fuel_burn_proxy, lap_time=lap_time))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--laps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/synth_telemetry.csv")
    args = ap.parse_args()
    df = generate(args.laps, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Saved {args.out}")
