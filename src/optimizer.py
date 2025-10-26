
import argparse, pandas as pd, joblib, numpy as np

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="outputs/optimized_strategy.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    model = joblib.load(args.model)
    features = ["rpm","throttle","ers","speed","gear"]
    baseline = model.predict(df[features]).sum()
    best=None
    for ts in np.linspace(0.92,0.995,10):
        for es in np.linspace(-0.05,0.1,10):
            sim=df.copy()
            sim["throttle"]=(sim["throttle"]*ts).clip(0.6,1.0)
            sim["ers"]=(sim["ers"]+es).clip(0,1)
            fuel=model.predict(sim[features]).sum()
            if best is None or fuel<best["fuel"]:
                best=dict(throttle_scale=ts,ers_shift=es,fuel=fuel)
    pd.DataFrame([best]).to_csv(args.out,index=False)
    print("Best strategy",best,"baseline fuel",baseline)
