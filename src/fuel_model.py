
import argparse, pandas as pd, joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--save-model", default="outputs/fuel_model.pkl")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    X = df[["rpm","throttle","ers","speed","gear"]]
    y = df["fuel_burn_proxy"]
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.25,random_state=42)
    model = RandomForestRegressor(n_estimators=300,random_state=42).fit(Xtr,ytr)
    ypred = model.predict(Xte)
    print("R2=%.3f MAE=%.4f"%(r2_score(yte,ypred), mean_absolute_error(yte,ypred)))
    joblib.dump(model,args.save_model)
    print("Model saved to",args.save_model)
