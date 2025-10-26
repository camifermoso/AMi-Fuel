import fastf1
import pandas as pd
from pathlib import Path

# Enable on-disk cache
fastf1.Cache.enable_cache("cache")

# Circuits we consider fuel-heavy for this study
GP_MAP = {
    "singapore": "Singapore Grand Prix",
    "barcelona": "Spanish Grand Prix",
    "bahrain": "Bahrain Grand Prix",
    "montreal": "Canadian Grand Prix",
    "suzuka": "Japanese Grand Prix",
}

def fetch_session(year: int, gp_key: str, session_code: str = "R") -> pd.DataFrame:
    """Download a race session and return lap-aggregated features suitable for our ML pipeline."""
    gp_name = GP_MAP[gp_key]
    session = fastf1.get_session(year, gp_name, session_code)
    session.load()

    # Use quick laps to avoid in/out/Safety Car laps
    laps = session.laps.pick_quicklaps()
    if laps.empty:
        return pd.DataFrame()

    # Base lap-level info
    df = laps[[
        "LapNumber", "Driver", "Team", "LapTime", "Stint", "Compound",
        "SpeedI1", "SpeedI2", "SpeedFL", "TrackStatus"
    ]].copy()

    # Aggregate per-lap telemetry
    rows = []
    for _, lap in laps.iterlaps():
        tel = lap.get_telemetry()
        if tel is None or tel.empty:
            continue
        row = {
            "LapNumber": int(lap["LapNumber"]),
            "avg_throttle": float(tel["Throttle"].mean(skipna=True)) if "Throttle" in tel else None,
            "avg_rpm": float(tel["RPM"].mean(skipna=True)) if "RPM" in tel else None,
            "avg_speed": float(tel["Speed"].mean(skipna=True)) if "Speed" in tel else None,
            "avg_gear": float(tel["nGear"].mean(skipna=True)) if "nGear" in tel else None,
            "avg_drs": float(tel["DRS"].mean(skipna=True)) if "DRS" in tel else None,
            # ERSDeployMode often missing in public telemetry; handle gracefully
            "avg_ers_mode": float(tel["ERSDeployMode"].mean(skipna=True)) if "ERSDeployMode" in tel else None,
        }
        rows.append(row)

    agg = pd.DataFrame(rows)
    df = df.merge(agg, on="LapNumber", how="inner")
    return df

def build_split(train_years, test_years, gps):
    train_frames, test_frames = [], []
    for y in train_years:
        for gp in gps:
            print(f"[TRAIN] {y} {gp}")
            part = fetch_session(y, gp)
            if not part.empty:
                part["year"] = y
                part["gp"] = gp
                train_frames.append(part)
    for y in test_years:
        for gp in gps:
            print(f"[TEST] {y} {gp}")
            part = fetch_session(y, gp)
            if not part.empty:
                part["year"] = y
                part["gp"] = gp
                test_frames.append(part)
    train = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame()
    test  = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame()
    return train, test

if __name__ == "__main__":
    GPS = ["singapore", "barcelona", "bahrain", "montreal", "suzuka"]
    train, test = build_split(train_years=[2022], test_years=[2023], gps=GPS)

    Path("data").mkdir(exist_ok=True, parents=True)
    train.to_csv("data/train_highfuel.csv", index=False)
    test.to_csv("data/test_highfuel.csv", index=False)
    print("Saved: data/train_highfuel.csv, data/test_highfuel.csv")
