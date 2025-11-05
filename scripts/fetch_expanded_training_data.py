"""
Fetch expanded training data for AMi-Fuel model:
1. High-fuel circuits: 10 years of data (2015-2024)
2. Other circuits: Random years to add variety
"""

import fastf1
import pandas as pd
from pathlib import Path
import random
import signal
from contextlib import contextmanager

# Enable on-disk cache
fastf1.Cache.enable_cache("cache")


class TimeoutException(Exception):
    """Custom exception for timeout."""
    pass


@contextmanager
def time_limit(seconds):
    """Context manager for timing out operations."""
    def signal_handler(signum, frame):
        raise TimeoutException(f"Timed out after {seconds} seconds")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# Primary high-fuel circuits - train on many years
HIGH_FUEL_CIRCUITS = {
    "singapore": "Singapore Grand Prix",
    "barcelona": "Spanish Grand Prix", 
    "bahrain": "Bahrain Grand Prix",
    "montreal": "Canadian Grand Prix",
    "suzuka": "Japanese Grand Prix",
}

# Additional circuits to include for variety (not as fuel-intensive)
ADDITIONAL_CIRCUITS = {
    "austria": "Austrian Grand Prix",
    "silverstone": "British Grand Prix",
    "monza": "Italian Grand Prix",
    "spa": "Belgian Grand Prix",
    "abu-dhabi": "Abu Dhabi Grand Prix",
    "mexico": "Mexico City Grand Prix",
    "brazil": "São Paulo Grand Prix",
    "hungary": "Hungarian Grand Prix",
    "monaco": "Monaco Grand Prix",
    "austin": "United States Grand Prix",
}


def fetch_session(year: int, gp_name: str, session_code: str = "R", timeout: int = 180) -> pd.DataFrame:
    """Download a race session and return lap-aggregated features."""
    try:
        # Use timeout to prevent hanging
        with time_limit(timeout):
            session = fastf1.get_session(year, gp_name, session_code)
            session.load()
    except TimeoutException as e:
        print(f"   ⏱️  Timeout loading {year} {gp_name} (>{timeout}s)")
        return pd.DataFrame()
    except Exception as e:
        print(f"   ⚠️  Failed to load {year} {gp_name}: {e}")
        return pd.DataFrame()

    # Use quick laps to avoid in/out/Safety Car laps
    laps = session.laps.pick_quicklaps()
    if laps.empty:
        print(f"   ⚠️  No quick laps for {year} {gp_name}")
        return pd.DataFrame()

    # Base lap-level info
    df = laps[[
        "LapNumber", "Driver", "Team", "LapTime", "Stint", "Compound",
        "SpeedI1", "SpeedI2", "SpeedFL", "TrackStatus"
    ]].copy()

    # Get weather data for the session
    weather_data = session.weather_data
    
    # Aggregate per-lap telemetry + weather
    rows = []
    for _, lap in laps.iterlaps():
        try:
            tel = lap.get_telemetry()
            if tel is None or tel.empty:
                continue
            
            # Get weather at lap time
            lap_time = lap["LapStartTime"]
            if weather_data is not None and not weather_data.empty and lap_time is not None:
                # Find closest weather reading to lap start time
                weather_at_lap = weather_data.iloc[(weather_data['Time'] - lap_time).abs().argmin()]
                air_temp = float(weather_at_lap['AirTemp']) if 'AirTemp' in weather_at_lap else None
                track_temp = float(weather_at_lap['TrackTemp']) if 'TrackTemp' in weather_at_lap else None
                humidity = float(weather_at_lap['Humidity']) if 'Humidity' in weather_at_lap else None
                pressure = float(weather_at_lap['Pressure']) if 'Pressure' in weather_at_lap else None
                rainfall = bool(weather_at_lap['Rainfall']) if 'Rainfall' in weather_at_lap else False
                wind_speed = float(weather_at_lap['WindSpeed']) if 'WindSpeed' in weather_at_lap else None
                wind_direction = float(weather_at_lap['WindDirection']) if 'WindDirection' in weather_at_lap else None
            else:
                air_temp = track_temp = humidity = pressure = wind_speed = wind_direction = None
                rainfall = False
            
            row = {
                "LapNumber": int(lap["LapNumber"]),
                "avg_throttle": float(tel["Throttle"].mean(skipna=True)) if "Throttle" in tel else None,
                "avg_rpm": float(tel["RPM"].mean(skipna=True)) if "RPM" in tel else None,
                "avg_speed": float(tel["Speed"].mean(skipna=True)) if "Speed" in tel else None,
                "avg_gear": float(tel["nGear"].mean(skipna=True)) if "nGear" in tel else None,
                "avg_drs": float(tel["DRS"].mean(skipna=True)) if "DRS" in tel else None,
                "avg_ers_mode": float(tel["ERSDeployMode"].mean(skipna=True)) if "ERSDeployMode" in tel else None,
                # Weather features
                "air_temp": air_temp,
                "track_temp": track_temp,
                "humidity": humidity,
                "pressure": pressure,
                "rainfall": rainfall,
                "wind_speed": wind_speed,
                "wind_direction": wind_direction,
            }
            rows.append(row)
        except Exception as e:
            continue

    if not rows:
        print(f"   ⚠️  No telemetry data for {year} {gp_name}")
        return pd.DataFrame()

    agg = pd.DataFrame(rows)
    df = df.merge(agg, on="LapNumber", how="inner")
    
    print(f"   ✓ Fetched {len(df)} laps from {year} {gp_name}")
    return df


def fetch_multi_year_data():
    """
    Fetch expanded training data:
    - High-fuel circuits: 2018-2024 (7 years)
    - Additional circuits: 2-3 random years each
    """
    
    print("="*80)
    print("FETCHING EXPANDED TRAINING DATA")
    print("="*80)
    print()
    
    all_frames = []
    
    # 1. High-fuel circuits: fetch many years
    print("📊 PHASE 1: HIGH-FUEL CIRCUITS (2018-2024)")
    print("-"*80)
    high_fuel_years = range(2018, 2025)  # 2018-2024
    
    total_sessions = len(HIGH_FUEL_CIRCUITS) * len(high_fuel_years)
    completed = 0
    
    for gp_key, gp_name in HIGH_FUEL_CIRCUITS.items():
        print(f"\n🏁 {gp_name}")
        for year in high_fuel_years:
            completed += 1
            print(f"   [{completed}/{total_sessions}] Fetching {year}...", end=" ", flush=True)
            df = fetch_session(year, gp_name)
            if not df.empty:
                df["year"] = year
                df["gp"] = gp_key
                df["circuit_type"] = "high_fuel"
                all_frames.append(df)
                print(f"✓ ({len(df)} laps)")
            else:
                print("✗ (no data)")
    
    print()
    print(f"✓ Phase 1 complete: {len(all_frames)} sessions fetched")
    print()
    
    print()
    print()
    print("📊 PHASE 2: ADDITIONAL CIRCUITS (Random Years for Variety)")
    print("-"*80)
    
    # 2. Additional circuits: sample 3 random years each
    all_years = list(range(2018, 2025))
    phase1_count = len(all_frames)
    total_additional = len(ADDITIONAL_CIRCUITS) * 3
    completed_additional = 0
    
    for gp_key, gp_name in ADDITIONAL_CIRCUITS.items():
        print(f"\n🏁 {gp_name}")
        # Pick 3 random years for variety
        sampled_years = random.sample(all_years, min(3, len(all_years)))
        sampled_years.sort()
        
        for year in sampled_years:
            completed_additional += 1
            print(f"   [{completed_additional}/{total_additional}] Fetching {year}...", end=" ", flush=True)
            df = fetch_session(year, gp_name)
            if not df.empty:
                df["year"] = year
                df["gp"] = gp_key
                df["circuit_type"] = "additional"
                all_frames.append(df)
                print(f"✓ ({len(df)} laps)")
            else:
                print("✗ (no data)")
    
    print()
    print(f"✓ Phase 2 complete: {len(all_frames) - phase1_count} sessions fetched")
    print()
    
    print()
    print()
    print("="*80)
    print("COMBINING DATA")
    print("="*80)
    
    if not all_frames:
        print("❌ No data fetched!")
        return pd.DataFrame()
    
    combined = pd.concat(all_frames, ignore_index=True)
    
    # Summary statistics
    print(f"\n✓ Total laps collected: {len(combined):,}")
    print(f"✓ Years covered: {sorted(combined['year'].unique())}")
    print(f"✓ Circuits: {len(combined['gp'].unique())}")
    print()
    
    print("📊 Breakdown by circuit type:")
    print(combined.groupby('circuit_type').size())
    print()
    
    print("📊 Breakdown by year:")
    print(combined.groupby('year').size().sort_index())
    print()
    
    print("📊 Breakdown by circuit:")
    circuit_counts = combined.groupby('gp').size().sort_values(ascending=False)
    for circuit, count in circuit_counts.items():
        print(f"   {circuit:15s}: {count:,} laps")
    
    return combined


def create_train_test_split(df: pd.DataFrame, test_size: float = 0.15):
    """
    Create train/test split:
    - Stratified by circuit to ensure all circuits represented
    - Random split within each circuit
    """
    from sklearn.model_selection import train_test_split
    
    print()
    print("="*80)
    print("CREATING TRAIN/TEST SPLIT")
    print("="*80)
    print()
    
    # Stratify by circuit to ensure representation
    train_frames = []
    test_frames = []
    
    for circuit in df['gp'].unique():
        circuit_data = df[df['gp'] == circuit]
        
        if len(circuit_data) < 10:  # Too few samples
            print(f"⚠️  Skipping {circuit} - only {len(circuit_data)} samples")
            continue
        
        train, test = train_test_split(
            circuit_data, 
            test_size=test_size, 
            random_state=42,
            shuffle=True
        )
        
        train_frames.append(train)
        test_frames.append(test)
        
        print(f"   {circuit:15s}: {len(train):,} train, {len(test):,} test")
    
    train_df = pd.concat(train_frames, ignore_index=True)
    test_df = pd.concat(test_frames, ignore_index=True)
    
    print()
    print(f"✓ Training set: {len(train_df):,} laps ({len(train_df)/len(df)*100:.1f}%)")
    print(f"✓ Test set: {len(test_df):,} laps ({len(test_df)/len(df)*100:.1f}%)")
    print()
    
    return train_df, test_df


def main():
    """Main execution."""
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Fetch all data
    combined_df = fetch_multi_year_data()
    
    if combined_df.empty:
        print("❌ No data to process!")
        return
    
    # Create train/test split
    train_df, test_df = create_train_test_split(combined_df, test_size=0.15)
    
    # Save to CSV
    print("="*80)
    print("SAVING DATA")
    print("="*80)
    print()
    
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    train_path = output_dir / "train_highfuel_expanded.csv"
    test_path = output_dir / "test_highfuel_expanded.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"✓ Training data saved: {train_path}")
    print(f"   → {len(train_df):,} laps")
    print()
    print(f"✓ Test data saved: {test_path}")
    print(f"   → {len(test_df):,} laps")
    print()
    
    # Show some statistics
    print("="*80)
    print("FINAL STATISTICS")
    print("="*80)
    print()
    
    print("🏆 Training Data Coverage:")
    print(f"   Years: {sorted(train_df['year'].unique())}")
    print(f"   Circuits: {len(train_df['gp'].unique())} unique")
    print(f"   High-fuel circuits: {len(train_df[train_df['circuit_type']=='high_fuel']['gp'].unique())}")
    print(f"   Additional circuits: {len(train_df[train_df['circuit_type']=='additional']['gp'].unique())}")
    print()
    
    print("📈 Year distribution in training data:")
    year_dist = train_df.groupby('year').size().sort_index()
    for year, count in year_dist.items():
        pct = count / len(train_df) * 100
        print(f"   {year}: {count:,} laps ({pct:.1f}%)")
    print()
    
    print("✅ Data fetching complete!")
    print()
    print("Next steps:")
    print("   1. Run: python scripts/train_improved_model.py")
    print("      (Update the script to use train_highfuel_expanded.csv)")
    print("   2. Compare model performance with expanded data")
    print()


if __name__ == "__main__":
    main()
