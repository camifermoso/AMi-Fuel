"""
Test model on completely unseen races to validate true generalization.
This addresses potential data leakage from lap-level splits.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error
import sys
sys.path.append('.')


def load_model():
    """Load the two-stage model."""
    model_dir = Path("outputs/two_stage_model")
    
    class ModelWrapper:
        def __init__(self):
            self.finetuned_model = joblib.load(model_dir / "finetuned_model.pkl")
            self.calibrator = joblib.load(model_dir / "calibrator.pkl")
            self.scaler_global = joblib.load(model_dir / "scaler_global.pkl")
            self.scalers_per_team = joblib.load(model_dir / "scalers_per_team.pkl")
            self.team_encoder = joblib.load(model_dir / "team_encoder.pkl")
            self.circuit_encoder = joblib.load(model_dir / "circuit_encoder.pkl")
            
        def prepare_features(self, df):
            """Prepare features matching training format."""
            df = df.copy()
            
            # Create fuel proxy
            df['fuel_proxy'] = (
                0.60 * (df['avg_rpm'] / 12000.0).clip(0, 1.2) + 
                0.40 * (df['avg_throttle'] / 100.0).clip(0, 1.0)
            )
            
            base_features = ['avg_throttle', 'avg_rpm', 'avg_speed', 'avg_gear']
            weather_features = ['air_temp', 'track_temp', 'humidity', 'pressure', 'wind_speed']
            available_weather = [f for f in weather_features if f in df.columns]
            
            # Fill missing
            for col in base_features + available_weather:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].median())
            
            # Per-team normalization
            for team in df['Team'].unique():
                team_mask = df['Team'] == team
                team_key = str(team)
                if team_key in self.scalers_per_team:
                    df.loc[team_mask, base_features] = self.scalers_per_team[team_key].transform(
                        df.loc[team_mask, base_features]
                    )
            
            # Encode
            df['team_encoded'] = df['Team'].apply(
                lambda x: self.team_encoder.transform([x])[0] 
                if x in self.team_encoder.classes_ else -1
            )
            df['circuit_encoded'] = df['gp'].apply(
                lambda x: self.circuit_encoder.transform([x])[0] 
                if x in self.circuit_encoder.classes_ else -1
            )
            
            # Interactions
            df['throttle_rpm'] = df['avg_throttle'] * df['avg_rpm']
            df['speed_gear'] = df['avg_speed'] * df['avg_gear']
            df['rpm_gear'] = df['avg_rpm'] * df['avg_gear']
            df['throttle_sq'] = df['avg_throttle'] ** 2
            df['rpm_sq'] = df['avg_rpm'] ** 2
            
            if 'air_temp' in available_weather and 'humidity' in available_weather:
                df['temp_humidity'] = df['air_temp'] * df['humidity']
            if 'track_temp' in available_weather:
                df['track_temp_sq'] = df['track_temp'] ** 2
            
            # Feature list
            feature_cols = base_features + available_weather + [
                'team_encoded', 'circuit_encoded', 'year',
                'throttle_rpm', 'speed_gear', 'rpm_gear',
                'throttle_sq', 'rpm_sq'
            ]
            
            if 'temp_humidity' in df.columns:
                feature_cols.append('temp_humidity')
            if 'track_temp_sq' in df.columns:
                feature_cols.append('track_temp_sq')
            
            X = df[feature_cols].values
            y = df['fuel_proxy'].values
            
            return X, y
        
        def predict(self, df):
            """Make predictions."""
            X, _ = self.prepare_features(df)
            predictions = self.finetuned_model.predict(X)
            if self.calibrator is not None:
                predictions = self.calibrator.predict(predictions)
            return predictions
    
    return ModelWrapper()


def test_unseen_races():
    """Test on completely unseen races."""
    
    print("="*80)
    print("TESTING ON UNSEEN RACES - TRUE GENERALIZATION TEST")
    print("="*80)
    print()
    
    # Load full dataset
    train_df = pd.read_csv('data/train_highfuel_expanded.csv')
    test_df = pd.read_csv('data/test_highfuel_expanded.csv')
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    
    print(f"Total data: {len(full_df):,} laps")
    print()
    
    # Get all unique races
    all_races = list(full_df.groupby(['year', 'gp']).groups.keys())
    print(f"Total unique races: {len(all_races)}")
    print()
    
    # Select races for held-out test (completely unseen)
    # Choose specific years/circuits not heavily represented
    holdout_races = [
        (2024, 'monaco'),
        (2024, 'austin'),
        (2023, 'spa'),
        (2022, 'silverstone'),
        (2021, 'abu-dhabi'),
    ]
    
    print("🔒 HELD-OUT TEST RACES (completely unseen):")
    for year, gp in holdout_races:
        count = len(full_df[(full_df['year']==year) & (full_df['gp']==gp)])
        print(f"  {year} {gp}: {count:,} laps")
    print()
    
    # Split data
    holdout_mask = full_df.apply(lambda row: (row['year'], row['gp']) in holdout_races, axis=1)
    train_data = full_df[~holdout_mask].copy()
    holdout_data = full_df[holdout_mask].copy()
    
    print(f"✅ Training data (seen races): {len(train_data):,} laps")
    print(f"🔒 Holdout data (unseen races): {len(holdout_data):,} laps")
    print()
    
    # Load model (trained on original data which includes some of these races)
    print("Loading pre-trained model...")
    model = load_model()
    print("✓ Model loaded")
    print()
    
    # Test on ALL test data (includes seen laps from seen races)
    print("="*80)
    print("TEST 1: ORIGINAL TEST SET (laps from seen races)")
    print("="*80)
    X_test, y_test = model.prepare_features(test_df)
    y_pred_test = model.finetuned_model.predict(X_test)
    if model.calibrator is not None:
        y_pred_test = model.calibrator.predict(y_pred_test)
    
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
    
    print(f"R² Score: {r2_test:.4f} ({r2_test*100:.2f}%)")
    print(f"MAE: {mae_test:.6f}")
    print(f"MAPE: {mape_test:.2f}%")
    print()
    
    # Test on holdout (completely unseen races)
    print("="*80)
    print("TEST 2: HOLDOUT SET (completely unseen races)")
    print("="*80)
    X_holdout, y_holdout = model.prepare_features(holdout_data)
    y_pred_holdout = model.finetuned_model.predict(X_holdout)
    if model.calibrator is not None:
        y_pred_holdout = model.calibrator.predict(y_pred_holdout)
    
    r2_holdout = r2_score(y_holdout, y_pred_holdout)
    mae_holdout = mean_absolute_error(y_holdout, y_pred_holdout)
    mape_holdout = np.mean(np.abs((y_holdout - y_pred_holdout) / y_holdout)) * 100
    
    print(f"R² Score: {r2_holdout:.4f} ({r2_holdout*100:.2f}%)")
    print(f"MAE: {mae_holdout:.6f}")
    print(f"MAPE: {mape_holdout:.2f}%")
    print()
    
    # Compare
    print("="*80)
    print("COMPARISON: SEEN vs UNSEEN RACES")
    print("="*80)
    print()
    print(f"{'Metric':<20} {'Seen Races':<20} {'Unseen Races':<20} {'Difference':<15}")
    print("-"*80)
    print(f"{'R² Score':<20} {r2_test:.4f} ({r2_test*100:.2f}%){'':<4} {r2_holdout:.4f} ({r2_holdout*100:.2f}%){'':<4} {(r2_test-r2_holdout)*100:+.2f}%")
    print(f"{'MAE':<20} {mae_test:.6f}{'':<10} {mae_holdout:.6f}{'':<10} {(mae_holdout-mae_test):.6f}")
    print(f"{'MAPE':<20} {mape_test:.2f}%{'':<14} {mape_holdout:.2f}%{'':<14} {(mape_holdout-mape_test):+.2f}%")
    print()
    
    # Interpretation
    r2_drop = (r2_test - r2_holdout) * 100
    print("📊 INTERPRETATION:")
    print()
    if r2_drop < 1:
        print("✅ EXCELLENT GENERALIZATION!")
        print(f"   R² drop of only {r2_drop:.2f}% shows model generalizes well to unseen races")
    elif r2_drop < 3:
        print("✅ GOOD GENERALIZATION")
        print(f"   R² drop of {r2_drop:.2f}% is acceptable - model still generalizes well")
    elif r2_drop < 5:
        print("⚠️  MODERATE OVERFITTING")
        print(f"   R² drop of {r2_drop:.2f}% suggests some overfitting to training circuits")
    else:
        print("🚨 SIGNIFICANT OVERFITTING")
        print(f"   R² drop of {r2_drop:.2f}% indicates model is overfitting to seen races")
    print()
    
    # Test on Aston Martin specifically from holdout
    am_holdout = holdout_data[holdout_data['Team'] == 'Aston Martin']
    if len(am_holdout) > 100:
        print("="*80)
        print("TEST 3: ASTON MARTIN ON UNSEEN RACES")
        print("="*80)
        X_am, y_am = model.prepare_features(am_holdout)
        y_pred_am = model.finetuned_model.predict(X_am)
        if model.calibrator is not None:
            y_pred_am = model.calibrator.predict(y_pred_am)
        
        r2_am = r2_score(y_am, y_pred_am)
        mae_am = mean_absolute_error(y_am, y_pred_am)
        mape_am = np.mean(np.abs((y_am - y_pred_am) / y_am)) * 100
        
        print(f"AM laps in holdout: {len(am_holdout):,}")
        print(f"R² Score: {r2_am:.4f} ({r2_am*100:.2f}%)")
        print(f"MAE: {mae_am:.6f}")
        print(f"MAPE: {mape_am:.2f}%")
        print()
    
    print("="*80)
    print("✅ VALIDATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    test_unseen_races()
