"""
Two-Stage Training Pipeline for Aston Martin Fuel Optimization

Stage 1: PRETRAIN on all teams
- Learn general fuel-telemetry patterns across all teams
- Normalize per-team to reduce hardware bias
- Include team/circuit indicators

Stage 2: FINE-TUNE on Aston Martin
- Freeze most model capacity
- Fine-tune on AM-specific data
- Calibrate predictions to AM's fuel scale
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


class TwoStageAMFuelModel:
    """
    Two-stage fuel prediction model:
    1. Pretrain on all teams (breadth)
    2. Fine-tune on Aston Martin (specificity)
    """
    
    def __init__(self):
        self.pretrain_model = None
        self.finetuned_model = None
        self.calibrator = None
        self.scaler_global = StandardScaler()
        self.scalers_per_team = {}
        self.team_encoder = LabelEncoder()
        self.circuit_encoder = LabelEncoder()
        
    def normalize_per_team(self, df, features_to_normalize, fit=False):
        """Z-score normalize features per team to reduce hardware bias."""
        df = df.copy()
        
        for team in df['Team'].unique():
            team_mask = df['Team'] == team
            team_key = str(team)
            
            if fit:
                # Fit scaler for this team
                scaler = StandardScaler()
                df.loc[team_mask, features_to_normalize] = scaler.fit_transform(
                    df.loc[team_mask, features_to_normalize]
                )
                self.scalers_per_team[team_key] = scaler
            else:
                # Use existing scaler
                if team_key in self.scalers_per_team:
                    df.loc[team_mask, features_to_normalize] = self.scalers_per_team[team_key].transform(
                        df.loc[team_mask, features_to_normalize]
                    )
                else:
                    print(f"⚠️  No scaler for team {team}, using global scaler")
                    df.loc[team_mask, features_to_normalize] = self.scaler_global.transform(
                        df.loc[team_mask, features_to_normalize]
                    )
        
        return df
    
    def prepare_features(self, df, fit=False):
        """Prepare features with team/circuit indicators and per-team normalization."""
        df = df.copy()
        
        # Create fuel proxy target
        if 'avg_ers_mode' in df.columns and not df['avg_ers_mode'].isna().all():
            df['fuel_proxy'] = (
                0.48 * (df['avg_rpm'] / 12000.0).clip(0, 1.2) + 
                0.32 * (df['avg_throttle'] / 100.0).clip(0, 1.0) + 
                0.20 * (df['avg_ers_mode'] / 4.0).clip(0, 1.0)
            )
        else:
            df['fuel_proxy'] = (
                0.60 * (df['avg_rpm'] / 12000.0).clip(0, 1.2) + 
                0.40 * (df['avg_throttle'] / 100.0).clip(0, 1.0)
            )
        
        # Drop rows with missing target
        df = df.dropna(subset=['fuel_proxy'])
        
        # Core telemetry features
        base_features = ['avg_throttle', 'avg_rpm', 'avg_speed', 'avg_gear']
        
        # Fill missing values
        for col in base_features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Per-team normalization of telemetry
        df = self.normalize_per_team(df, base_features, fit=fit)
        
        # Encode team and circuit
        if fit:
            df['team_encoded'] = self.team_encoder.fit_transform(df['Team'])
            df['circuit_encoded'] = self.circuit_encoder.fit_transform(df['gp'])
        else:
            # Handle unseen teams/circuits gracefully
            df['team_encoded'] = df['Team'].apply(
                lambda x: self.team_encoder.transform([x])[0] 
                if x in self.team_encoder.classes_ else -1
            )
            df['circuit_encoded'] = df['gp'].apply(
                lambda x: self.circuit_encoder.transform([x])[0] 
                if x in self.circuit_encoder.classes_ else -1
            )
        
        # Create interaction features
        df['throttle_rpm'] = df['avg_throttle'] * df['avg_rpm']
        df['speed_gear'] = df['avg_speed'] * df['avg_gear']
        df['rpm_gear'] = df['avg_rpm'] * df['avg_gear']
        
        # Polynomial features
        df['throttle_sq'] = df['avg_throttle'] ** 2
        df['rpm_sq'] = df['avg_rpm'] ** 2
        
        # Final feature set
        feature_cols = base_features + [
            'team_encoded', 'circuit_encoded', 'year',
            'throttle_rpm', 'speed_gear', 'rpm_gear',
            'throttle_sq', 'rpm_sq'
        ]
        
        X = df[feature_cols].copy()
        y = df['fuel_proxy'].copy()
        
        return X, y, df
    
    def stage1_pretrain(self, train_df, val_df=None):
        """
        Stage 1: Pretrain on ALL teams
        - Learn general fuel-telemetry patterns
        - Use all available data for maximum diversity
        """
        print("="*80)
        print("STAGE 1: PRETRAINING ON ALL TEAMS")
        print("="*80)
        print()
        
        # Show team distribution
        print("📊 Training data distribution:")
        print(train_df['Team'].value_counts())
        print()
        
        # Prepare features
        print("Preparing features with per-team normalization...")
        X_train, y_train, _ = self.prepare_features(train_df, fit=True)
        
        if val_df is not None:
            X_val, y_val, _ = self.prepare_features(val_df, fit=False)
        else:
            # Split off validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.15, random_state=42
            )
        
        print(f"✓ Training samples: {len(X_train):,}")
        print(f"✓ Validation samples: {len(X_val):,}")
        print(f"✓ Features: {X_train.shape[1]}")
        print()
        
        # Train XGBoost model (general patterns)
        print("Training pretrained model...")
        self.pretrain_model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=50
        )
        
        self.pretrain_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Evaluate
        y_train_pred = self.pretrain_model.predict(X_train)
        y_val_pred = self.pretrain_model.predict(X_val)
        
        train_r2 = r2_score(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        
        print()
        print("✓ Pretraining complete!")
        print(f"   Train R²: {train_r2:.4f}, MAE: {train_mae:.4f}")
        print(f"   Val R²: {val_r2:.4f}, MAE: {val_mae:.4f}")
        print()
        
        return self.pretrain_model
    
    def stage2_finetune(self, am_train_df, am_val_df=None, learning_rate=0.01):
        """
        Stage 2: Fine-tune on Aston Martin
        - Use pretrained model as initialization
        - Lower learning rate for careful adaptation
        - Add isotonic calibration layer
        """
        print("="*80)
        print("STAGE 2: FINE-TUNING ON ASTON MARTIN")
        print("="*80)
        print()
        
        # Prepare AM-specific data
        print("Preparing Aston Martin data...")
        X_am_train, y_am_train, _ = self.prepare_features(am_train_df, fit=False)
        
        if am_val_df is not None:
            X_am_val, y_am_val, _ = self.prepare_features(am_val_df, fit=False)
        else:
            X_am_train, X_am_val, y_am_train, y_am_val = train_test_split(
                X_am_train, y_am_train, test_size=0.15, random_state=42
            )
        
        print(f"✓ AM Training samples: {len(X_am_train):,}")
        print(f"✓ AM Validation samples: {len(X_am_val):,}")
        print()
        
        # Fine-tune: use pretrained model with lower LR
        print(f"Fine-tuning with learning_rate={learning_rate}...")
        self.finetuned_model = xgb.XGBRegressor(
            n_estimators=200,  # Fewer trees for fine-tuning
            max_depth=6,  # Slightly shallower
            learning_rate=learning_rate,  # Lower LR
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=30
        )
        
        # Initialize from pretrained model (warm start simulation)
        # XGBoost doesn't have warm_start, so we continue training
        self.finetuned_model.fit(
            X_am_train, y_am_train,
            eval_set=[(X_am_val, y_am_val)],
            verbose=False,
            xgb_model=self.pretrain_model.get_booster()  # Continue from pretrained
        )
        
        # Calibrate predictions to AM's fuel scale
        print("Calibrating to Aston Martin fuel scale...")
        y_am_train_pred = self.finetuned_model.predict(X_am_train)
        
        # Isotonic regression calibration
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(y_am_train_pred, y_am_train)
        
        # Evaluate fine-tuned model
        y_am_train_calibrated = self.calibrator.predict(y_am_train_pred)
        y_am_val_pred = self.finetuned_model.predict(X_am_val)
        y_am_val_calibrated = self.calibrator.predict(y_am_val_pred)
        
        train_r2 = r2_score(y_am_train, y_am_train_calibrated)
        train_mae = mean_absolute_error(y_am_train, y_am_train_calibrated)
        val_r2 = r2_score(y_am_val, y_am_val_calibrated)
        val_mae = mean_absolute_error(y_am_val, y_am_val_calibrated)
        
        print()
        print("✓ Fine-tuning complete!")
        print(f"   AM Train R²: {train_r2:.4f}, MAE: {train_mae:.4f}")
        print(f"   AM Val R²: {val_r2:.4f}, MAE: {val_mae:.4f}")
        print()
        
        return self.finetuned_model
    
    def predict(self, df, use_finetuned=True, calibrate=True):
        """Make predictions using the two-stage model."""
        X, _, _ = self.prepare_features(df, fit=False)
        
        if use_finetuned and self.finetuned_model is not None:
            y_pred = self.finetuned_model.predict(X)
        else:
            y_pred = self.pretrain_model.predict(X)
        
        if calibrate and self.calibrator is not None:
            y_pred = self.calibrator.predict(y_pred)
        
        return y_pred
    
    def save(self, output_dir="outputs/two_stage_model"):
        """Save the two-stage model."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.pretrain_model, output_dir / "pretrain_model.pkl")
        joblib.dump(self.finetuned_model, output_dir / "finetuned_model.pkl")
        joblib.dump(self.calibrator, output_dir / "calibrator.pkl")
        joblib.dump(self.scaler_global, output_dir / "scaler_global.pkl")
        joblib.dump(self.scalers_per_team, output_dir / "scalers_per_team.pkl")
        joblib.dump(self.team_encoder, output_dir / "team_encoder.pkl")
        joblib.dump(self.circuit_encoder, output_dir / "circuit_encoder.pkl")
        
        print(f"✓ Model saved to {output_dir}")


def compare_baselines(train_df, test_df):
    """
    Compare three approaches:
    1. All teams (baseline)
    2. AM only (car-specific but limited)
    3. Two-stage (best of both)
    """
    print("="*80)
    print("BASELINE COMPARISON")
    print("="*80)
    print()
    
    am_train = train_df[train_df['Team'] == 'Aston Martin'].copy()
    am_test = test_df[test_df['Team'] == 'Aston Martin'].copy()
    
    print(f"All Teams Train: {len(train_df):,} laps")
    print(f"All Teams Test: {len(test_df):,} laps")
    print(f"AM Only Train: {len(am_train):,} laps")
    print(f"AM Only Test: {len(am_test):,} laps")
    print()
    
    results = {}
    
    # Baseline 1: Train on AM only
    print("📊 Baseline 1: AM-Only Model")
    print("-"*80)
    model_am = TwoStageAMFuelModel()
    model_am.stage1_pretrain(am_train)
    
    y_test_am = model_am.predict(am_test, use_finetuned=False, calibrate=False)
    X_test, y_true, _ = model_am.prepare_features(am_test, fit=False)
    
    results['am_only'] = {
        'r2': r2_score(y_true, y_test_am),
        'mae': mean_absolute_error(y_true, y_test_am),
        'mape': np.mean(np.abs((y_true - y_test_am) / y_true)) * 100
    }
    
    print(f"✓ R²: {results['am_only']['r2']:.4f}")
    print(f"✓ MAE: {results['am_only']['mae']:.4f}")
    print(f"✓ MAPE: {results['am_only']['mape']:.2f}%")
    print()
    
    # Baseline 2: Train on all teams, test on AM
    print("📊 Baseline 2: All-Teams Model (tested on AM)")
    print("-"*80)
    model_all = TwoStageAMFuelModel()
    model_all.stage1_pretrain(train_df)
    
    y_test_all = model_all.predict(am_test, use_finetuned=False, calibrate=False)
    X_test, y_true, _ = model_all.prepare_features(am_test, fit=False)
    
    results['all_teams'] = {
        'r2': r2_score(y_true, y_test_all),
        'mae': mean_absolute_error(y_true, y_test_all),
        'mape': np.mean(np.abs((y_true - y_test_all) / y_true)) * 100
    }
    
    print(f"✓ R²: {results['all_teams']['r2']:.4f}")
    print(f"✓ MAE: {results['all_teams']['mae']:.4f}")
    print(f"✓ MAPE: {results['all_teams']['mape']:.2f}%")
    print()
    
    # Two-stage: Pretrain on all, fine-tune on AM
    print("📊 Two-Stage Model: Pretrain (All) → Fine-tune (AM)")
    print("-"*80)
    model_two_stage = TwoStageAMFuelModel()
    model_two_stage.stage1_pretrain(train_df)
    model_two_stage.stage2_finetune(am_train)
    
    y_test_two_stage = model_two_stage.predict(am_test, use_finetuned=True, calibrate=True)
    X_test, y_true, _ = model_two_stage.prepare_features(am_test, fit=False)
    
    results['two_stage'] = {
        'r2': r2_score(y_true, y_test_two_stage),
        'mae': mean_absolute_error(y_true, y_test_two_stage),
        'mape': np.mean(np.abs((y_true - y_test_two_stage) / y_true)) * 100
    }
    
    print(f"✓ R²: {results['two_stage']['r2']:.4f}")
    print(f"✓ MAE: {results['two_stage']['mae']:.4f}")
    print(f"✓ MAPE: {results['two_stage']['mape']:.2f}%")
    print()
    
    # Summary
    print("="*80)
    print("SUMMARY: MODEL COMPARISON")
    print("="*80)
    print()
    print(f"{'Model':<30} {'R²':>10} {'MAE':>10} {'MAPE':>10}")
    print("-"*80)
    print(f"{'AM-Only (limited data)':<30} {results['am_only']['r2']:>10.4f} {results['am_only']['mae']:>10.4f} {results['am_only']['mape']:>9.2f}%")
    print(f"{'All-Teams (generic)':<30} {results['all_teams']['r2']:>10.4f} {results['all_teams']['mae']:>10.4f} {results['all_teams']['mape']:>9.2f}%")
    print(f"{'Two-Stage (best)':<30} {results['two_stage']['r2']:>10.4f} {results['two_stage']['mae']:>10.4f} {results['two_stage']['mape']:>9.2f}%")
    print()
    
    # Save best model
    model_two_stage.save()
    
    return model_two_stage, results


def main():
    """Main execution."""
    print()
    print("="*80)
    print("TWO-STAGE ASTON MARTIN FUEL MODEL")
    print("Stage 1: Pretrain on all teams (breadth)")
    print("Stage 2: Fine-tune on Aston Martin (specificity)")
    print("="*80)
    print()
    
    # Load data
    train_df = pd.read_csv('data/train_highfuel.csv')
    test_df = pd.read_csv('data/test_highfuel.csv')
    
    # Compare approaches
    model, results = compare_baselines(train_df, test_df)
    
    print("✅ Training complete!")
    print()
    print("Model saved to: outputs/two_stage_model/")
    print()
    print("Next steps:")
    print("  - Use the two-stage model for AM-specific fuel predictions")
    print("  - Generate AM-specific recommendations")
    print("  - Compare with generic all-teams recommendations")
    print()


if __name__ == "__main__":
    main()
