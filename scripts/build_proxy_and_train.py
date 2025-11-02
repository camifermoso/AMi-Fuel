"""
Complete pipeline: Preprocess data and train fuel model.
This script integrates the full preprocessing and training workflow.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_preprocessing import preprocess_train_test_split
from fuel_model import train_fuel_model


def main():
    """
    Run the complete preprocessing and training pipeline.
    """
    print("\n" + "="*70)
    print(" "*15 + "AMi-FUEL: COMPLETE TRAINING PIPELINE")
    print("="*70 + "\n")
    
    # Step 1: Preprocess data
    print("STEP 1: DATA PREPROCESSING")
    print("-" * 70)
    
    train_processed, test_processed = preprocess_train_test_split(
        train_path="data/train_highfuel.csv",
        test_path="data/test_highfuel.csv",
        output_dir="data/processed",
        scaler_type='robust',
        verbose=True
    )
    
    # Step 2: Train model on preprocessed training data
    print("\n\nSTEP 2: MODEL TRAINING")
    print("-" * 70)
    
    model = train_fuel_model(
        data_path="data/processed/train_processed.csv",
        model_type='random_forest',
        test_size=0.25,
        cv=5,
        output_path="outputs/fuel_model_real.pkl",
        verbose=True
    )
    
    # Step 3: Evaluate on fully held-out test set (different year/circuits)
    print("\n\nSTEP 3: HELD-OUT YEAR/CIRCUIT EVALUATION")
    print("-" * 70)
    
    import pandas as pd
    test_data = pd.read_csv("data/processed/test_processed.csv")
    
    # Get features used in training
    feature_cols = [
        'avg_rpm', 'avg_throttle', 'avg_speed', 'avg_gear', 
        'avg_drs', 'avg_ers_mode', 'power_estimate', 'speed_per_rpm', 
        'drs_speed_factor', 'energy_intensity', 'speed_per_gear', 
        'speed_variance', 'avg_sector_speed'
    ]
    feature_cols = [f for f in feature_cols if f in test_data.columns]
    
    X_holdout = test_data[feature_cols]
    y_holdout = test_data['fuel_burn_proxy']
    
    holdout_metrics, y_pred_holdout = model.evaluate(X_holdout, y_holdout, verbose=True)
    
    # Save held-out predictions
    holdout_pred_path = "outputs/test_preds_holdout.csv"
    pd.DataFrame({
        'y_true': y_holdout.values,
        'y_pred': y_pred_holdout,
        'residual': y_holdout.values - y_pred_holdout,
        'year': test_data['year'] if 'year' in test_data.columns else None,
        'gp': test_data['gp'] if 'gp' in test_data.columns else None
    }).to_csv(holdout_pred_path, index=False)
    
    print(f"\n✓ Held-out predictions saved to {holdout_pred_path}")
    
    print("\n" + "="*70)
    print(" "*20 + "PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70 + "\n")
    
    print("OUTPUTS:")
    print("  ✓ Preprocessed training data: data/processed/train_processed.csv")
    print("  ✓ Preprocessed test data: data/processed/test_processed.csv")
    print("  ✓ Preprocessor state: data/processed/preprocessor.pkl")
    print("  ✓ Trained model: outputs/fuel_model_real.pkl")
    print("  ✓ Training predictions: outputs/test_predictions.csv")
    print("  ✓ Held-out predictions: outputs/test_preds_holdout.csv")
    print("\n")


if __name__ == "__main__":
    main()
