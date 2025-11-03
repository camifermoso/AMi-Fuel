"""
Improved Training Pipeline with Advanced Features and Models
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_preprocessing import TelemetryPreprocessor
from enhanced_features import EnhancedFeatureEngineer
from advanced_fuel_model import AdvancedFuelModel


def improved_training_pipeline(
    train_path: str = "data/train_highfuel.csv",
    test_path: str = "data/test_highfuel.csv",
    model_type: str = 'xgboost',
    use_stacking: bool = False,
    tune_hyperparams: bool = True,
    include_enhanced_features: bool = True,
    include_lags: bool = True,
    include_circuits: bool = True,
    output_dir: str = "outputs",
    verbose: bool = True
):
    """
    Run improved training pipeline with advanced features and models.
    
    Args:
        train_path: Path to training data
        test_path: Path to test data
        model_type: 'xgboost', 'lightgbm', 'random_forest', or 'gradient_boosting'
        use_stacking: Whether to use stacking ensemble
        tune_hyperparams: Whether to perform hyperparameter tuning
        include_enhanced_features: Whether to create enhanced features
        include_lags: Whether to include lag features
        include_circuits: Whether to include circuit-specific features
        output_dir: Directory to save outputs
        verbose: Print progress
    """
    print("\n" + "="*70)
    print(" "*10 + "AMi-FUEL: IMPROVED TRAINING PIPELINE")
    print("="*70 + "\n")
    
    # ============== STEP 1: DATA PREPROCESSING ==============
    print("STEP 1: DATA PREPROCESSING")
    print("-" * 70)
    
    # Load data
    print(f"Loading training data from: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"  Loaded {len(train_df)} training samples")
    
    print(f"Loading test data from: {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"  Loaded {len(test_df)} test samples")
    
    # Initialize preprocessor
    preprocessor = TelemetryPreprocessor(scaler_type='robust')
    
    # Basic preprocessing
    print("\nProcessing training data...")
    train_df = preprocessor.clean_data(train_df, verbose=False)
    train_df = preprocessor.aggregate_laps(train_df, verbose=False)
    train_df = preprocessor.engineer_features(train_df, verbose=False)
    
    print("Processing test data...")
    test_df = preprocessor.clean_data(test_df, verbose=False)
    test_df = preprocessor.aggregate_laps(test_df, verbose=False)
    test_df = preprocessor.engineer_features(test_df, verbose=False)
    
    print(f"✓ Basic preprocessing complete")
    print(f"  Training: {len(train_df)} laps")
    print(f"  Test: {len(test_df)} laps")
    
    # ============== STEP 2: ENHANCED FEATURE ENGINEERING ==============
    if include_enhanced_features:
        print("\n\nSTEP 2: ENHANCED FEATURE ENGINEERING")
        print("-" * 70)
        
        train_df = EnhancedFeatureEngineer.create_all_features(
            train_df,
            include_lags=include_lags,
            include_circuits=include_circuits,
            verbose=True
        )
        
        test_df = EnhancedFeatureEngineer.create_all_features(
            test_df,
            include_lags=include_lags,
            include_circuits=include_circuits,
            verbose=False
        )
        
        print(f"✓ Enhanced features created")
    
    # ============== STEP 3: CREATE FUEL PROXY & NORMALIZE ==============
    print("\n\nSTEP 3: FUEL PROXY & NORMALIZATION")
    print("-" * 70)
    
    # Create fuel proxy before normalization
    train_df = preprocessor.create_fuel_proxy(train_df, verbose=True)
    test_df = preprocessor.create_fuel_proxy(test_df, verbose=False)
    
    # Normalize features
    train_df = preprocessor.normalize_features(train_df, fit=True, verbose=True)
    test_df = preprocessor.normalize_features(test_df, fit=False, verbose=False)
    
    # Save preprocessed data
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    train_output = Path(output_dir) / "train_enhanced.csv"
    test_output = Path(output_dir) / "test_enhanced.csv"
    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)
    print(f"\n✓ Preprocessed data saved:")
    print(f"  {train_output}")
    print(f"  {test_output}")
    
    # ============== STEP 4: PREPARE FEATURES ==============
    print("\n\nSTEP 4: FEATURE SELECTION")
    print("-" * 70)
    
    # Identify all numeric features (excluding metadata and target)
    exclude_cols = ['fuel_burn_proxy', 'LapNumber', 'Driver', 'year', 'gp', 
                   'Team', 'Stint', 'Compound', 'TrackStatus', 'LapTime', 'sample_count']
    
    feature_cols = [col for col in train_df.columns 
                   if col not in exclude_cols 
                   and train_df[col].dtype in ['int64', 'float64', 'uint8']]
    
    print(f"Total features selected: {len(feature_cols)}")
    
    # Prepare train-validation-test split
    X_train_full = train_df[feature_cols]
    y_train_full = train_df['fuel_burn_proxy']
    
    X_test = test_df[feature_cols]
    y_test = test_df['fuel_burn_proxy']
    
    # Split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )
    
    print(f"\nData splits:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test (held-out): {len(X_test)} samples")
    
    # ============== STEP 5: MODEL TRAINING ==============
    print("\n\nSTEP 5: MODEL TRAINING")
    print("-" * 70)
    
    model = AdvancedFuelModel(model_type=model_type, use_stacking=use_stacking)
    
    if tune_hyperparams and not use_stacking:
        print(f"Training {model_type} with hyperparameter tuning...")
        model.train(
            X_train,
            y_train,
            tune_hyperparams=True,
            tuning_method='random',
            verbose=verbose
        )
    else:
        print(f"Training {model_type}{'(stacking ensemble)' if use_stacking else ''}...")
        model.train(
            X_train,
            y_train,
            tune_hyperparams=False,
            verbose=verbose
        )
    
    # ============== STEP 6: VALIDATION EVALUATION ==============
    print("\n\nSTEP 6: VALIDATION EVALUATION")
    print("-" * 70)
    
    val_metrics, y_pred_val = model.evaluate(X_val, y_val, verbose=True)
    
    # ============== STEP 7: FINAL TEST EVALUATION ==============
    print("\n\nSTEP 7: HELD-OUT TEST EVALUATION")
    print("-" * 70)
    
    test_metrics, y_pred_test = model.evaluate(X_test, y_test, verbose=True)
    
    # ============== STEP 8: SAVE OUTPUTS ==============
    print("\n\nSTEP 8: SAVING OUTPUTS")
    print("-" * 70)
    
    # Save model
    model_path = Path(output_dir) / f"fuel_model_{model_type}_enhanced.pkl"
    model.save(str(model_path))
    
    # Save predictions
    val_pred_path = Path(output_dir) / "validation_predictions.csv"
    pd.DataFrame({
        'y_true': y_val.values,
        'y_pred': y_pred_val,
        'residual': y_val.values - y_pred_val,
        'abs_error': np.abs(y_val.values - y_pred_val)
    }).to_csv(val_pred_path, index=False)
    print(f"✓ Validation predictions: {val_pred_path}")
    
    test_pred_path = Path(output_dir) / "test_predictions_enhanced.csv"
    pd.DataFrame({
        'y_true': y_test.values,
        'y_pred': y_pred_test,
        'residual': y_test.values - y_pred_test,
        'abs_error': np.abs(y_test.values - y_pred_test),
        'year': test_df['year'].values if 'year' in test_df.columns else None,
        'gp': test_df['gp'].values if 'gp' in test_df.columns else None
    }).to_csv(test_pred_path, index=False)
    print(f"✓ Test predictions: {test_pred_path}")
    
    # Save feature importance
    if model.feature_importance_ is not None:
        importance_path = Path(output_dir) / "feature_importance.csv"
        model.feature_importance_.to_csv(importance_path, index=False)
        print(f"✓ Feature importance: {importance_path}")
    
    # Save metrics summary
    metrics_path = Path(output_dir) / "metrics_summary.txt"
    with open(metrics_path, 'w') as f:
        f.write("AMi-FUEL MODEL PERFORMANCE SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Stacking: {use_stacking}\n")
        f.write(f"Hyperparameter Tuning: {tune_hyperparams}\n")
        f.write(f"Enhanced Features: {include_enhanced_features}\n")
        f.write(f"Features Used: {len(feature_cols)}\n\n")
        
        f.write("VALIDATION METRICS\n")
        f.write("-"*60 + "\n")
        for metric, value in val_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        
        f.write("\nTEST METRICS (HELD-OUT)\n")
        f.write("-"*60 + "\n")
        for metric, value in test_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        
        if model.best_params_ is not None:
            f.write("\nBEST HYPERPARAMETERS\n")
            f.write("-"*60 + "\n")
            for param, value in model.best_params_.items():
                f.write(f"{param}: {value}\n")
    
    print(f"✓ Metrics summary: {metrics_path}")
    
    # ============== COMPLETION ==============
    print("\n" + "="*70)
    print(" "*15 + "TRAINING PIPELINE COMPLETED!")
    print("="*70 + "\n")
    
    print("PERFORMANCE IMPROVEMENT:")
    print(f"  Validation R²: {val_metrics['r2']:.4f}")
    print(f"  Test R²: {test_metrics['r2']:.4f}")
    print(f"  Test MAE: {test_metrics['mae']:.4f}")
    print(f"  Test MAPE: {test_metrics['mape']:.2f}%")
    print()
    
    return model, val_metrics, test_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Improved AMi-Fuel training pipeline")
    parser.add_argument("--train", default="data/train_highfuel.csv")
    parser.add_argument("--test", default="data/test_highfuel.csv")
    parser.add_argument("--model", default="xgboost", 
                       choices=['xgboost', 'lightgbm', 'random_forest', 'gradient_boosting'])
    parser.add_argument("--stacking", action="store_true", 
                       help="Use stacking ensemble")
    parser.add_argument("--tune", action="store_true", default=True,
                       help="Perform hyperparameter tuning")
    parser.add_argument("--no-enhanced-features", action="store_true",
                       help="Skip enhanced feature engineering")
    parser.add_argument("--no-lags", action="store_true",
                       help="Skip lag features")
    parser.add_argument("--no-circuits", action="store_true",
                       help="Skip circuit-specific features")
    parser.add_argument("--output", default="outputs")
    
    args = parser.parse_args()
    
    model, val_metrics, test_metrics = improved_training_pipeline(
        train_path=args.train,
        test_path=args.test,
        model_type=args.model,
        use_stacking=args.stacking,
        tune_hyperparams=args.tune,
        include_enhanced_features=not args.no_enhanced_features,
        include_lags=not args.no_lags,
        include_circuits=not args.no_circuits,
        output_dir=args.output,
        verbose=True
    )
