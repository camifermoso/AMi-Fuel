"""
Fuel Model Training Module for AMi-Fuel
Trains ensemble models on preprocessed telemetry data to predict fuel consumption.
"""

import argparse
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings('ignore')


class FuelModel:
    """
    Ensemble model for predicting fuel consumption from telemetry data.
    """
    
    def __init__(self, model_type: str = 'random_forest', **model_kwargs):
        """
        Initialize the fuel model.
        
        Args:
            model_type: Type of model ('random_forest' or 'gradient_boosting')
            **model_kwargs: Additional arguments for the model
        """
        self.model_type = model_type
        self.model = None
        self.feature_importance_ = None
        self.training_score_ = None
        
        if model_type == 'random_forest':
            default_params = {
                'n_estimators': 300,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            }
            default_params.update(model_kwargs)
            self.model = RandomForestRegressor(**default_params)
            
        elif model_type == 'gradient_boosting':
            default_params = {
                'n_estimators': 200,
                'learning_rate': 0.1,
                'max_depth': 5,
                'min_samples_split': 5,
                'random_state': 42
            }
            default_params.update(model_kwargs)
            self.model = GradientBoostingRegressor(**default_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X: pd.DataFrame, y: pd.Series, verbose: bool = True) -> dict:
        """
        Train the fuel consumption model.
        
        Args:
            X: Feature dataframe
            y: Target variable (fuel_burn_proxy)
            verbose: Print training progress
            
        Returns:
            Dictionary with training metrics
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"TRAINING {self.model_type.upper().replace('_', ' ')} MODEL")
            print(f"{'='*60}")
            print(f"Training samples: {len(X)}")
            print(f"Features: {X.shape[1]}")
        
        # Fit model
        self.model.fit(X, y)
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Training metrics
        y_pred_train = self.model.predict(X)
        metrics = {
            'r2_train': r2_score(y, y_pred_train),
            'mae_train': mean_absolute_error(y, y_pred_train),
            'rmse_train': np.sqrt(mean_squared_error(y, y_pred_train))
        }
        
        self.training_score_ = metrics
        
        if verbose:
            print(f"\nTraining Metrics:")
            print(f"  R² Score: {metrics['r2_train']:.4f}")
            print(f"  MAE: {metrics['mae_train']:.4f}")
            print(f"  RMSE: {metrics['rmse_train']:.4f}")
            
            if self.feature_importance_ is not None:
                print(f"\nTop 10 Most Important Features:")
                for idx, row in self.feature_importance_.head(10).iterrows():
                    print(f"  {row['feature']:<25} {row['importance']:.4f}")
        
        return metrics
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, verbose: bool = True) -> dict:
        """
        Evaluate the model on test data.
        
        Args:
            X: Feature dataframe
            y: Target variable
            verbose: Print evaluation results
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        y_pred = self.model.predict(X)
        
        metrics = {
            'r2': r2_score(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mape': np.mean(np.abs((y - y_pred) / (y + 1e-8))) * 100  # Avoid division by zero
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print("EVALUATION METRICS")
            print(f"{'='*60}")
            print(f"Test samples: {len(X)}")
            print(f"\nPerformance:")
            print(f"  R² Score: {metrics['r2']:.4f}")
            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  MAPE: {metrics['mape']:.2f}%")
            
            # Residual analysis
            residuals = y - y_pred
            print(f"\nResidual Analysis:")
            print(f"  Mean: {residuals.mean():.4f}")
            print(f"  Std: {residuals.std():.4f}")
            print(f"  Min: {residuals.min():.4f}")
            print(f"  Max: {residuals.max():.4f}")
        
        return metrics, y_pred
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5, 
                      verbose: bool = True) -> dict:
        """
        Perform cross-validation.
        
        Args:
            X: Feature dataframe
            y: Target variable
            cv: Number of folds
            verbose: Print results
            
        Returns:
            Dictionary with cross-validation scores
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"CROSS-VALIDATION ({cv}-fold)")
            print(f"{'='*60}")
        
        cv_scores = cross_val_score(self.model, X, y, cv=cv, 
                                    scoring='r2', n_jobs=-1)
        
        results = {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        if verbose:
            print(f"R² Scores per fold: {cv_scores}")
            print(f"Mean R²: {results['cv_mean']:.4f} (+/- {results['cv_std']:.4f})")
        
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def save(self, path: str):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_importance': self.feature_importance_,
            'training_score': self.training_score_
        }
        joblib.dump(model_data, path)
        print(f"✓ Model saved to {path}")
    
    @staticmethod
    def load(path: str):
        """Load a trained model."""
        model_data = joblib.load(path)
        fuel_model = FuelModel(model_type=model_data['model_type'])
        fuel_model.model = model_data['model']
        fuel_model.feature_importance_ = model_data.get('feature_importance')
        fuel_model.training_score_ = model_data.get('training_score')
        print(f"✓ Model loaded from {path}")
        return fuel_model


def train_fuel_model(data_path: str, model_type: str = 'random_forest',
                     test_size: float = 0.25, cv: int = 5,
                     output_path: str = "outputs/fuel_model.pkl",
                     verbose: bool = True) -> FuelModel:
    """
    Train a fuel consumption model on preprocessed data.
    
    Args:
        data_path: Path to preprocessed data CSV
        model_type: Type of model to train
        test_size: Fraction of data to use for testing
        cv: Number of cross-validation folds
        output_path: Path to save the trained model
        verbose: Print detailed progress
        
    Returns:
        Trained FuelModel instance
    """
    print(f"\n{'#'*60}")
    print("AMi-FUEL MODEL TRAINING")
    print(f"{'#'*60}\n")
    
    # Load data
    print(f"Loading preprocessed data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # Define features
    # Core telemetry features
    feature_cols = [
        'avg_rpm', 'avg_throttle', 'avg_speed', 'avg_gear', 
        'avg_drs', 'avg_ers_mode'
    ]
    
    # Add engineered features if available
    engineered_features = [
        'power_estimate', 'speed_per_rpm', 'drs_speed_factor',
        'energy_intensity', 'speed_per_gear', 'speed_variance', 
        'avg_sector_speed'
    ]
    feature_cols.extend([f for f in engineered_features if f in df.columns])
    
    # Check for missing features
    missing_features = [f for f in feature_cols if f not in df.columns]
    if missing_features:
        print(f"\nWarning: Missing features: {missing_features}")
        feature_cols = [f for f in feature_cols if f in df.columns]
    
    print(f"\nUsing {len(feature_cols)} features:")
    for i, feat in enumerate(feature_cols, 1):
        print(f"  {i}. {feat}")
    
    # Prepare data
    X = df[feature_cols]
    y = df['fuel_burn_proxy']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    print(f"\nData split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Testing: {len(X_test)} samples")
    
    # Initialize and train model
    fuel_model = FuelModel(model_type=model_type)
    train_metrics = fuel_model.train(X_train, y_train, verbose=verbose)
    
    # Cross-validation
    if cv > 1:
        cv_results = fuel_model.cross_validate(X_train, y_train, cv=cv, verbose=verbose)
    
    # Evaluate on test set
    test_metrics, y_pred = fuel_model.evaluate(X_test, y_test, verbose=verbose)
    
    # Save model
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    fuel_model.save(output_path)
    
    # Save predictions for analysis
    pred_output = Path(output_path).parent / "test_predictions.csv"
    pd.DataFrame({
        'y_true': y_test.values,
        'y_pred': y_pred,
        'residual': y_test.values - y_pred
    }).to_csv(pred_output, index=False)
    print(f"✓ Test predictions saved to {pred_output}")
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}\n")
    
    return fuel_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train fuel consumption model on preprocessed data"
    )
    parser.add_argument("--data", required=True,
                       help="Path to preprocessed training data CSV")
    parser.add_argument("--model-type", default="random_forest",
                       choices=['random_forest', 'gradient_boosting'],
                       help="Type of model to train")
    parser.add_argument("--test-size", type=float, default=0.25,
                       help="Fraction of data to use for testing")
    parser.add_argument("--cv", type=int, default=5,
                       help="Number of cross-validation folds")
    parser.add_argument("--save-model", default="outputs/fuel_model.pkl",
                       help="Path to save the trained model")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Print detailed progress")
    
    args = parser.parse_args()
    
    # Train model
    model = train_fuel_model(
        data_path=args.data,
        model_type=args.model_type,
        test_size=args.test_size,
        cv=args.cv,
        output_path=args.save_model,
        verbose=args.verbose
    )
