"""
Advanced Fuel Model with XGBoost, Stacking, and Hyperparameter Tuning
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import warnings

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Run: pip install xgboost")

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not installed. Run: pip install lightgbm")

warnings.filterwarnings('ignore')


class AdvancedFuelModel:
    """
    Advanced ensemble model with hyperparameter tuning and stacking.
    """
    
    def __init__(self, model_type: str = 'xgboost', use_stacking: bool = False):
        """
        Initialize advanced model.
        
        Args:
            model_type: 'xgboost', 'lightgbm', 'random_forest', or 'gradient_boosting'
            use_stacking: Whether to use stacking ensemble
        """
        self.model_type = model_type
        self.use_stacking = use_stacking
        self.model = None
        self.best_params_ = None
        self.feature_importance_ = None
        
    def get_model(self, **params):
        """Get the base model with specified parameters."""
        if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            default_params = {
                'n_estimators': 500,
                'learning_rate': 0.05,
                'max_depth': 7,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'n_jobs': -1
            }
            default_params.update(params)
            return XGBRegressor(**default_params)
            
        elif self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            default_params = {
                'n_estimators': 500,
                'learning_rate': 0.05,
                'max_depth': 7,
                'num_leaves': 31,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            default_params.update(params)
            return LGBMRegressor(**default_params)
            
        elif self.model_type == 'random_forest':
            default_params = {
                'n_estimators': 500,
                'max_depth': 20,
                'min_samples_split': 3,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'bootstrap': True,
                'random_state': 42,
                'n_jobs': -1
            }
            default_params.update(params)
            return RandomForestRegressor(**default_params)
            
        elif self.model_type == 'gradient_boosting':
            default_params = {
                'n_estimators': 500,
                'learning_rate': 0.05,
                'max_depth': 7,
                'min_samples_split': 3,
                'min_samples_leaf': 2,
                'subsample': 0.8,
                'random_state': 42
            }
            default_params.update(params)
            return GradientBoostingRegressor(**default_params)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def create_stacking_ensemble(self):
        """Create a stacking ensemble with multiple base models."""
        base_estimators = []
        
        # Add Random Forest
        base_estimators.append((
            'rf',
            RandomForestRegressor(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        ))
        
        # Add Gradient Boosting
        base_estimators.append((
            'gb',
            GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            )
        ))
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            base_estimators.append((
                'xgb',
                XGBRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=6,
                    random_state=42,
                    n_jobs=-1
                )
            ))
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            base_estimators.append((
                'lgbm',
                LGBMRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=6,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
            ))
        
        # Meta-learner: Ridge regression for stability
        meta_learner = Ridge(alpha=1.0)
        
        return StackingRegressor(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=5,
            n_jobs=-1
        )
    
    def hyperparameter_tuning(self, X, y, method='grid', n_iter=50, cv=5, verbose=True):
        """
        Perform hyperparameter tuning.
        
        Args:
            X: Feature matrix
            y: Target variable
            method: 'grid' or 'random'
            n_iter: Number of iterations for random search
            cv: Number of cross-validation folds
            verbose: Print progress
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"HYPERPARAMETER TUNING - {self.model_type.upper()}")
            print(f"Method: {method.upper()}")
            print(f"{'='*60}")
        
        base_model = self.get_model()
        
        # Define parameter grids
        if self.model_type == 'xgboost':
            param_grid = {
                'n_estimators': [300, 500, 700],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [5, 7, 9],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'gamma': [0, 0.1, 0.2],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0.5, 1.0, 2.0]
            }
            
        elif self.model_type == 'lightgbm':
            param_grid = {
                'n_estimators': [300, 500, 700],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [5, 7, 9],
                'num_leaves': [20, 31, 50],
                'min_child_samples': [10, 20, 30],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0.5, 1.0, 2.0]
            }
            
        elif self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [300, 500, 700],
                'max_depth': [15, 20, 25, None],
                'min_samples_split': [2, 3, 5],
                'min_samples_leaf': [1, 2, 3],
                'max_features': ['sqrt', 'log2', 0.5]
            }
            
        elif self.model_type == 'gradient_boosting':
            param_grid = {
                'n_estimators': [300, 500, 700],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [5, 7, 9],
                'min_samples_split': [2, 3, 5],
                'min_samples_leaf': [1, 2, 3],
                'subsample': [0.7, 0.8, 0.9]
            }
        
        # Perform search
        if method == 'grid':
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring='r2',
                n_jobs=-1,
                verbose=1 if verbose else 0
            )
        else:  # random
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring='r2',
                n_jobs=-1,
                random_state=42,
                verbose=1 if verbose else 0
            )
        
        search.fit(X, y)
        
        self.best_params_ = search.best_params_
        self.model = search.best_estimator_
        
        if verbose:
            print(f"\nBest Parameters:")
            for param, value in self.best_params_.items():
                print(f"  {param}: {value}")
            print(f"\nBest CV R² Score: {search.best_score_:.4f}")
        
        return self.best_params_
    
    def train(self, X, y, tune_hyperparams=False, tuning_method='random', 
              verbose=True):
        """
        Train the model.
        
        Args:
            X: Feature matrix
            y: Target variable
            tune_hyperparams: Whether to perform hyperparameter tuning
            tuning_method: 'grid' or 'random'
            verbose: Print progress
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"TRAINING ADVANCED MODEL - {self.model_type.upper()}")
            print(f"{'='*60}")
            print(f"Training samples: {len(X)}")
            print(f"Features: {X.shape[1]}")
        
        if tune_hyperparams:
            self.hyperparameter_tuning(X, y, method=tuning_method, verbose=verbose)
        else:
            if self.use_stacking:
                self.model = self.create_stacking_ensemble()
            else:
                self.model = self.get_model()
            
            self.model.fit(X, y)
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        elif self.use_stacking and hasattr(self.model.estimators_[0], 'feature_importances_'):
            # Average importance across base models
            importances = []
            for name, estimator in self.model.estimators_:
                if hasattr(estimator, 'feature_importances_'):
                    importances.append(estimator.feature_importances_)
            
            if importances:
                avg_importance = np.mean(importances, axis=0)
                self.feature_importance_ = pd.DataFrame({
                    'feature': X.columns,
                    'importance': avg_importance
                }).sort_values('importance', ascending=False)
        
        # Training metrics
        y_pred_train = self.model.predict(X)
        metrics = {
            'r2_train': r2_score(y, y_pred_train),
            'mae_train': mean_absolute_error(y, y_pred_train),
            'rmse_train': np.sqrt(mean_squared_error(y, y_pred_train))
        }
        
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
    
    def evaluate(self, X, y, verbose=True):
        """Evaluate model on test data."""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        y_pred = self.model.predict(X)
        
        metrics = {
            'r2': r2_score(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mape': np.mean(np.abs((y - y_pred) / (y + 1e-8))) * 100
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
        
        return metrics, y_pred
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained.")
        return self.model.predict(X)
    
    def save(self, path):
        """Save the trained model."""
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'use_stacking': self.use_stacking,
            'best_params': self.best_params_,
            'feature_importance': self.feature_importance_
        }
        joblib.dump(model_data, path)
        print(f"✓ Model saved to {path}")
    
    @staticmethod
    def load(path):
        """Load a trained model."""
        model_data = joblib.load(path)
        model = AdvancedFuelModel(
            model_type=model_data['model_type'],
            use_stacking=model_data.get('use_stacking', False)
        )
        model.model = model_data['model']
        model.best_params_ = model_data.get('best_params')
        model.feature_importance_ = model_data.get('feature_importance')
        print(f"✓ Model loaded from {path}")
        return model
