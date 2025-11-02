"""
Data Preprocessing Pipeline for AMi-Fuel
Handles cleaning, aggregation, normalization, and validation of F1 telemetry data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings

warnings.filterwarnings('ignore')


class TelemetryPreprocessor:
    """
    Comprehensive preprocessing pipeline for F1 telemetry data.
    Handles cleaning, aggregation, normalization, and feature engineering.
    """
    
    def __init__(self, scaler_type: str = 'robust'):
        """
        Initialize the preprocessor.
        
        Args:
            scaler_type: Type of scaler ('standard', 'robust', or 'minmax')
        """
        self.scaler_type = scaler_type
        self.scalers: Dict[str, any] = {}
        self.feature_stats: Dict[str, Dict] = {}
        self.is_fitted = False
        
        # Define expected ranges for each feature (for outlier detection)
        self.feature_ranges = {
            'avg_rpm': (5000, 14000),      # RPM typical range
            'avg_throttle': (0, 100),       # Throttle percentage
            'avg_speed': (50, 350),         # Speed in km/h
            'avg_gear': (1, 8),             # Gear number
            'avg_drs': (0, 14),             # DRS status (can be encoded differently)
            'avg_ers_mode': (0, 5),         # ERS deployment mode
            'SpeedI1': (100, 350),          # Speed at intermediate 1
            'SpeedI2': (100, 350),          # Speed at intermediate 2
            'SpeedFL': (100, 350),          # Speed at finish line
        }
    
    def clean_data(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Clean the raw telemetry data.
        
        Steps:
        1. Remove duplicate rows
        2. Handle missing values
        3. Remove outliers
        4. Ensure correct data types
        5. Filter invalid records
        
        Args:
            df: Raw dataframe
            verbose: Print cleaning statistics
            
        Returns:
            Cleaned dataframe
        """
        if verbose:
            print(f"[CLEAN] Starting with {len(df)} records")
        
        initial_count = len(df)
        
        # 1. Remove exact duplicates
        df = df.drop_duplicates()
        if verbose:
            duplicates_removed = initial_count - len(df)
            print(f"[CLEAN] Removed {duplicates_removed} duplicate rows")
        
        # 2. Ensure numeric columns are properly typed
        numeric_cols = ['avg_rpm', 'avg_throttle', 'avg_speed', 'avg_gear', 
                       'avg_drs', 'avg_ers_mode', 'SpeedI1', 'SpeedI2', 'SpeedFL',
                       'LapNumber']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 3. Handle missing values strategically
        before_na = len(df)
        
        # Critical features: drop rows if missing
        critical_features = ['avg_rpm', 'avg_throttle', 'avg_speed']
        df = df.dropna(subset=[col for col in critical_features if col in df.columns])
        
        if verbose:
            na_removed = before_na - len(df)
            print(f"[CLEAN] Removed {na_removed} rows with missing critical features")
        
        # Non-critical features: fill with sensible defaults
        if 'avg_gear' in df.columns:
            df['avg_gear'] = df['avg_gear'].fillna(df['avg_gear'].median())
        
        if 'avg_drs' in df.columns:
            df['avg_drs'] = df['avg_drs'].fillna(0.0)  # Assume DRS off
        
        if 'avg_ers_mode' in df.columns:
            df['avg_ers_mode'] = df['avg_ers_mode'].fillna(0.0)  # Assume no ERS deployment
        
        # 4. Remove outliers using IQR method and domain knowledge
        before_outliers = len(df)
        df = self._remove_outliers(df)
        
        if verbose:
            outliers_removed = before_outliers - len(df)
            print(f"[CLEAN] Removed {outliers_removed} outlier records")
        
        # 5. Remove invalid lap data
        if 'TrackStatus' in df.columns:
            # Keep only green flag laps (1) and some yellow (2-4 for caution)
            df = df[df['TrackStatus'].isin([1, 2, 3, 4])]
        
        # Remove laps with unrealistic lap times if present
        if 'LapTime' in df.columns:
            df = df[df['LapTime'].notna()]
        
        if verbose:
            final_count = len(df)
            print(f"[CLEAN] Completed: {final_count} records retained "
                  f"({100 * final_count / initial_count:.1f}% of original)")
        
        return df.reset_index(drop=True)
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers using both IQR method and domain-specific ranges.
        
        Args:
            df: DataFrame to clean
            
        Returns:
            DataFrame with outliers removed
        """
        mask = pd.Series([True] * len(df), index=df.index)
        
        for col, (min_val, max_val) in self.feature_ranges.items():
            if col in df.columns:
                # Domain-based filtering
                valid_range = (df[col] >= min_val) & (df[col] <= max_val)
                mask &= valid_range
                
                # IQR-based filtering (for additional outlier detection)
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR  # Using 3*IQR for less aggressive filtering
                upper_bound = Q3 + 3 * IQR
                
                iqr_valid = (df[col] >= lower_bound) & (df[col] <= upper_bound)
                mask &= iqr_valid
        
        return df[mask]
    
    def aggregate_laps(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Aggregate multiple telemetry samples per lap into single lap records.
        The raw data often has multiple rows per lap (from different telemetry samples).
        
        Args:
            df: Cleaned dataframe with multiple samples per lap
            verbose: Print aggregation statistics
            
        Returns:
            Aggregated dataframe with one row per lap
        """
        if verbose:
            print(f"[AGGREGATE] Starting with {len(df)} telemetry samples")
        
        # Identify grouping columns
        group_cols = ['LapNumber', 'Driver', 'year', 'gp']
        group_cols = [col for col in group_cols if col in df.columns]
        
        if not group_cols:
            print("[AGGREGATE] Warning: No grouping columns found, returning as-is")
            return df
        
        # Define aggregation strategies for each column type
        agg_dict = {}
        
        # Telemetry features: use mean for primary aggregation
        telemetry_features = ['avg_rpm', 'avg_throttle', 'avg_speed', 'avg_gear', 
                             'avg_drs', 'avg_ers_mode']
        for col in telemetry_features:
            if col in df.columns:
                agg_dict[col] = 'mean'
        
        # Speed traps: take max (representative of optimal conditions)
        speed_features = ['SpeedI1', 'SpeedI2', 'SpeedFL']
        for col in speed_features:
            if col in df.columns:
                agg_dict[col] = 'max'
        
        # Categorical: take first (they should be same within a lap)
        categorical_features = ['Team', 'Stint', 'Compound', 'TrackStatus']
        for col in categorical_features:
            if col in df.columns:
                agg_dict[col] = 'first'
        
        # LapTime: take mean (in case of multiple samples)
        if 'LapTime' in df.columns:
            agg_dict['LapTime'] = 'first'
        
        # Perform aggregation
        df_agg = df.groupby(group_cols, as_index=False).agg(agg_dict)
        
        # Add aggregation metadata
        lap_sample_counts = df.groupby(group_cols).size().reset_index(name='sample_count')
        df_agg = df_agg.merge(lap_sample_counts, on=group_cols, how='left')
        
        # Add variability metrics (useful for understanding data quality)
        for col in ['avg_rpm', 'avg_throttle', 'avg_speed']:
            if col in df.columns:
                std_col = f'{col}_std'
                std_values = df.groupby(group_cols)[col].std().reset_index(name=std_col)
                df_agg = df_agg.merge(std_values, on=group_cols, how='left')
                df_agg[std_col] = df_agg[std_col].fillna(0)
        
        if verbose:
            print(f"[AGGREGATE] Aggregated to {len(df_agg)} unique laps")
            print(f"[AGGREGATE] Average samples per lap: {df_agg['sample_count'].mean():.1f}")
        
        return df_agg
    
    def engineer_features(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Create derived features that may improve model performance.
        
        Args:
            df: Aggregated dataframe
            verbose: Print feature engineering info
            
        Returns:
            DataFrame with additional engineered features
        """
        if verbose:
            print(f"[FEATURES] Engineering additional features...")
        
        df = df.copy()
        
        # 1. Power-related features
        if 'avg_rpm' in df.columns and 'avg_throttle' in df.columns:
            # Estimated power output (normalized)
            df['power_estimate'] = (df['avg_rpm'] / 12000.0) * (df['avg_throttle'] / 100.0)
        
        # 2. Efficiency metrics
        if 'avg_speed' in df.columns and 'avg_rpm' in df.columns:
            # Speed per RPM (efficiency indicator)
            df['speed_per_rpm'] = df['avg_speed'] / (df['avg_rpm'] + 1)  # Avoid division by zero
        
        # 3. Aerodynamic efficiency
        if 'avg_drs' in df.columns and 'avg_speed' in df.columns:
            # DRS effectiveness
            df['drs_speed_factor'] = df['avg_drs'] * df['avg_speed']
        
        # 4. Energy deployment intensity
        if 'avg_ers_mode' in df.columns and 'avg_throttle' in df.columns:
            # Combined power deployment
            df['energy_intensity'] = (df['avg_ers_mode'] / 5.0) * (df['avg_throttle'] / 100.0)
        
        # 5. Gear efficiency
        if 'avg_gear' in df.columns and 'avg_speed' in df.columns:
            # Speed per gear (indicator of corner vs straight)
            df['speed_per_gear'] = df['avg_speed'] / (df['avg_gear'] + 1)
        
        # 6. Track position indicators
        if all(col in df.columns for col in ['SpeedI1', 'SpeedI2', 'SpeedFL']):
            df['speed_variance'] = df[['SpeedI1', 'SpeedI2', 'SpeedFL']].std(axis=1)
            df['avg_sector_speed'] = df[['SpeedI1', 'SpeedI2', 'SpeedFL']].mean(axis=1)
        
        if verbose:
            engineered_cols = [col for col in df.columns if col in [
                'power_estimate', 'speed_per_rpm', 'drs_speed_factor', 
                'energy_intensity', 'speed_per_gear', 'speed_variance', 'avg_sector_speed'
            ]]
            print(f"[FEATURES] Created {len(engineered_cols)} new features: {engineered_cols}")
        
        return df
    
    def normalize_features(self, df: pd.DataFrame, fit: bool = True, 
                          verbose: bool = True) -> pd.DataFrame:
        """
        Normalize numerical features using the specified scaler.
        
        Args:
            df: DataFrame to normalize
            fit: Whether to fit the scaler (True for training, False for test)
            verbose: Print normalization info
            
        Returns:
            DataFrame with normalized features
        """
        if verbose:
            action = "Fitting and transforming" if fit else "Transforming"
            print(f"[NORMALIZE] {action} features using {self.scaler_type} scaler...")
        
        df = df.copy()
        
        # Features to normalize
        features_to_scale = [
            'avg_rpm', 'avg_throttle', 'avg_speed', 'avg_gear', 
            'avg_drs', 'avg_ers_mode', 'SpeedI1', 'SpeedI2', 'SpeedFL',
            'power_estimate', 'speed_per_rpm', 'drs_speed_factor',
            'energy_intensity', 'speed_per_gear', 'speed_variance', 'avg_sector_speed'
        ]
        
        # Filter to only existing columns
        features_to_scale = [col for col in features_to_scale if col in df.columns]
        
        if not features_to_scale:
            print("[NORMALIZE] Warning: No features to normalize")
            return df
        
        # Initialize or retrieve scaler
        if fit:
            if self.scaler_type == 'standard':
                scaler = StandardScaler()
            elif self.scaler_type == 'robust':
                scaler = RobustScaler()  # Better for data with outliers
            else:
                raise ValueError(f"Unknown scaler type: {self.scaler_type}")
            
            # Fit and transform
            df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
            
            # Store scaler and statistics
            self.scalers['features'] = scaler
            self.feature_stats = {
                col: {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max()
                }
                for col in features_to_scale
            }
            self.is_fitted = True
            
            if verbose:
                print(f"[NORMALIZE] Fitted scaler on {len(features_to_scale)} features")
        
        else:
            if not self.is_fitted:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            
            # Transform only
            scaler = self.scalers['features']
            df[features_to_scale] = scaler.transform(df[features_to_scale])
            
            if verbose:
                print(f"[NORMALIZE] Transformed {len(features_to_scale)} features")
        
        return df
    
    def create_fuel_proxy(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Create the fuel consumption proxy target variable.
        This is a physics-inspired weighted combination of key features.
        
        Args:
            df: DataFrame with features
            verbose: Print proxy creation info
            
        Returns:
            DataFrame with fuel_burn_proxy column added
        """
        if verbose:
            print("[PROXY] Creating fuel consumption proxy...")
        
        df = df.copy()
        
        # Use normalized or raw values depending on state
        rpm_col = 'avg_rpm'
        thr_col = 'avg_throttle'
        ers_col = 'avg_ers_mode'
        
        # Get values
        rpm = df[rpm_col].fillna(df[rpm_col].median())
        throttle = df[thr_col].fillna(df[thr_col].median())
        ers = df[ers_col].fillna(0.0) if ers_col in df.columns else pd.Series(0.0, index=df.index)
        
        # Scale to 0-1 range if not already normalized
        if not self.is_fitted:
            rpm_scaled = np.clip(rpm / 12000.0, 0, 1.2)
            thr_scaled = np.clip(throttle / 100.0, 0, 1.0)
            ers_scaled = np.clip(ers / 4.0, 0, 1.0)
        else:
            # Already normalized, use as-is
            rpm_scaled = rpm
            thr_scaled = throttle
            ers_scaled = ers
        
        # Physics-inspired weights:
        # RPM: 48% - engine speed is primary fuel consumer
        # Throttle: 32% - throttle position directly affects fuel injection
        # ERS: 20% - energy deployment affects fuel efficiency
        df['fuel_burn_proxy'] = (
            0.48 * rpm_scaled + 
            0.32 * thr_scaled + 
            0.20 * ers_scaled
        )
        
        if verbose:
            print(f"[PROXY] Fuel proxy stats: mean={df['fuel_burn_proxy'].mean():.3f}, "
                  f"std={df['fuel_burn_proxy'].std():.3f}")
        
        return df
    
    def validate_data(self, df: pd.DataFrame, stage: str = "unknown") -> Dict:
        """
        Validate data quality and print diagnostic report.
        
        Args:
            df: DataFrame to validate
            stage: Name of the processing stage
            
        Returns:
            Dictionary with validation metrics
        """
        print(f"\n{'='*60}")
        print(f"DATA VALIDATION REPORT - {stage.upper()}")
        print(f"{'='*60}")
        
        report = {
            'stage': stage,
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        print(f"Rows: {report['n_rows']:,}")
        print(f"Columns: {report['n_cols']}")
        print(f"Memory: {report['memory_mb']:.2f} MB")
        
        # Missing values
        missing = df.isnull().sum()
        missing_pct = 100 * missing / len(df)
        if missing.sum() > 0:
            print(f"\nMissing Values:")
            for col in missing[missing > 0].index:
                print(f"  {col}: {missing[col]} ({missing_pct[col]:.1f}%)")
        else:
            print("\nMissing Values: None ✓")
        
        report['missing_values'] = missing.to_dict()
        
        # Data types
        print(f"\nData Types:")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count} columns")
        
        # Key feature statistics
        key_features = ['avg_rpm', 'avg_throttle', 'avg_speed', 'avg_gear']
        key_features = [f for f in key_features if f in df.columns]
        
        if key_features:
            print(f"\nKey Feature Statistics:")
            stats = df[key_features].describe().loc[['mean', 'std', 'min', 'max']]
            print(stats.to_string())
        
        # Check for duplicates
        n_duplicates = df.duplicated().sum()
        report['n_duplicates'] = n_duplicates
        print(f"\nDuplicate Rows: {n_duplicates}")
        
        # Distribution by year/circuit if available
        if 'year' in df.columns and 'gp' in df.columns:
            print(f"\nData Distribution:")
            dist = df.groupby(['year', 'gp']).size().reset_index(name='count')
            print(dist.to_string(index=False))
        
        print(f"{'='*60}\n")
        
        return report
    
    def process_pipeline(self, df: pd.DataFrame, fit: bool = True, 
                        verbose: bool = True) -> pd.DataFrame:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            df: Raw dataframe
            fit: Whether to fit scalers (True for training, False for test)
            verbose: Print detailed progress
            
        Returns:
            Fully preprocessed dataframe
        """
        if verbose:
            print(f"\n{'='*60}")
            print("STARTING PREPROCESSING PIPELINE")
            print(f"{'='*60}\n")
        
        # Stage 1: Cleaning
        df = self.clean_data(df, verbose=verbose)
        if verbose:
            self.validate_data(df, stage="After Cleaning")
        
        # Stage 2: Aggregation
        df = self.aggregate_laps(df, verbose=verbose)
        if verbose:
            self.validate_data(df, stage="After Aggregation")
        
        # Stage 3: Feature Engineering
        df = self.engineer_features(df, verbose=verbose)
        
        # Stage 4: Create Fuel Proxy (before normalization for interpretability)
        df = self.create_fuel_proxy(df, verbose=verbose)
        
        # Stage 5: Normalization
        df = self.normalize_features(df, fit=fit, verbose=verbose)
        
        if verbose:
            self.validate_data(df, stage="After Full Pipeline")
            print(f"{'='*60}")
            print("PREPROCESSING PIPELINE COMPLETED")
            print(f"{'='*60}\n")
        
        return df
    
    def save_preprocessor(self, path: str):
        """Save the preprocessor state (scalers and stats)."""
        import joblib
        state = {
            'scalers': self.scalers,
            'feature_stats': self.feature_stats,
            'is_fitted': self.is_fitted,
            'scaler_type': self.scaler_type,
            'feature_ranges': self.feature_ranges
        }
        joblib.dump(state, path)
        print(f"[SAVE] Preprocessor saved to {path}")
    
    def load_preprocessor(self, path: str):
        """Load the preprocessor state."""
        import joblib
        state = joblib.load(path)
        self.scalers = state['scalers']
        self.feature_stats = state['feature_stats']
        self.is_fitted = state['is_fitted']
        self.scaler_type = state['scaler_type']
        self.feature_ranges = state['feature_ranges']
        print(f"[LOAD] Preprocessor loaded from {path}")


def preprocess_train_test_split(train_path: str, test_path: str, 
                                 output_dir: str = "data/processed",
                                 scaler_type: str = 'robust',
                                 verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to preprocess train and test sets.
    
    Args:
        train_path: Path to training data CSV
        test_path: Path to test data CSV
        output_dir: Directory to save processed data
        scaler_type: Type of scaler to use
        verbose: Print progress
        
    Returns:
        Tuple of (processed_train_df, processed_test_df)
    """
    print(f"\n{'#'*60}")
    print("AMi-FUEL DATA PREPROCESSING")
    print(f"{'#'*60}\n")
    
    # Load data
    print(f"Loading training data from: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"Loading test data from: {test_path}")
    test_df = pd.read_csv(test_path)
    
    # Initialize preprocessor
    preprocessor = TelemetryPreprocessor(scaler_type=scaler_type)
    
    # Process training data (fit=True)
    print("\n" + "="*60)
    print("PROCESSING TRAINING DATA")
    print("="*60)
    train_processed = preprocessor.process_pipeline(train_df, fit=True, verbose=verbose)
    
    # Process test data (fit=False, use fitted scalers)
    print("\n" + "="*60)
    print("PROCESSING TEST DATA")
    print("="*60)
    test_processed = preprocessor.process_pipeline(test_df, fit=False, verbose=verbose)
    
    # Save processed data
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    train_output = Path(output_dir) / "train_processed.csv"
    test_output = Path(output_dir) / "test_processed.csv"
    preprocessor_output = Path(output_dir) / "preprocessor.pkl"
    
    train_processed.to_csv(train_output, index=False)
    test_processed.to_csv(test_output, index=False)
    preprocessor.save_preprocessor(str(preprocessor_output))
    
    print(f"\n{'='*60}")
    print("OUTPUTS")
    print(f"{'='*60}")
    print(f"✓ Processed training data: {train_output}")
    print(f"✓ Processed test data: {test_output}")
    print(f"✓ Preprocessor state: {preprocessor_output}")
    print(f"{'='*60}\n")
    
    return train_processed, test_processed


if __name__ == "__main__":
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess F1 telemetry data for AMi-Fuel")
    parser.add_argument("--train", default="data/train_highfuel.csv", 
                       help="Path to training data CSV")
    parser.add_argument("--test", default="data/test_highfuel.csv",
                       help="Path to test data CSV")
    parser.add_argument("--output-dir", default="data/processed",
                       help="Directory to save processed data")
    parser.add_argument("--scaler", default="robust", choices=['standard', 'robust'],
                       help="Type of scaler to use")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Print detailed progress")
    
    args = parser.parse_args()
    
    # Run preprocessing
    train_processed, test_processed = preprocess_train_test_split(
        train_path=args.train,
        test_path=args.test,
        output_dir=args.output_dir,
        scaler_type=args.scaler,
        verbose=args.verbose
    )
    
    print("\n✓ Preprocessing complete! Ready for model training.")
