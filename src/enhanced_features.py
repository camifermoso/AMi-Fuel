"""
Enhanced Feature Engineering for AMi-Fuel
Creates advanced physics-based and interaction features.
"""

import pandas as pd
import numpy as np
from typing import List


class EnhancedFeatureEngineer:
    """
    Advanced feature engineering for F1 fuel consumption prediction.
    """
    
    @staticmethod
    def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between key variables.
        These capture non-linear relationships.
        """
        df = df.copy()
        
        # RPM × Throttle (core power interaction)
        if 'avg_rpm' in df.columns and 'avg_throttle' in df.columns:
            df['rpm_throttle_interaction'] = df['avg_rpm'] * df['avg_throttle']
            df['rpm_squared'] = df['avg_rpm'] ** 2
            df['throttle_squared'] = df['avg_throttle'] ** 2
        
        # Speed × Throttle (acceleration efficiency)
        if 'avg_speed' in df.columns and 'avg_throttle' in df.columns:
            df['speed_throttle_interaction'] = df['avg_speed'] * df['avg_throttle']
        
        # Gear × RPM (engine efficiency zone)
        if 'avg_gear' in df.columns and 'avg_rpm' in df.columns:
            df['gear_rpm_interaction'] = df['avg_gear'] * df['avg_rpm']
        
        # ERS × Throttle (hybrid power deployment)
        if 'avg_ers_mode' in df.columns and 'avg_throttle' in df.columns:
            df['ers_throttle_interaction'] = df['avg_ers_mode'] * df['avg_throttle']
        
        # DRS × Speed (aerodynamic efficiency)
        if 'avg_drs' in df.columns and 'avg_speed' in df.columns:
            df['drs_speed_interaction'] = df['avg_drs'] * df['avg_speed']
        
        return df
    
    @staticmethod
    def create_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ratio-based features that capture efficiency metrics.
        """
        df = df.copy()
        
        # Speed efficiency ratios
        if 'avg_speed' in df.columns:
            if 'avg_rpm' in df.columns:
                df['speed_to_rpm_ratio'] = df['avg_speed'] / (df['avg_rpm'] + 1)
            
            if 'avg_throttle' in df.columns:
                df['speed_to_throttle_ratio'] = df['avg_speed'] / (df['avg_throttle'] + 1)
            
            if 'avg_gear' in df.columns:
                df['speed_to_gear_ratio'] = df['avg_speed'] / (df['avg_gear'] + 1)
        
        # Power efficiency ratios
        if 'avg_rpm' in df.columns and 'avg_gear' in df.columns:
            df['rpm_to_gear_ratio'] = df['avg_rpm'] / (df['avg_gear'] + 1)
        
        # ERS efficiency
        if 'avg_ers_mode' in df.columns and 'avg_rpm' in df.columns:
            df['ers_to_rpm_ratio'] = df['avg_ers_mode'] / (df['avg_rpm'] + 1)
        
        return df
    
    @staticmethod
    def create_polynomial_features(df: pd.DataFrame, degree: int = 2,
                                   features: List[str] = None) -> pd.DataFrame:
        """
        Create polynomial features for key variables.
        """
        df = df.copy()
        
        if features is None:
            features = ['avg_rpm', 'avg_throttle', 'avg_speed']
        
        features = [f for f in features if f in df.columns]
        
        for feature in features:
            for d in range(2, degree + 1):
                df[f'{feature}_pow{d}'] = df[feature] ** d
        
        return df
    
    @staticmethod
    def create_binned_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create categorical bins for continuous variables to capture non-linear patterns.
        """
        df = df.copy()
        
        # RPM zones (different fuel consumption characteristics)
        if 'avg_rpm' in df.columns:
            df['rpm_zone'] = pd.cut(
                df['avg_rpm'],
                bins=[-np.inf, 8000, 10000, 12000, np.inf],
                labels=['low', 'medium', 'high', 'very_high']
            )
            # One-hot encode
            rpm_dummies = pd.get_dummies(df['rpm_zone'], prefix='rpm_zone')
            df = pd.concat([df, rpm_dummies], axis=1)
            df = df.drop('rpm_zone', axis=1)
        
        # Throttle zones
        if 'avg_throttle' in df.columns:
            df['throttle_zone'] = pd.cut(
                df['avg_throttle'],
                bins=[-np.inf, 25, 50, 75, np.inf],
                labels=['low', 'medium', 'high', 'full']
            )
            throttle_dummies = pd.get_dummies(df['throttle_zone'], prefix='throttle_zone')
            df = pd.concat([df, throttle_dummies], axis=1)
            df = df.drop('throttle_zone', axis=1)
        
        # Speed zones
        if 'avg_speed' in df.columns:
            df['speed_zone'] = pd.cut(
                df['avg_speed'],
                bins=[-np.inf, 150, 250, 300, np.inf],
                labels=['slow', 'medium', 'fast', 'very_fast']
            )
            speed_dummies = pd.get_dummies(df['speed_zone'], prefix='speed_zone')
            df = pd.concat([df, speed_dummies], axis=1)
            df = df.drop('speed_zone', axis=1)
        
        return df
    
    @staticmethod
    def create_lag_features(df: pd.DataFrame, lag_cols: List[str] = None, 
                           lags: List[int] = [1, 2, 3]) -> pd.DataFrame:
        """
        Create lag features for sequential lap data.
        Captures how previous laps affect fuel consumption.
        """
        df = df.copy()
        
        if lag_cols is None:
            lag_cols = ['avg_rpm', 'avg_throttle', 'avg_speed']
        
        lag_cols = [col for col in lag_cols if col in df.columns]
        
        # Sort by driver and lap number if available
        if 'Driver' in df.columns and 'LapNumber' in df.columns:
            df = df.sort_values(['Driver', 'LapNumber'])
            
            for col in lag_cols:
                for lag in lags:
                    df[f'{col}_lag{lag}'] = df.groupby('Driver')[col].shift(lag)
                
                # Rolling statistics
                df[f'{col}_rolling_mean_3'] = df.groupby('Driver')[col].transform(
                    lambda x: x.rolling(window=3, min_periods=1).mean()
                )
                df[f'{col}_rolling_std_3'] = df.groupby('Driver')[col].transform(
                    lambda x: x.rolling(window=3, min_periods=1).std()
                ).fillna(0)
        
        return df
    
    @staticmethod
    def create_circuit_specific_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on circuit characteristics if available.
        """
        df = df.copy()
        
        # If we have circuit/GP information
        if 'gp' in df.columns:
            # Encode circuit as categorical
            gp_dummies = pd.get_dummies(df['gp'], prefix='circuit')
            df = pd.concat([df, gp_dummies], axis=1)
            
            # Create power circuit vs. handling circuit indicator
            power_circuits = ['Monza', 'Spa', 'Silverstone', 'Bahrain']
            if df['gp'].dtype == 'object':
                df['is_power_circuit'] = df['gp'].apply(
                    lambda x: 1 if any(pc in str(x) for pc in power_circuits) else 0
                )
        
        # Stint/tire information
        if 'Stint' in df.columns:
            df['stint_number'] = df['Stint'].astype(float)
        
        if 'Compound' in df.columns:
            compound_dummies = pd.get_dummies(df['Compound'], prefix='tire')
            df = pd.concat([df, compound_dummies], axis=1)
        
        return df
    
    @staticmethod
    def create_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical aggregations across speed traps and other measures.
        """
        df = df.copy()
        
        speed_cols = ['SpeedI1', 'SpeedI2', 'SpeedFL']
        speed_cols = [col for col in speed_cols if col in df.columns]
        
        if len(speed_cols) >= 2:
            # Speed consistency (lower = more consistent)
            df['speed_trap_std'] = df[speed_cols].std(axis=1)
            df['speed_trap_range'] = df[speed_cols].max(axis=1) - df[speed_cols].min(axis=1)
            df['speed_trap_cv'] = df['speed_trap_std'] / (df[speed_cols].mean(axis=1) + 1)
            
            # Speed acceleration indicators
            if 'SpeedI1' in df.columns and 'SpeedFL' in df.columns:
                df['speed_gain'] = df['SpeedFL'] - df['SpeedI1']
        
        # Variability features from std columns if they exist
        std_cols = [col for col in df.columns if col.endswith('_std')]
        if std_cols:
            df['overall_variability'] = df[std_cols].mean(axis=1)
        
        return df
    
    @staticmethod
    def create_all_features(df: pd.DataFrame, include_lags: bool = False,
                          include_circuits: bool = False, verbose: bool = True) -> pd.DataFrame:
        """
        Create all enhanced features.
        
        Args:
            df: Input dataframe
            include_lags: Whether to create lag features (for sequential data)
            include_circuits: Whether to create circuit-specific features
            verbose: Print progress
        """
        if verbose:
            print(f"[ENHANCED FEATURES] Creating advanced features...")
            initial_cols = len(df.columns)
        
        # Apply all feature engineering methods
        df = EnhancedFeatureEngineer.create_interaction_features(df)
        df = EnhancedFeatureEngineer.create_ratio_features(df)
        df = EnhancedFeatureEngineer.create_polynomial_features(df, degree=2)
        df = EnhancedFeatureEngineer.create_binned_features(df)
        df = EnhancedFeatureEngineer.create_statistical_features(df)
        
        if include_lags:
            df = EnhancedFeatureEngineer.create_lag_features(df)
        
        if include_circuits:
            df = EnhancedFeatureEngineer.create_circuit_specific_features(df)
        
        # Fill any NaN values created during feature engineering
        df = df.fillna(0)
        
        if verbose:
            new_features = len(df.columns) - initial_cols
            print(f"[ENHANCED FEATURES] Created {new_features} new features")
            print(f"[ENHANCED FEATURES] Total features: {len(df.columns)}")
        
        return df
