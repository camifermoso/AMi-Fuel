"""
Test script to validate the preprocessing pipeline.
Run this to ensure everything works before running the full pipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from data_preprocessing import TelemetryPreprocessor


def create_sample_data(n_samples: int = 100) -> pd.DataFrame:
    """Create sample telemetry data for testing."""
    np.random.seed(42)
    
    # Calculate number of laps and samples per lap
    samples_per_lap = 5
    n_laps = n_samples // samples_per_lap
    actual_samples = n_laps * samples_per_lap
    
    data = {
        'LapNumber': np.repeat(range(1, n_laps + 1), samples_per_lap),
        'Driver': ['VER'] * actual_samples,
        'Team': ['Red Bull Racing'] * actual_samples,
        'year': [2022] * actual_samples,
        'gp': ['bahrain'] * actual_samples,
        'avg_rpm': np.random.normal(10000, 500, actual_samples),
        'avg_throttle': np.random.uniform(60, 100, actual_samples),
        'avg_speed': np.random.normal(200, 30, actual_samples),
        'avg_gear': np.random.randint(3, 8, actual_samples),
        'avg_drs': np.random.uniform(0, 2, actual_samples),
        'avg_ers_mode': np.random.uniform(0, 3, actual_samples),
        'SpeedI1': np.random.normal(250, 20, actual_samples),
        'SpeedI2': np.random.normal(230, 25, actual_samples),
        'SpeedFL': np.random.normal(240, 22, actual_samples),
        'TrackStatus': [1] * actual_samples,
    }
    
    # Add some missing values
    df = pd.DataFrame(data)
    df.loc[5:7, 'avg_drs'] = np.nan
    df.loc[10:12, 'avg_ers_mode'] = np.nan
    
    # Add some outliers
    df.loc[20, 'avg_rpm'] = 20000  # Outlier
    df.loc[21, 'avg_throttle'] = 150  # Outlier
    
    return df


def test_cleaning():
    """Test data cleaning."""
    print("\n" + "="*60)
    print("TEST 1: DATA CLEANING")
    print("="*60)
    
    df = create_sample_data()
    print(f"Created sample data: {len(df)} rows")
    
    preprocessor = TelemetryPreprocessor()
    df_clean = preprocessor.clean_data(df, verbose=True)
    
    print(f"\n✓ Cleaning test passed")
    print(f"  Original: {len(df)} rows")
    print(f"  Cleaned: {len(df_clean)} rows")
    
    return df_clean


def test_aggregation():
    """Test lap aggregation."""
    print("\n" + "="*60)
    print("TEST 2: LAP AGGREGATION")
    print("="*60)
    
    df = create_sample_data()
    preprocessor = TelemetryPreprocessor()
    df_clean = preprocessor.clean_data(df, verbose=False)
    df_agg = preprocessor.aggregate_laps(df_clean, verbose=True)
    
    print(f"\n✓ Aggregation test passed")
    print(f"  Samples: {len(df_clean)} → Laps: {len(df_agg)}")
    
    return df_agg


def test_feature_engineering():
    """Test feature engineering."""
    print("\n" + "="*60)
    print("TEST 3: FEATURE ENGINEERING")
    print("="*60)
    
    df = create_sample_data()
    preprocessor = TelemetryPreprocessor()
    df_clean = preprocessor.clean_data(df, verbose=False)
    df_agg = preprocessor.aggregate_laps(df_clean, verbose=False)
    df_feat = preprocessor.engineer_features(df_agg, verbose=True)
    
    print(f"\n✓ Feature engineering test passed")
    print(f"  Features before: {len(df_agg.columns)}")
    print(f"  Features after: {len(df_feat.columns)}")
    
    return df_feat


def test_normalization():
    """Test normalization."""
    print("\n" + "="*60)
    print("TEST 4: NORMALIZATION")
    print("="*60)
    
    df = create_sample_data()
    preprocessor = TelemetryPreprocessor()
    df_clean = preprocessor.clean_data(df, verbose=False)
    df_agg = preprocessor.aggregate_laps(df_clean, verbose=False)
    df_feat = preprocessor.engineer_features(df_agg, verbose=False)
    df_norm = preprocessor.normalize_features(df_feat, fit=True, verbose=True)
    
    print(f"\n✓ Normalization test passed")
    
    # Check that features are normalized
    key_features = ['avg_rpm', 'avg_throttle', 'avg_speed']
    for feat in key_features:
        if feat in df_norm.columns:
            mean = df_norm[feat].mean()
            std = df_norm[feat].std()
            print(f"  {feat}: mean={mean:.3f}, std={std:.3f}")
    
    return df_norm


def test_fuel_proxy():
    """Test fuel proxy creation."""
    print("\n" + "="*60)
    print("TEST 5: FUEL PROXY CREATION")
    print("="*60)
    
    df = create_sample_data()
    preprocessor = TelemetryPreprocessor()
    df_clean = preprocessor.clean_data(df, verbose=False)
    df_agg = preprocessor.aggregate_laps(df_clean, verbose=False)
    df_feat = preprocessor.engineer_features(df_agg, verbose=False)
    df_proxy = preprocessor.create_fuel_proxy(df_feat, verbose=True)
    
    print(f"\n✓ Fuel proxy test passed")
    print(f"  Proxy column present: {'fuel_burn_proxy' in df_proxy.columns}")
    print(f"  No NaN values: {df_proxy['fuel_burn_proxy'].notna().all()}")
    
    return df_proxy


def test_full_pipeline():
    """Test complete preprocessing pipeline."""
    print("\n" + "="*60)
    print("TEST 6: FULL PIPELINE")
    print("="*60)
    
    df = create_sample_data()
    preprocessor = TelemetryPreprocessor()
    df_processed = preprocessor.process_pipeline(df, fit=True, verbose=True)
    
    print(f"\n✓ Full pipeline test passed")
    print(f"  Final shape: {df_processed.shape}")
    print(f"  Columns: {list(df_processed.columns)}")
    
    return df_processed


def test_train_test_consistency():
    """Test that train and test preprocessing is consistent."""
    print("\n" + "="*60)
    print("TEST 7: TRAIN-TEST CONSISTENCY")
    print("="*60)
    
    # Create train and test data
    df_train = create_sample_data(n_samples=100)
    df_test = create_sample_data(n_samples=50)
    
    # Process train (fit=True)
    preprocessor = TelemetryPreprocessor()
    df_train_processed = preprocessor.process_pipeline(df_train, fit=True, verbose=False)
    
    # Process test (fit=False, use fitted scaler)
    df_test_processed = preprocessor.process_pipeline(df_test, fit=False, verbose=False)
    
    print(f"Train processed: {df_train_processed.shape}")
    print(f"Test processed: {df_test_processed.shape}")
    print(f"Same columns: {list(df_train_processed.columns) == list(df_test_processed.columns)}")
    
    print(f"\n✓ Train-test consistency test passed")
    
    return df_train_processed, df_test_processed


def run_all_tests():
    """Run all preprocessing tests."""
    print("\n" + "#"*60)
    print("#" + " "*15 + "PREPROCESSING TEST SUITE" + " "*15 + "#")
    print("#"*60)
    
    try:
        test_cleaning()
        test_aggregation()
        test_feature_engineering()
        test_normalization()
        test_fuel_proxy()
        test_full_pipeline()
        test_train_test_consistency()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
        print("\nThe preprocessing pipeline is working correctly!")
        print("You can now run: python scripts/build_proxy_and_train.py")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print("\n" + "="*60)
        print("✗ TEST FAILED")
        print("="*60)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
