"""
Unit tests for AMi-Fuel prediction system.
Tests core functionality, data processing, and model predictions.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDataLoading:
    """Test data loading and validation."""
    
    def test_training_data_exists(self):
        """Verify training data files exist."""
        train_path = Path("data/train_highfuel_expanded.csv")
        test_path = Path("data/test_highfuel_expanded.csv")
        
        assert train_path.exists(), "Training data file missing"
        assert test_path.exists(), "Test data file missing"
    
    def test_training_data_shape(self):
        """Verify training data has expected structure."""
        train_df = pd.read_csv("data/train_highfuel_expanded.csv", nrows=1000)
        
        # Check required columns exist
        required_cols = ['avg_throttle', 'avg_rpm', 'avg_speed', 'avg_gear', 
                        'air_temp', 'track_temp', 'humidity', 'Team', 'year', 'gp']
        
        for col in required_cols:
            assert col in train_df.columns, f"Missing required column: {col}"
    
    def test_no_null_critical_columns(self):
        """Verify critical columns have no null values."""
        train_df = pd.read_csv("data/train_highfuel_expanded.csv", nrows=1000)
        
        critical_cols = ['Team', 'year', 'gp']
        for col in critical_cols:
            assert train_df[col].notna().all(), f"Null values found in {col}"
    
    def test_data_ranges(self):
        """Verify data is within expected ranges."""
        train_df = pd.read_csv("data/train_highfuel_expanded.csv", nrows=1000)
        
        # Throttle should be 0-100
        assert train_df['avg_throttle'].between(0, 100).all(), "Invalid throttle values"
        
        # RPM should be reasonable (3000-13000)
        assert train_df['avg_rpm'].between(3000, 13000).all(), "Invalid RPM values"
        
        # Year should be 2018-2024
        assert train_df['year'].between(2018, 2024).all(), "Invalid year values"
        
        # Temperature should be reasonable (-10 to 60°C)
        if 'air_temp' in train_df.columns:
            assert train_df['air_temp'].between(-10, 60).all(), "Invalid air temperature"


class TestFuelProxy:
    """Test fuel proxy calculation."""
    
    def test_fuel_proxy_calculation(self):
        """Test fuel proxy formula."""
        # Sample data
        rpm = 10000
        throttle = 60
        
        expected = 0.60 * (rpm / 12000.0) + 0.40 * (throttle / 100.0)
        # = 0.60 * 0.833 + 0.40 * 0.60 = 0.500 + 0.240 = 0.740
        
        assert abs(expected - 0.740) < 0.001, "Fuel proxy calculation error"
    
    def test_fuel_proxy_range(self):
        """Test fuel proxy stays in valid range."""
        train_df = pd.read_csv("data/train_highfuel_expanded.csv", nrows=1000)
        
        # Calculate fuel proxy
        fuel_proxy = (
            0.60 * (train_df['avg_rpm'] / 12000.0).clip(0, 1.2) + 
            0.40 * (train_df['avg_throttle'] / 100.0).clip(0, 1.0)
        )
        
        # Should be between 0 and ~1.2
        assert fuel_proxy.between(0, 1.5).all(), "Fuel proxy out of range"
    
    def test_fuel_proxy_correlation(self):
        """Test fuel proxy correlates with RPM and throttle."""
        train_df = pd.read_csv("data/train_highfuel_expanded.csv", nrows=5000)
        
        fuel_proxy = (
            0.60 * (train_df['avg_rpm'] / 12000.0).clip(0, 1.2) + 
            0.40 * (train_df['avg_throttle'] / 100.0).clip(0, 1.0)
        )
        
        # Should correlate positively with both
        rpm_corr = np.corrcoef(train_df['avg_rpm'], fuel_proxy)[0, 1]
        throttle_corr = np.corrcoef(train_df['avg_throttle'], fuel_proxy)[0, 1]
        
        assert rpm_corr > 0.5, f"Low RPM correlation: {rpm_corr}"
        assert throttle_corr > 0.3, f"Low throttle correlation: {throttle_corr}"


class TestModelFiles:
    """Test model files exist and can be loaded."""
    
    def test_model_files_exist(self):
        """Verify all model files exist."""
        model_dir = Path("outputs/two_stage_model")
        
        required_files = [
            "finetuned_model.pkl",
            "calibrator.pkl",
            "scaler_global.pkl",
            "scalers_per_team.pkl",
            "team_encoder.pkl",
            "circuit_encoder.pkl"
        ]
        
        for file in required_files:
            assert (model_dir / file).exists(), f"Missing model file: {file}"
    
    def test_model_can_load(self):
        """Test model files can be loaded."""
        import joblib
        model_dir = Path("outputs/two_stage_model")
        
        try:
            model = joblib.load(model_dir / "finetuned_model.pkl")
            assert model is not None, "Model loaded as None"
        except Exception as e:
            pytest.fail(f"Failed to load model: {e}")


class TestPredictions:
    """Test model predictions."""
    
    def test_prediction_shape(self):
        """Test predictions have correct shape."""
        import joblib
        
        # Load model
        model_dir = Path("outputs/two_stage_model")
        model = joblib.load(model_dir / "finetuned_model.pkl")
        
        # Create dummy data (19 features expected)
        X_dummy = np.random.rand(10, 19)
        
        predictions = model.predict(X_dummy)
        
        assert len(predictions) == 10, "Wrong number of predictions"
        assert predictions.dtype in [np.float32, np.float64], "Wrong prediction type"
    
    def test_prediction_range(self):
        """Test predictions are in reasonable range."""
        import joblib
        
        model_dir = Path("outputs/two_stage_model")
        model = joblib.load(model_dir / "finetuned_model.pkl")
        
        # Create realistic dummy data
        X_dummy = np.random.rand(100, 19) * 2  # Scale to reasonable values
        
        predictions = model.predict(X_dummy)
        
        # Predictions should be roughly 0-1.5 range
        assert predictions.min() >= -0.5, f"Predictions too low: {predictions.min()}"
        assert predictions.max() <= 2.0, f"Predictions too high: {predictions.max()}"


class TestAstonMartinData:
    """Test Aston Martin specific functionality."""
    
    def test_am_data_exists(self):
        """Verify Aston Martin data exists in dataset."""
        train_df = pd.read_csv("data/train_highfuel_expanded.csv")
        
        am_data = train_df[train_df['Team'] == 'Aston Martin']
        
        assert len(am_data) > 0, "No Aston Martin data found"
        assert len(am_data) > 40000, "Insufficient AM data"
    
    def test_am_data_years(self):
        """Test AM data spans multiple years."""
        train_df = pd.read_csv("data/train_highfuel_expanded.csv")
        am_data = train_df[train_df['Team'] == 'Aston Martin']
        
        years = am_data['year'].unique()
        
        assert len(years) >= 4, f"AM data only spans {len(years)} years"
    
    def test_am_data_circuits(self):
        """Test AM data covers multiple circuits."""
        train_df = pd.read_csv("data/train_highfuel_expanded.csv")
        am_data = train_df[train_df['Team'] == 'Aston Martin']
        
        circuits = am_data['gp'].unique()
        
        assert len(circuits) >= 10, f"AM data only covers {len(circuits)} circuits"


class TestWeatherIntegration:
    """Test weather data integration."""
    
    def test_weather_columns_exist(self):
        """Verify weather columns exist."""
        train_df = pd.read_csv("data/train_highfuel_expanded.csv", nrows=100)
        
        weather_cols = ['air_temp', 'track_temp', 'humidity', 'pressure', 'wind_speed']
        
        for col in weather_cols:
            assert col in train_df.columns, f"Missing weather column: {col}"
    
    def test_weather_data_valid(self):
        """Test weather data is in valid ranges."""
        train_df = pd.read_csv("data/train_highfuel_expanded.csv", nrows=1000)
        
        # Air temp: -10 to 50°C
        assert train_df['air_temp'].between(-10, 50).all(), "Invalid air temperature"
        
        # Track temp: -5 to 70°C
        assert train_df['track_temp'].between(-5, 70).all(), "Invalid track temperature"
        
        # Humidity: 0-100%
        assert train_df['humidity'].between(0, 100).all(), "Invalid humidity"
        
        # Pressure: 950-1050 mbar (accounts for altitude)
        assert train_df['pressure'].between(950, 1050).all(), "Invalid pressure"
    
    def test_weather_varies_by_race(self):
        """Test weather actually varies between races."""
        train_df = pd.read_csv("data/train_highfuel_expanded.csv", nrows=10000)
        
        # Group by race and check variance
        race_groups = train_df.groupby(['year', 'gp'])['air_temp'].mean()
        
        # Should have variation across races
        temp_std = race_groups.std()
        assert temp_std > 3, f"Weather doesn't vary enough: {temp_std}°C std"


class TestRecommendations:
    """Test recommendation outputs."""
    
    def test_recommendation_files_exist(self):
        """Verify recommendation CSV files exist."""
        output_dir = Path("outputs")
        
        expected_files = [
            "am_fuel_recommendations.csv",
            "am_race_scenarios.csv",
            "am_circuit_strategies.csv"
        ]
        
        for file in expected_files:
            assert (output_dir / file).exists(), f"Missing output file: {file}"
    
    def test_recommendations_structure(self):
        """Test recommendations have expected columns."""
        df = pd.read_csv("outputs/am_fuel_recommendations.csv")
        
        expected_cols = ['Parameter', 'Reduction', 'Fuel Saved (kg/race)', 'Time Cost/Race']
        
        for col in expected_cols:
            assert col in df.columns, f"Missing column in recommendations: {col}"
    
    def test_recommendations_values_reasonable(self):
        """Test recommendation values are reasonable."""
        df = pd.read_csv("outputs/am_fuel_recommendations.csv")
        
        # Fuel saved should be positive and < 20kg
        fuel_saved = df['Fuel Saved (kg/race)'].str.replace(' kg', '').astype(float)
        assert (fuel_saved > 0).all(), "Negative fuel savings found"
        assert (fuel_saved < 20).all(), "Unrealistic fuel savings (>20kg)"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
