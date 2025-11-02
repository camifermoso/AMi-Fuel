"""
Fuel Optimization Module for AMi-Fuel
Optimizes throttle and ERS deployment strategies to minimize fuel consumption.
"""

import argparse
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore')


class FuelOptimizer:
    """
    Optimizes driving parameters to minimize fuel consumption while maintaining performance.
    """
    
    def __init__(self, model_path: str, preprocessor_path: str = None):
        """
        Initialize the optimizer.
        
        Args:
            model_path: Path to trained fuel model
            preprocessor_path: Path to preprocessor (optional, for denormalization)
        """
        self.model = self._load_model(model_path)
        self.preprocessor = None
        
        if preprocessor_path and Path(preprocessor_path).exists():
            self.preprocessor = joblib.load(preprocessor_path)
            print(f"✓ Loaded preprocessor from {preprocessor_path}")
    
    def _load_model(self, path: str):
        """Load the trained model."""
        model_data = joblib.load(path)
        if isinstance(model_data, dict):
            return model_data['model']
        return model_data
    
    def calculate_baseline(self, data: pd.DataFrame, features: List[str]) -> float:
        """
        Calculate baseline fuel consumption.
        
        Args:
            data: Dataframe with telemetry
            features: List of feature names
            
        Returns:
            Total fuel consumption (sum of predictions)
        """
        X = data[features]
        predictions = self.model.predict(X)
        return predictions.sum()
    
    def optimize_strategy(self, data: pd.DataFrame, features: List[str],
                         throttle_range: Tuple[float, float] = (0.90, 0.999),
                         throttle_steps: int = 15,
                         ers_range: Tuple[float, float] = (-0.1, 0.15),
                         ers_steps: int = 15,
                         verbose: bool = True) -> Dict:
        """
        Find optimal throttle scaling and ERS shift to minimize fuel consumption.
        
        Args:
            data: Dataframe with telemetry
            features: List of feature names
            throttle_range: (min, max) for throttle scaling factor
            throttle_steps: Number of throttle values to test
            ers_range: (min, max) for ERS shift
            ers_steps: Number of ERS values to test
            verbose: Print optimization progress
            
        Returns:
            Dictionary with optimal parameters and metrics
        """
        if verbose:
            print(f"\n{'='*60}")
            print("FUEL OPTIMIZATION")
            print(f"{'='*60}")
            print(f"Testing {throttle_steps} × {ers_steps} = "
                  f"{throttle_steps * ers_steps} configurations...")
        
        # Calculate baseline
        baseline_fuel = self.calculate_baseline(data, features)
        
        if verbose:
            print(f"Baseline fuel consumption: {baseline_fuel:.4f}")
            print("\nOptimizing...")
        
        # Grid search over throttle and ERS parameters
        best_result = None
        min_fuel = float('inf')
        
        throttle_values = np.linspace(throttle_range[0], throttle_range[1], throttle_steps)
        ers_values = np.linspace(ers_range[0], ers_range[1], ers_steps)
        
        results = []
        
        for throttle_scale in throttle_values:
            for ers_shift in ers_values:
                # Create modified scenario
                sim_data = data.copy()
                
                # Apply throttle scaling
                if 'avg_throttle' in sim_data.columns:
                    # Clip to reasonable range (60-100% throttle)
                    sim_data['avg_throttle'] = (sim_data['avg_throttle'] * throttle_scale).clip(0.6, 1.0)
                
                # Apply ERS shift
                if 'avg_ers_mode' in sim_data.columns:
                    sim_data['avg_ers_mode'] = (sim_data['avg_ers_mode'] + ers_shift).clip(0, 1.0)
                
                # Recalculate dependent features
                if 'power_estimate' in sim_data.columns:
                    if 'avg_rpm' in sim_data.columns and 'avg_throttle' in sim_data.columns:
                        sim_data['power_estimate'] = (
                            (sim_data['avg_rpm'] / 12000.0) * (sim_data['avg_throttle'] / 100.0)
                        )
                
                if 'energy_intensity' in sim_data.columns:
                    if 'avg_ers_mode' in sim_data.columns and 'avg_throttle' in sim_data.columns:
                        sim_data['energy_intensity'] = (
                            (sim_data['avg_ers_mode'] / 5.0) * (sim_data['avg_throttle'] / 100.0)
                        )
                
                # Predict fuel consumption
                X_sim = sim_data[features]
                fuel_consumption = self.model.predict(X_sim).sum()
                
                result = {
                    'throttle_scale': throttle_scale,
                    'ers_shift': ers_shift,
                    'fuel_consumption': fuel_consumption,
                    'fuel_saved': baseline_fuel - fuel_consumption,
                    'fuel_saved_pct': 100 * (baseline_fuel - fuel_consumption) / baseline_fuel
                }
                results.append(result)
                
                # Track best result
                if fuel_consumption < min_fuel:
                    min_fuel = fuel_consumption
                    best_result = result.copy()
        
        # Add baseline to best result
        best_result['baseline_fuel'] = baseline_fuel
        
        if verbose:
            print(f"\n{'='*60}")
            print("OPTIMIZATION RESULTS")
            print(f"{'='*60}")
            print(f"Baseline fuel: {baseline_fuel:.4f}")
            print(f"Optimized fuel: {best_result['fuel_consumption']:.4f}")
            print(f"Fuel saved: {best_result['fuel_saved']:.4f} "
                  f"({best_result['fuel_saved_pct']:.2f}%)")
            print(f"\nOptimal Parameters:")
            print(f"  Throttle scale: {best_result['throttle_scale']:.4f} "
                  f"({(1 - best_result['throttle_scale'])*100:.2f}% reduction)")
            print(f"  ERS shift: {best_result['ers_shift']:.4f}")
            print(f"{'='*60}\n")
        
        return best_result, pd.DataFrame(results)
    
    def optimize_lap_by_lap(self, data: pd.DataFrame, features: List[str],
                           fuel_target: float = None,
                           verbose: bool = True) -> pd.DataFrame:
        """
        Optimize fuel usage on a lap-by-lap basis.
        
        Args:
            data: Dataframe with telemetry
            features: List of feature names
            fuel_target: Target total fuel consumption (if None, minimize)
            verbose: Print progress
            
        Returns:
            DataFrame with lap-by-lap optimization recommendations
        """
        if verbose:
            print(f"\n{'='*60}")
            print("LAP-BY-LAP OPTIMIZATION")
            print(f"{'='*60}")
        
        results = []
        
        for idx, row in data.iterrows():
            # Get baseline fuel for this lap
            X_baseline = pd.DataFrame([row[features]])
            baseline_fuel = self.model.predict(X_baseline)[0]
            
            # Try different strategies for this lap
            best_lap_fuel = baseline_fuel
            best_throttle = 1.0
            best_ers = 0.0
            
            for throttle_scale in np.linspace(0.92, 0.999, 10):
                for ers_shift in np.linspace(-0.05, 0.1, 10):
                    lap_sim = row.copy()
                    
                    if 'avg_throttle' in lap_sim.index:
                        lap_sim['avg_throttle'] = (lap_sim['avg_throttle'] * throttle_scale).clip(0.6, 1.0)
                    
                    if 'avg_ers_mode' in lap_sim.index:
                        lap_sim['avg_ers_mode'] = (lap_sim['avg_ers_mode'] + ers_shift).clip(0, 1.0)
                    
                    X_sim = pd.DataFrame([lap_sim[features]])
                    lap_fuel = self.model.predict(X_sim)[0]
                    
                    if lap_fuel < best_lap_fuel:
                        best_lap_fuel = lap_fuel
                        best_throttle = throttle_scale
                        best_ers = ers_shift
            
            results.append({
                'lap': idx,
                'baseline_fuel': baseline_fuel,
                'optimized_fuel': best_lap_fuel,
                'fuel_saved': baseline_fuel - best_lap_fuel,
                'optimal_throttle_scale': best_throttle,
                'optimal_ers_shift': best_ers
            })
        
        results_df = pd.DataFrame(results)
        
        if verbose:
            total_baseline = results_df['baseline_fuel'].sum()
            total_optimized = results_df['optimized_fuel'].sum()
            total_saved = total_baseline - total_optimized
            
            print(f"\nTotal baseline fuel: {total_baseline:.4f}")
            print(f"Total optimized fuel: {total_optimized:.4f}")
            print(f"Total saved: {total_saved:.4f} ({100*total_saved/total_baseline:.2f}%)")
            print(f"{'='*60}\n")
        
        return results_df
    
    def save_strategy(self, strategy: Dict, output_path: str):
        """Save optimization strategy to CSV."""
        Path(output_path).parent.mkdir(exist_ok=True, parents=True)
        pd.DataFrame([strategy]).to_csv(output_path, index=False)
        print(f"✓ Strategy saved to {output_path}")


def main():
    """Main optimization workflow."""
    parser = argparse.ArgumentParser(description="Optimize fuel consumption strategy")
    parser.add_argument("--data", required=True,
                       help="Path to telemetry data CSV")
    parser.add_argument("--model", required=True,
                       help="Path to trained fuel model")
    parser.add_argument("--preprocessor", default=None,
                       help="Path to preprocessor (optional)")
    parser.add_argument("--out", default="outputs/optimized_strategy.csv",
                       help="Path to save optimization results")
    parser.add_argument("--lap-by-lap", action="store_true",
                       help="Perform lap-by-lap optimization")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Print detailed progress")
    
    args = parser.parse_args()
    
    print(f"\n{'#'*60}")
    print("AMi-FUEL OPTIMIZATION")
    print(f"{'#'*60}\n")
    
    # Load data
    print(f"Loading data from: {args.data}")
    data = pd.read_csv(args.data)
    print(f"Loaded {len(data)} samples")
    
    # Define features
    features = [
        'avg_rpm', 'avg_throttle', 'avg_speed', 'avg_gear', 
        'avg_drs', 'avg_ers_mode'
    ]
    
    # Add engineered features if available
    engineered = [
        'power_estimate', 'speed_per_rpm', 'drs_speed_factor',
        'energy_intensity', 'speed_per_gear', 'speed_variance', 'avg_sector_speed'
    ]
    features.extend([f for f in engineered if f in data.columns])
    
    print(f"Using {len(features)} features for optimization")
    
    # Initialize optimizer
    optimizer = FuelOptimizer(
        model_path=args.model,
        preprocessor_path=args.preprocessor
    )
    
    if args.lap_by_lap:
        # Lap-by-lap optimization
        results_df = optimizer.optimize_lap_by_lap(data, features, verbose=args.verbose)
        
        output_path = args.out.replace('.csv', '_lap_by_lap.csv')
        results_df.to_csv(output_path, index=False)
        print(f"✓ Lap-by-lap results saved to {output_path}")
    
    else:
        # Global optimization
        best_strategy, all_results = optimizer.optimize_strategy(
            data, features, verbose=args.verbose
        )
        
        # Save best strategy
        optimizer.save_strategy(best_strategy, args.out)
        
        # Save all results for analysis
        all_results_path = args.out.replace('.csv', '_all_configs.csv')
        all_results.to_csv(all_results_path, index=False)
        print(f"✓ All configurations saved to {all_results_path}")
    
    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()