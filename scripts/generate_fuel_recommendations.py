"""
Numerical Fuel Reduction Recommendations
Generates actionable parameter changes with quantified fuel savings.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path


def load_model_and_data():
    """Load the trained model and test data."""
    # Try to load the most recent enhanced model
    model_files = [
        'outputs/fuel_model_xgboost_enhanced.pkl',
        'outputs/fuel_model_random_forest_enhanced.pkl',
        'outputs/fuel_model_real.pkl'
    ]
    
    model = None
    for mf in model_files:
        if Path(mf).exists():
            model_data = joblib.load(mf)
            model = model_data['model'] if isinstance(model_data, dict) else model_data
            print(f"✓ Loaded model from {mf}")
            break
    
    if model is None:
        raise FileNotFoundError("No trained model found. Run training first.")
    
    # Load test data
    test_data = pd.read_csv('outputs/test_enhanced.csv')
    
    return model, test_data


def calculate_sensitivity(model, test_data, feature_cols):
    """
    Calculate how much fuel consumption changes when each parameter changes.
    Returns sensitivity coefficients (% fuel change per % parameter change).
    """
    # Get baseline predictions
    X_baseline = test_data[feature_cols].copy()
    baseline_fuel = model.predict(X_baseline).mean()
    
    sensitivities = {}
    
    # Test each feature with small perturbations
    perturbations = [-0.10, -0.05, -0.02, 0.02, 0.05, 0.10]  # ±2%, 5%, 10%
    
    for feature in feature_cols:
        if feature in ['avg_throttle', 'avg_rpm', 'avg_speed', 'avg_gear', 
                      'avg_drs', 'avg_ers_mode']:
            
            fuel_changes = []
            
            for pct_change in perturbations:
                X_modified = X_baseline.copy()
                X_modified[feature] = X_modified[feature] * (1 + pct_change)
                
                new_fuel = model.predict(X_modified).mean()
                fuel_pct_change = ((new_fuel - baseline_fuel) / baseline_fuel) * 100
                
                fuel_changes.append({
                    'param_change': pct_change * 100,
                    'fuel_change': fuel_pct_change
                })
            
            # Calculate average sensitivity (fuel change per param change)
            avg_sensitivity = np.mean([fc['fuel_change'] / fc['param_change'] 
                                      for fc in fuel_changes if fc['param_change'] != 0])
            
            sensitivities[feature] = {
                'sensitivity': avg_sensitivity,
                'changes': fuel_changes,
                'baseline_value': X_baseline[feature].mean()
            }
    
    return sensitivities, baseline_fuel


def generate_recommendations_table(sensitivities, baseline_fuel, test_data):
    """Generate actionable recommendations table."""
    
    recommendations = []
    
    # Get actual value ranges from data
    param_info = {
        'avg_throttle': {
            'name': 'Throttle',
            'unit': '%',
            'typical_range': (test_data['avg_throttle'].quantile(0.25), 
                            test_data['avg_throttle'].quantile(0.75)),
            'baseline': test_data['avg_throttle'].mean()
        },
        'avg_rpm': {
            'name': 'Engine RPM',
            'unit': 'RPM',
            'typical_range': (test_data['avg_rpm'].quantile(0.25), 
                            test_data['avg_rpm'].quantile(0.75)),
            'baseline': test_data['avg_rpm'].mean()
        },
        'avg_speed': {
            'name': 'Speed',
            'unit': 'km/h',
            'typical_range': (test_data['avg_speed'].quantile(0.25), 
                            test_data['avg_speed'].quantile(0.75)),
            'baseline': test_data['avg_speed'].mean()
        },
        'avg_gear': {
            'name': 'Gear',
            'unit': '',
            'typical_range': (test_data['avg_gear'].quantile(0.25), 
                            test_data['avg_gear'].quantile(0.75)),
            'baseline': test_data['avg_gear'].mean()
        },
        'avg_drs': {
            'name': 'DRS Usage',
            'unit': '%',
            'typical_range': (test_data['avg_drs'].quantile(0.25), 
                            test_data['avg_drs'].quantile(0.75)),
            'baseline': test_data['avg_drs'].mean()
        },
        'avg_ers_mode': {
            'name': 'ERS Deployment',
            'unit': 'mode',
            'typical_range': (test_data['avg_ers_mode'].quantile(0.25), 
                            test_data['avg_ers_mode'].quantile(0.75)),
            'baseline': test_data['avg_ers_mode'].mean()
        }
    }
    
    # Generate recommendations for each parameter
    for param, sens_data in sensitivities.items():
        if param not in param_info:
            continue
        
        info = param_info[param]
        sensitivity = sens_data['sensitivity']
        baseline_val = info['baseline']
        
        # Generate multiple reduction scenarios
        for reduction_pct in [2, 5, 10]:
            new_value = baseline_val * (1 - reduction_pct/100)
            fuel_saving = -sensitivity * reduction_pct  # Negative because we're reducing
            
            recommendations.append({
                'Parameter': info['name'],
                'Current Avg': f"{baseline_val:.1f} {info['unit']}",
                'Reduce To': f"{new_value:.1f} {info['unit']}",
                'Change': f"-{reduction_pct}%",
                'Fuel Saved': f"{fuel_saving:.2f}%",
                'Difficulty': assess_difficulty(param, reduction_pct),
                'Race Impact': assess_race_impact(param, reduction_pct)
            })
    
    return pd.DataFrame(recommendations)


def assess_difficulty(param, reduction_pct):
    """Assess implementation difficulty."""
    difficulty_map = {
        'avg_throttle': {2: 'Easy', 5: 'Easy', 10: 'Medium'},
        'avg_rpm': {2: 'Medium', 5: 'Medium', 10: 'Hard'},
        'avg_speed': {2: 'Hard', 5: 'Hard', 10: 'Very Hard'},
        'avg_gear': {2: 'Easy', 5: 'Medium', 10: 'Hard'},
        'avg_drs': {2: 'Easy', 5: 'Easy', 10: 'Medium'},
        'avg_ers_mode': {2: 'Easy', 5: 'Medium', 10: 'Medium'}
    }
    return difficulty_map.get(param, {}).get(reduction_pct, 'Medium')


def assess_race_impact(param, reduction_pct):
    """Assess impact on lap time / race performance."""
    impact_map = {
        'avg_throttle': {2: 'Low', 5: 'Medium', 10: 'High'},
        'avg_rpm': {2: 'Low', 5: 'Medium', 10: 'High'},
        'avg_speed': {2: 'High', 5: 'Very High', 10: 'Extreme'},
        'avg_gear': {2: 'Low', 5: 'Medium', 10: 'High'},
        'avg_drs': {2: 'Low', 5: 'Low', 10: 'Medium'},
        'avg_ers_mode': {2: 'Low', 5: 'Medium', 10: 'High'}
    }
    return impact_map.get(param, {}).get(reduction_pct, 'Medium')


def generate_combined_strategies(sensitivities, param_info):
    """Generate multi-parameter strategies for maximum fuel savings."""
    
    strategies = []
    
    # Strategy 1: Throttle Management (Conservative)
    strategies.append({
        'Strategy': 'Conservative Throttle',
        'Changes': 'Throttle -3%, RPM -2%',
        'Total Fuel Saved': f"{calculate_combined_saving(sensitivities, {'avg_throttle': -3, 'avg_rpm': -2}):.2f}%",
        'Lap Time Impact': '0.1-0.2s slower',
        'Best For': 'Long stints, tire management'
    })
    
    # Strategy 2: Aerodynamic Efficiency
    strategies.append({
        'Strategy': 'Aero Efficiency',
        'Changes': 'DRS +5%, Speed -1%',
        'Total Fuel Saved': f"{calculate_combined_saving(sensitivities, {'avg_drs': 5, 'avg_speed': -1}):.2f}%",
        'Lap Time Impact': 'Minimal',
        'Best For': 'High-speed circuits'
    })
    
    # Strategy 3: Engine Management
    strategies.append({
        'Strategy': 'Engine Lift & Coast',
        'Changes': 'Throttle -5%, RPM -3%, Speed -2%',
        'Total Fuel Saved': f"{calculate_combined_saving(sensitivities, {'avg_throttle': -5, 'avg_rpm': -3, 'avg_speed': -2}):.2f}%",
        'Lap Time Impact': '0.3-0.5s slower',
        'Best For': 'Critical fuel saving'
    })
    
    # Strategy 4: Power Unit Optimization
    strategies.append({
        'Strategy': 'PU Mode Optimization',
        'Changes': 'Throttle -2%, ERS +10%',
        'Total Fuel Saved': f"{calculate_combined_saving(sensitivities, {'avg_throttle': -2, 'avg_ers_mode': 10}):.2f}%",
        'Lap Time Impact': 'Neutral to positive',
        'Best For': 'Energy-rich phases'
    })
    
    return pd.DataFrame(strategies)


def calculate_combined_saving(sensitivities, changes):
    """Calculate total fuel saving from multiple parameter changes."""
    total_saving = 0
    for param, pct_change in changes.items():
        if param in sensitivities:
            # Multiply sensitivity by % change (negative for reductions)
            total_saving += sensitivities[param]['sensitivity'] * pct_change
    return abs(total_saving)


def main():
    """Generate complete numerical recommendations report."""
    
    print("="*100)
    print(" " * 25 + "AMi-FUEL NUMERICAL RECOMMENDATIONS")
    print(" " * 30 + "Actionable Fuel Savings")
    print("="*100)
    print()
    
    # Load model and data
    print("Loading model and calculating sensitivities...")
    model, test_data = load_model_and_data()
    
    # Get ALL feature columns used in training (excluding target and metadata)
    exclude_cols = ['fuel_burn_proxy', 'LapNumber', 'Driver', 'year', 'gp', 
                   'Team', 'Stint', 'Compound', 'TrackStatus', 'LapTime', 'sample_count']
    
    feature_cols = [col for col in test_data.columns 
                   if col not in exclude_cols 
                   and test_data[col].dtype in ['int64', 'float64', 'uint8']]
    
    # Calculate sensitivities
    sensitivities, baseline_fuel = calculate_sensitivity(model, test_data, feature_cols)
    
    print(f"✓ Analysis complete. Baseline fuel: {baseline_fuel:.4f} units/lap")
    print()
    print()
    
    # Generate recommendations table
    print("┌" + "─" * 98 + "┐")
    print("│" + " " * 25 + "INDIVIDUAL PARAMETER RECOMMENDATIONS" + " " * 37 + "│")
    print("└" + "─" * 98 + "┘")
    print()
    
    param_info = {}  # We'll populate this in generate_recommendations_table
    recommendations_df = generate_recommendations_table(sensitivities, baseline_fuel, test_data)
    
    # Sort by fuel saved (descending)
    recommendations_df['Fuel_Saved_Numeric'] = recommendations_df['Fuel Saved'].str.rstrip('%').astype(float)
    recommendations_df = recommendations_df.sort_values('Fuel_Saved_Numeric', ascending=False)
    recommendations_df = recommendations_df.drop('Fuel_Saved_Numeric', axis=1)
    
    # Display table
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print(recommendations_df.to_string(index=False))
    print()
    print()
    
    # Generate combined strategies
    print("┌" + "─" * 98 + "┐")
    print("│" + " " * 30 + "COMBINED STRATEGIES" + " " * 49 + "│")
    print("└" + "─" * 98 + "┘")
    print()
    
    strategies_df = generate_combined_strategies(sensitivities, param_info)
    print(strategies_df.to_string(index=False))
    print()
    print()
    
    # Key insights
    print("┌" + "─" * 98 + "┐")
    print("│" + " " * 35 + "KEY INSIGHTS" + " " * 51 + "│")
    print("└" + "─" * 98 + "┘")
    print()
    
    # Find most sensitive parameter
    most_sensitive = max(sensitivities.items(), 
                        key=lambda x: abs(x[1]['sensitivity']))
    
    print(f"🎯 MOST IMPACTFUL PARAMETER: {most_sensitive[0]}")
    print(f"   • Sensitivity: {most_sensitive[1]['sensitivity']:.3f}% fuel per 1% change")
    print(f"   • Reducing by 5% → saves ~{abs(most_sensitive[1]['sensitivity'] * 5):.2f}% fuel")
    print()
    
    # Quick wins
    print("💡 QUICK WINS (Easy + High Savings):")
    quick_wins = recommendations_df[
        (recommendations_df['Difficulty'] == 'Easy') & 
        (recommendations_df['Race Impact'].isin(['Low', 'Medium']))
    ].head(3)
    for idx, row in quick_wins.iterrows():
        print(f"   • {row['Parameter']}: {row['Change']} → {row['Fuel Saved']} saved")
    print()
    
    # Critical savings
    print("🔥 CRITICAL FUEL SAVING MODE (When fuel-limited):")
    critical = recommendations_df.nlargest(3, recommendations_df['Fuel Saved'].str.rstrip('%').astype(float))
    for idx, row in critical.iterrows():
        print(f"   • {row['Parameter']}: {row['Change']} → {row['Fuel Saved']} saved ({row['Race Impact']} lap time impact)")
    print()
    
    print("="*100)
    print()
    
    # Save to CSV
    recommendations_df.to_csv('outputs/fuel_reduction_recommendations.csv', index=False)
    strategies_df.to_csv('outputs/fuel_saving_strategies.csv', index=False)
    
    print("✓ Tables saved:")
    print("  • outputs/fuel_reduction_recommendations.csv")
    print("  • outputs/fuel_saving_strategies.csv")
    print()


if __name__ == "__main__":
    main()
