"""
Visualization utilities for AMi-Fuel results.
Creates plots for fuel comparison, optimization results, and model performance.
"""

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path


def plot_predictions(predictions_path: str, output_dir: str = "outputs"):
    """
    Plot actual vs predicted fuel consumption.
    
    Args:
        predictions_path: Path to predictions CSV
        output_dir: Directory to save plots
    """
    df = pd.read_csv(predictions_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    axes[0].scatter(df['y_true'], df['y_pred'], alpha=0.5, s=20)
    axes[0].plot([df['y_true'].min(), df['y_true'].max()], 
                 [df['y_true'].min(), df['y_true'].max()], 
                 'r--', lw=2, label='Perfect prediction')
    axes[0].set_xlabel('Actual Fuel Consumption')
    axes[0].set_ylabel('Predicted Fuel Consumption')
    axes[0].set_title('Prediction Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residual plot
    if 'residual' in df.columns:
        residuals = df['residual']
    else:
        residuals = df['y_true'] - df['y_pred']
    
    axes[1].scatter(df['y_pred'], residuals, alpha=0.5, s=20)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Fuel Consumption')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "prediction_accuracy.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved prediction plot to {output_path}")


def plot_optimization_results(strategy_path: str, all_configs_path: str = None,
                              output_dir: str = "outputs"):
    """
    Plot optimization results showing fuel savings.
    
    Args:
        strategy_path: Path to optimized strategy CSV
        all_configs_path: Path to all configurations tested (optional)
        output_dir: Directory to save plots
    """
    strategy = pd.read_csv(strategy_path)
    
    if all_configs_path and Path(all_configs_path).exists():
        all_configs = pd.read_csv(all_configs_path)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Heatmap of fuel consumption
        pivot = all_configs.pivot_table(
            values='fuel_consumption',
            index='ers_shift',
            columns='throttle_scale',
            aggfunc='mean'
        )
        
        im = axes[0].imshow(pivot.values, aspect='auto', cmap='RdYlGn_r')
        axes[0].set_xlabel('Throttle Scale')
        axes[0].set_ylabel('ERS Shift')
        axes[0].set_title('Fuel Consumption Heatmap')
        
        # Set tick labels
        n_ticks = 5
        x_ticks = np.linspace(0, len(pivot.columns) - 1, n_ticks, dtype=int)
        y_ticks = np.linspace(0, len(pivot.index) - 1, n_ticks, dtype=int)
        axes[0].set_xticks(x_ticks)
        axes[0].set_yticks(y_ticks)
        axes[0].set_xticklabels([f"{pivot.columns[i]:.3f}" for i in x_ticks])
        axes[0].set_yticklabels([f"{pivot.index[i]:.3f}" for i in y_ticks])
        
        plt.colorbar(im, ax=axes[0], label='Fuel Consumption')
        
        # Savings distribution
        axes[1].hist(all_configs['fuel_saved_pct'], bins=30, edgecolor='black', alpha=0.7)
        axes[1].axvline(strategy['fuel_saved_pct'].iloc[0], color='r', 
                       linestyle='--', lw=2, label='Best strategy')
        axes[1].set_xlabel('Fuel Saved (%)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Fuel Savings')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Bar chart showing baseline vs optimized
        categories = ['Baseline', 'Optimized', 'Saved']
        values = [
            strategy['baseline_fuel'].iloc[0],
            strategy['fuel_consumption'].iloc[0],
            strategy['fuel_saved'].iloc[0]
        ]
        colors = ['red', 'green', 'blue']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Fuel Consumption')
        ax.set_title('Fuel Optimization Results')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}',
                   ha='center', va='bottom')
    
    plt.tight_layout()
    output_path = Path(output_dir) / "optimization_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved optimization plot to {output_path}")


def plot_feature_importance(model_path: str, output_dir: str = "outputs"):
    """
    Plot feature importance from trained model.
    
    Args:
        model_path: Path to trained model
        output_dir: Directory to save plots
    """
    model_data = joblib.load(model_path)
    
    if isinstance(model_data, dict) and 'feature_importance' in model_data:
        importance_df = model_data['feature_importance']
    else:
        print("Feature importance not available in model file")
        return
    
    # Plot top 15 features
    top_n = min(15, len(importance_df))
    top_features = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_features['importance'], alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importances')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_path = Path(output_dir) / "feature_importance.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved feature importance plot to {output_path}")


def plot_lap_by_lap(lap_results_path: str, output_dir: str = "outputs"):
    """
    Plot lap-by-lap optimization results.
    
    Args:
        lap_results_path: Path to lap-by-lap results CSV
        output_dir: Directory to save plots
    """
    df = pd.read_csv(lap_results_path)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Fuel comparison
    axes[0].plot(df['lap'], df['baseline_fuel'], 'o-', label='Baseline', alpha=0.7)
    axes[0].plot(df['lap'], df['optimized_fuel'], 's-', label='Optimized', alpha=0.7)
    axes[0].fill_between(df['lap'], df['baseline_fuel'], df['optimized_fuel'], 
                         alpha=0.3, label='Savings')
    axes[0].set_xlabel('Lap Number')
    axes[0].set_ylabel('Fuel Consumption')
    axes[0].set_title('Lap-by-Lap Fuel Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Optimal parameters
    ax2 = axes[1].twinx()
    
    line1 = axes[1].plot(df['lap'], df['optimal_throttle_scale'], 
                         'b-', marker='o', label='Throttle Scale')
    line2 = ax2.plot(df['lap'], df['optimal_ers_shift'], 
                     'r-', marker='s', label='ERS Shift')
    
    axes[1].set_xlabel('Lap Number')
    axes[1].set_ylabel('Throttle Scale', color='b')
    ax2.set_ylabel('ERS Shift', color='r')
    axes[1].tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    axes[1].set_title('Optimal Parameters per Lap')
    axes[1].grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    axes[1].legend(lines, labels, loc='best')
    
    plt.tight_layout()
    output_path = Path(output_dir) / "lap_by_lap_optimization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved lap-by-lap plot to {output_path}")


def create_all_plots(output_dir: str = "outputs"):
    """
    Create all available plots based on existing output files.
    
    Args:
        output_dir: Directory containing output files
    """
    print("\n" + "="*60)
    print("CREATING VISUALIZATION PLOTS")
    print("="*60 + "\n")
    
    output_path = Path(output_dir)
    
    # Plot predictions
    pred_files = [
        output_path / "test_predictions.csv",
        output_path / "test_preds_holdout.csv",
        output_path / "test_preds.csv"
    ]
    
    for pred_file in pred_files:
        if pred_file.exists():
            print(f"Creating prediction plots from {pred_file.name}...")
            plot_predictions(str(pred_file), output_dir)
            break
    
    # Plot optimization results
    strategy_file = output_path / "optimized_strategy.csv"
    all_configs_file = output_path / "optimized_strategy_all_configs.csv"
    
    if strategy_file.exists():
        print(f"Creating optimization plots...")
        plot_optimization_results(
            str(strategy_file),
            str(all_configs_file) if all_configs_file.exists() else None,
            output_dir
        )
    
    # Plot feature importance
    model_files = [
        output_path / "fuel_model.pkl",
        output_path / "fuel_model_real.pkl"
    ]
    
    for model_file in model_files:
        if model_file.exists():
            print(f"Creating feature importance plot from {model_file.name}...")
            plot_feature_importance(str(model_file), output_dir)
            break
    
    # Plot lap-by-lap results
    lap_file = output_path / "optimized_strategy_lap_by_lap.csv"
    if lap_file.exists():
        print(f"Creating lap-by-lap plots...")
        plot_lap_by_lap(str(lap_file), output_dir)
    
    print("\n" + "="*60)
    print("PLOTTING COMPLETE")
    print("="*60 + "\n")


def main():
    """Main plotting workflow."""
    parser = argparse.ArgumentParser(description="Create visualization plots for AMi-Fuel")
    parser.add_argument("--output-dir", default="outputs",
                       help="Directory containing output files")
    parser.add_argument("--predictions", default=None,
                       help="Path to predictions CSV (optional)")
    parser.add_argument("--strategy", default=None,
                       help="Path to optimization strategy CSV (optional)")
    parser.add_argument("--model", default=None,
                       help="Path to trained model (optional)")
    
    args = parser.parse_args()
    
    if args.predictions or args.strategy or args.model:
        # Plot specific files
        if args.predictions:
            plot_predictions(args.predictions, args.output_dir)
        
        if args.strategy:
            plot_optimization_results(args.strategy, None, args.output_dir)
        
        if args.model:
            plot_feature_importance(args.model, args.output_dir)
    else:
        # Create all available plots
        create_all_plots(args.output_dir)


if __name__ == "__main__":
    main()

