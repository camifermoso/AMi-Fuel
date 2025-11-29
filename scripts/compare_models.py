"""
Model Comparison Script
Compare performance of different models and configurations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def compare_predictions(prediction_files, labels=None, save_path="outputs/model_comparison.png"):
    """
    Compare predictions from multiple models.
    
    Args:
        prediction_files: List of paths to prediction CSV files
        labels: List of labels for each model
        save_path: Path to save comparison plot
    """
    if labels is None:
        labels = [f"Model {i+1}" for i in range(len(prediction_files))]
    
    results = []
    
    print("\n" + "="*70)
    print(" "*20 + "MODEL COMPARISON RESULTS")
    print("="*70 + "\n")
    
    for file, label in zip(prediction_files, labels):
        if not Path(file).exists():
            print(f"⚠️  File not found: {file}")
            continue
        
        df = pd.read_csv(file)
        
        if 'y_true' not in df.columns or 'y_pred' not in df.columns:
            print(f"⚠️  Missing required columns in {file}")
            continue
        
        y_true = df['y_true'].values
        y_pred = df['y_pred'].values
        
        # Calculate metrics
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        results.append({
            'Model': label,
            'R² Score': r2,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE (%)': mape,
            'y_true': y_true,
            'y_pred': y_pred
        })
    
    if not results:
        print("❌ No valid predictions to compare")
        return
    
    # Print comparison table
    print(f"{'Model':<30} {'R² Score':<12} {'MAE':<12} {'RMSE':<12} {'MAPE (%)':<12}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['Model']:<30} "
              f"{result['R² Score']:<12.4f} "
              f"{result['MAE']:<12.4f} "
              f"{result['RMSE']:<12.4f} "
              f"{result['MAPE (%)']:<12.2f}")
    
    print("\n" + "="*70)
    
    # Find best model
    best_r2 = max(results, key=lambda x: x['R² Score'])
    best_mae = min(results, key=lambda x: x['MAE'])
    
    print("\n🏆 BEST MODELS:")
    print(f"  Highest R² Score: {best_r2['Model']} ({best_r2['R² Score']:.4f})")
    print(f"  Lowest MAE: {best_mae['Model']} ({best_mae['MAE']:.4f})")
    
    # Calculate improvements
    if len(results) >= 2:
        baseline = results[0]  # Assume first is baseline
        print("\n📈 IMPROVEMENTS vs BASELINE:")
        for result in results[1:]:
            r2_improvement = ((result['R² Score'] - baseline['R² Score']) / baseline['R² Score']) * 100
            mae_improvement = ((baseline['MAE'] - result['MAE']) / baseline['MAE']) * 100
            
            print(f"\n  {result['Model']}:")
            print(f"    R² Score: {r2_improvement:+.2f}%")
            print(f"    MAE: {mae_improvement:+.2f}%")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Metrics comparison bar chart
    ax1 = axes[0, 0]
    metrics_df = pd.DataFrame([
        {
            'Model': r['Model'],
            'R² Score': r['R² Score'],
            'MAE': r['MAE'],
            'RMSE': r['RMSE']
        }
        for r in results
    ])
    
    metrics_df_normalized = metrics_df.copy()
    metrics_df_normalized['MAE'] = 1 - metrics_df['MAE']  # Invert for better visualization
    metrics_df_normalized['RMSE'] = 1 - metrics_df['RMSE']
    
    x = np.arange(len(results))
    width = 0.25
    
    ax1.bar(x - width, metrics_df['R² Score'], width, label='R² Score', alpha=0.8)
    ax1.bar(x, 1 - metrics_df['MAE'], width, label='1 - MAE', alpha=0.8)
    ax1.bar(x + width, 1 - metrics_df['RMSE'], width, label='1 - RMSE', alpha=0.8)
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Score (Higher is Better)')
    ax1.set_title('Model Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([r['Model'] for r in results], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Prediction scatter plots
    ax2 = axes[0, 1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for result, color in zip(results, colors):
        ax2.scatter(result['y_true'], result['y_pred'], alpha=0.5, 
                   label=result['Model'], s=10, c=[color])
    
    # Perfect prediction line
    min_val = min([min(r['y_true']) for r in results])
    max_val = max([max(r['y_true']) for r in results])
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax2.set_xlabel('True Fuel Consumption')
    ax2.set_ylabel('Predicted Fuel Consumption')
    ax2.set_title('Predictions vs Actual')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Residual distributions
    ax3 = axes[1, 0]
    
    for result, color in zip(results, colors):
        residuals = result['y_true'] - result['y_pred']
        ax3.hist(residuals, bins=30, alpha=0.5, label=result['Model'], 
                color=color, density=True)
    
    ax3.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
    ax3.set_xlabel('Residual (True - Predicted)')
    ax3.set_ylabel('Density')
    ax3.set_title('Residual Distribution')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Error metrics comparison
    ax4 = axes[1, 1]
    
    metrics_comparison = pd.DataFrame([
        {
            'Model': r['Model'],
            'R²': r['R² Score'],
            'MAE': r['MAE'],
            'RMSE': r['RMSE'],
            'MAPE': r['MAPE (%)']
        }
        for r in results
    ])
    
    # Create table
    table_data = []
    for _, row in metrics_comparison.iterrows():
        table_data.append([
            row['Model'],
            f"{row['R²']:.4f}",
            f"{row['MAE']:.4f}",
            f"{row['RMSE']:.4f}",
            f"{row['MAPE']:.2f}%"
        ])
    
    table = ax4.table(
        cellText=table_data,
        colLabels=['Model', 'R²', 'MAE', 'RMSE', 'MAPE'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax4.axis('off')
    ax4.set_title('Detailed Metrics Comparison', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to: {save_path}")
    print()
    
    return results


def quick_compare():
    """Quick comparison of existing models."""
    
    print("\n🔍 Searching for prediction files...")
    
    prediction_files = []
    labels = []
    
    # Check for common prediction files
    possible_files = [
        ("outputs/test_preds.csv", "Baseline Model"),
        ("outputs/test_predictions.csv", "Random Forest"),
        ("outputs/test_predictions_enhanced.csv", "Enhanced Model"),
        ("outputs/validation_predictions.csv", "Validation Set"),
    ]
    
    for file, label in possible_files:
        if Path(file).exists():
            prediction_files.append(file)
            labels.append(label)
            print(f"  ✓ Found: {file}")
    
    if not prediction_files:
        print("\n⚠️  No prediction files found. Please train a model first.")
        print("\nRun one of these commands:")
        print("  python scripts/build_proxy_and_train.py")
        print("  python scripts/train_improved_model.py")
        return
    
    print(f"\n📊 Comparing {len(prediction_files)} model(s)...\n")
    
    compare_predictions(prediction_files, labels)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Custom comparison
        files = sys.argv[1:]
        compare_predictions(files)
    else:
        # Quick comparison of existing models
        quick_compare()
