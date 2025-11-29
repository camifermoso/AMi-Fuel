#!/bin/bash

# AMi-Fuel Model Improvement - Quick Setup Script
# This script sets up and runs the improved model training

set -e  # Exit on error

echo ""
echo "========================================"
echo "   AMi-Fuel Model Improvement Setup"
echo "========================================"
echo ""

# Check Python version
echo "1. Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "   ✓ Python $python_version detected"
echo ""

# Install dependencies
echo "2. Installing required packages..."
echo "   This may take a few minutes..."
pip install xgboost lightgbm scipy --quiet || {
    echo "   ⚠️  pip install failed, trying with --user flag..."
    pip install xgboost lightgbm scipy --user --quiet
}
echo "   ✓ Dependencies installed"
echo ""

# Check if data exists
echo "3. Checking for training data..."
if [ ! -f "data/train_highfuel.csv" ]; then
    echo "   ⚠️  Training data not found at data/train_highfuel.csv"
    echo "   Please ensure your data files are in the correct location"
    exit 1
fi
if [ ! -f "data/test_highfuel.csv" ]; then
    echo "   ⚠️  Test data not found at data/test_highfuel.csv"
    echo "   Please ensure your data files are in the correct location"
    exit 1
fi
echo "   ✓ Data files found"
echo ""

# Train baseline for comparison (if not exists)
echo "4. Training models..."
echo ""

if [ ! -f "outputs/test_preds.csv" ]; then
    echo "   Training baseline model for comparison..."
    python scripts/build_proxy_and_train.py
    echo ""
fi

# Train improved model
echo "   Training improved XGBoost model..."
python scripts/train_improved_model.py --model xgboost --tune

echo ""
echo "5. Comparing models..."
python scripts/compare_models.py

echo ""
echo "========================================"
echo "           Setup Complete! ✓"
echo "========================================"
echo ""
echo "Your improved model has been trained!"
echo ""
echo "Check these files for results:"
echo "  📊 outputs/metrics_summary.txt - Performance metrics"
echo "  📈 outputs/model_comparison.png - Visual comparison"
echo "  🎯 outputs/feature_importance.csv - Most important features"
echo "  📁 outputs/fuel_model_xgboost_enhanced.pkl - Trained model"
echo ""
echo "Next steps:"
echo "  1. Review MODEL_IMPROVEMENT_GUIDE.md for details"
echo "  2. Try other models: python scripts/train_improved_model.py --model lightgbm"
echo "  3. Use stacking: python scripts/train_improved_model.py --stacking"
echo ""
