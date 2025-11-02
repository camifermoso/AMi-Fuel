# AMi-Fuel Quick Start Guide

## 🚀 Get Started in 5 Minutes

This guide will help you run the complete AMi-Fuel pipeline from scratch.

## Prerequisites

- Python 3.8 or higher
- 5-10 GB free disk space (for F1 data cache)
- Internet connection (for downloading F1 data)

## Installation

```bash
# 1. Navigate to project directory
cd AMi-Fuel

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt
```

## Quick Test (30 seconds)

Verify the preprocessing pipeline works:

```bash
python scripts/test_preprocessing.py
```

You should see:
```
✓ ALL TESTS PASSED

The preprocessing pipeline is working correctly!
```

## Run Complete Pipeline (5-10 minutes)

### Option 1: Use Existing Data (Fast)

If you already have the training data files:

```bash
python scripts/build_proxy_and_train.py
```

This will:
1. ✅ Clean and preprocess the data
2. ✅ Train a Random Forest model
3. ✅ Evaluate on held-out test set
4. ✅ Save all outputs

### Option 2: Fetch Fresh F1 Data (Slower, ~30 minutes)

Download real F1 telemetry first:

```bash
# This downloads data from 2022 and 2023 races
python scripts/fetch_fastf1_highfuel.py

# Then run the pipeline
python scripts/build_proxy_and_train.py
```

## View Results

### 1. Check Model Performance

After training completes, check the console output for:
- Training R² Score (typically 0.90-0.95)
- Test R² Score (typically 0.85-0.95)
- MAE (typically 0.02-0.04)

### 2. Inspect Output Files

```bash
# Preprocessed data
ls -lh data/processed/

# Model and predictions
ls -lh outputs/

# You should see:
# - train_processed.csv
# - test_processed.csv
# - preprocessor.pkl
# - fuel_model_real.pkl
# - test_predictions.csv
# - test_preds_holdout.csv
```

### 3. Optimize Fuel Strategy

```bash
python src/optimizer.py \
  --data data/processed/train_processed.csv \
  --model outputs/fuel_model_real.pkl \
  --out outputs/optimized_strategy.csv
```

Check the results:
```bash
cat outputs/optimized_strategy.csv
```

### 4. Create Visualizations

```bash
python scripts/plot_results.py
```

View the generated plots in `outputs/`:
- `prediction_accuracy.png` - Model performance
- `optimization_results.png` - Fuel savings
- `feature_importance.png` - Important features

## Understanding the Output

### Model Metrics

**R² Score**: Measures how well the model predicts fuel consumption
- 1.0 = Perfect prediction
- 0.9+ = Excellent
- 0.8-0.9 = Good
- <0.8 = Needs improvement

**MAE (Mean Absolute Error)**: Average prediction error
- Lower is better
- Typical range: 0.02-0.04

**RMSE (Root Mean Squared Error)**: Penalizes large errors more
- Lower is better
- Similar to MAE but emphasizes outliers

### Optimization Results

**Fuel Saved**: Difference between baseline and optimized strategy
- Typical savings: 2-5%
- Example: If baseline = 1.50, optimized = 1.47, saved = 0.03 (2%)

**Throttle Scale**: Recommended throttle adjustment
- 0.95 = Reduce throttle by 5%
- 0.98 = Reduce throttle by 2%

**ERS Shift**: Recommended ERS deployment change
- +0.1 = Deploy more ERS
- -0.05 = Deploy less ERS

## Troubleshooting

### Issue: Import errors

**Error**: `ModuleNotFoundError: No module named 'fastf1'`

**Solution**:
```bash
pip install -r requirements.txt
```

### Issue: No data files

**Error**: `FileNotFoundError: data/train_highfuel.csv not found`

**Solution**:
```bash
# Option 1: Fetch F1 data
python scripts/fetch_fastf1_highfuel.py

# Option 2: Use synthetic data for testing
python scripts/generate_synth_data.py
python src/fuel_model.py --data data/synth_telemetry.csv
```

### Issue: Low model performance

**Problem**: R² score < 0.8

**Solutions**:
1. Check data quality: `python scripts/test_preprocessing.py`
2. Try different model: `--model-type gradient_boosting`
3. Increase data: Fetch more races
4. Adjust preprocessing: Check `data/processed/` files

### Issue: No fuel savings found

**Problem**: Optimization finds 0% savings

**Solutions**:
1. Widen search ranges in optimizer
2. Check if data is already optimized
3. Try lap-by-lap mode: `--lap-by-lap`

## Advanced Usage

### Use Different Model Type

```bash
python src/fuel_model.py \
  --data data/processed/train_processed.csv \
  --model-type gradient_boosting \
  --save-model outputs/fuel_model_gb.pkl
```

### Lap-by-Lap Optimization

```bash
python src/optimizer.py \
  --data data/processed/train_processed.csv \
  --model outputs/fuel_model_real.pkl \
  --lap-by-lap \
  --out outputs/lap_optimization.csv
```

### Generate Synthetic Data

For quick testing without real F1 data:

```bash
python scripts/generate_synth_data.py --laps 60 --out data/test_synth.csv
```

## Project Structure

```
AMi-Fuel/
├── data/
│   ├── train_highfuel.csv          # Raw training data
│   ├── test_highfuel.csv           # Raw test data
│   └── processed/                  # Preprocessed data
│       ├── train_processed.csv
│       ├── test_processed.csv
│       └── preprocessor.pkl
├── outputs/
│   ├── fuel_model_real.pkl         # Trained model
│   ├── test_predictions.csv        # Predictions
│   ├── optimized_strategy.csv      # Optimal strategy
│   └── *.png                       # Visualization plots
├── src/
│   ├── data_preprocessing.py       # Preprocessing pipeline
│   ├── fuel_model.py               # Model training
│   └── optimizer.py                # Optimization engine
└── scripts/
    ├── build_proxy_and_train.py    # Complete pipeline
    ├── fetch_fastf1_highfuel.py    # Download F1 data
    ├── test_preprocessing.py       # Test suite
    └── plot_results.py             # Visualizations
```

## Next Steps

1. ✅ Run the test suite
2. ✅ Execute the complete pipeline
3. ✅ Optimize fuel strategy
4. ✅ Create visualizations
5. ✅ Analyze results

For detailed documentation, see:
- `README.md` - Complete project documentation
- `IMPLEMENTATION_SUMMARY.md` - Technical details

## Getting Help

If you encounter issues:
1. Check this guide's troubleshooting section
2. Review `README.md` for detailed usage
3. Run the test suite: `python scripts/test_preprocessing.py`
4. Check data files exist in `data/` directory
5. Verify model files in `outputs/` directory

## Success Criteria

You'll know everything is working when:
- ✅ Test suite passes (all 7 tests)
- ✅ Model R² score > 0.85
- ✅ Optimization finds 2-5% fuel savings
- ✅ Visualizations are generated
- ✅ All output files are created

Happy optimizing! 🏎️💨
