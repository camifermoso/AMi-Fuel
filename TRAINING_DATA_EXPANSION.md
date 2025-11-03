# Training Data Expansion Strategy

## Current State (Baseline)
- **Years**: 2022 (train), 2023 (test)
- **Circuits**: 5 high-fuel circuits only
- **Total samples**: ~6,000 laps
- **Model performance**: 89% R², 99% prediction accuracy

## Proposed Expansion

### Phase 1: High-Fuel Circuits (Multi-Year)
Train on **7 years of data (2018-2024)** for the fuel-intensive circuits:

1. **Singapore** - Street circuit, high fuel consumption
2. **Barcelona (Spain)** - Technical track, high fuel load
3. **Bahrain** - Desert heat, fuel-critical
4. **Montreal (Canada)** - Stop-and-go, heavy braking
5. **Suzuka (Japan)** - High-speed, fuel-intensive

**Rationale**: 
- Capture year-to-year regulation changes (2018-2021 vs 2022+ ground effect cars)
- Learn different tire compounds and fuel strategies
- Account for weather variations across years

### Phase 2: Additional Circuits (Sample Years)
Include **10 more circuits** with 2-3 random years each:

- Austria, Silverstone, Monza, Spa (high-speed)
- Abu Dhabi, Mexico, Brazil (varied conditions)
- Hungary, Monaco, Austin (technical/street)

**Rationale**:
- Add variety without overweighting non-critical circuits
- Model learns general fuel consumption patterns
- Better generalization to unseen circuits

## Expected Benefits

### 1. Improved Generalization
- **Cross-year patterns**: Model learns what's consistent vs what changes
- **Regulation adaptation**: Handles 2022 car regulation changes naturally
- **Weather robustness**: Different conditions across years

### 2. Better Predictions
- **Expanded feature space**: More diverse telemetry patterns
- **Reduced overfitting**: Larger, more varied dataset
- **Circuit-specific learning**: Model understands each track's fuel characteristics

### 3. Practical Advantages
- **New circuit predictions**: Can estimate fuel for circuits not in training set
- **Year-over-year comparison**: "How does 2024 compare to 2020?"
- **Strategy evolution**: Track how fuel strategies change over time

## Expected Dataset Size

### Training Data Estimates:
- **High-fuel circuits**: 5 circuits × 7 years × ~1,000 laps = ~35,000 laps
- **Additional circuits**: 10 circuits × 3 years × ~800 laps = ~24,000 laps
- **Total**: ~50,000-60,000 training samples (10x current size!)

### Test Data:
- 15% stratified split = ~8,000-9,000 test samples
- Ensures all circuits represented in test set

## Implementation

### Step 1: Fetch Expanded Data
```bash
python scripts/fetch_expanded_training_data.py
```

**Time estimate**: 2-4 hours (downloads and caches race data)

**Output**:
- `data/train_highfuel_expanded.csv` (~50,000+ laps)
- `data/test_highfuel_expanded.csv` (~8,000+ laps)

### Step 2: Update Training Pipeline
Update `scripts/train_improved_model.py` to use:
- `train_highfuel_expanded.csv` instead of `train_highfuel.csv`
- May need to adjust hyperparameters for larger dataset

### Step 3: Retrain Models
```bash
python scripts/train_improved_model.py
```

**Expected improvements**:
- R² may stay similar or slightly improve (already at 89%)
- **MAPE should decrease** (better accuracy)
- **Generalization** to 2024/2025 races will be stronger

## Risks & Mitigation

### Risk 1: Regulation Changes
**Issue**: 2022 introduced ground-effect cars (major change)  
**Mitigation**: Model can learn this as a feature; year is included in data

### Risk 2: Longer Training Time
**Issue**: 10x more data = slower training  
**Mitigation**: XGBoost/LightGBM are fast; use early stopping

### Risk 3: Data Quality Variations
**Issue**: Older years may have missing telemetry (especially ERS data)  
**Mitigation**: Already handle missing ERS data gracefully

## Success Metrics

After retraining with expanded data, measure:

1. **Test set performance**: R², MAE, MAPE on 2024 data
2. **Cross-year consistency**: Similar accuracy across 2018-2024
3. **New circuit prediction**: Test on a completely unseen circuit
4. **Feature importance stability**: Same features still important?

## Timeline

1. **Data Fetching**: 2-4 hours (run overnight)
2. **Code Updates**: 30 minutes
3. **Model Training**: 15-30 minutes (larger dataset)
4. **Validation**: 1 hour (compare old vs new)

**Total**: ~1 day of work for significantly better model

---

## Quick Start

To fetch the expanded data now:

```bash
# This will take 2-4 hours - run overnight
python scripts/fetch_expanded_training_data.py
```

The script will:
- ✓ Download 7 years of data for high-fuel circuits (2018-2024)
- ✓ Sample 3 years for 10 additional circuits
- ✓ Create stratified train/test split (85/15)
- ✓ Save to `train_highfuel_expanded.csv` and `test_highfuel_expanded.csv`
- ✓ Show detailed statistics and breakdowns

Ready to run when you are! 🚀
