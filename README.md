
# AMi-Fuel
**Aston Martin Intelligent Fuel (AMi-Fuel)**  
Streamlit dashboard and ML stack for Aston Martin F1 fuel intelligence.

## Description
AMi-Fuel blends a calibrated tree-based model with an engineer-focused UI. The model learns how RPM, throttle, ERS/DRS usage, gear, speed, and weather shape fuel burn, then maps a normalized `fuel_proxy` to kg/lap with circuit baselines. The dashboard exposes:
- **Fuel Strategy Simulator** (Live Calculator): per-lap fuel burn gauge, setup levers, and top-performer cards keyed to race context.
- **Race Fuel Debrief**: lap-by-lap fuel cost from FastF1 sessions with telemetry or fallbacks, total/avg fuel, and variation across laps.
- **AI Model Briefing**: training coverage (2018-2024, 676k laps), Aston Martin subset, circuit coverage image, and live data summaries when training data is available.

### Key Features
- **Live dashboard**: Streamlit app with always-visible sidebar and custom Aston Martin theming.
- **Calibrated ML**: Gradient-boosted trees with calibration and circuit baselines to convert `fuel_proxy` → kg/lap.
- **Fuel debref**: Per-lap fuel estimation from race telemetry; totals normalized to realistic race ranges (100–105 kg, Monaco 95–100 kg).
- **Data pipelines**: Preprocessing, feature engineering, and training scripts for fuel modeling (real + synthetic telemetry).
- **FastF1 integration**: Pulls and caches race sessions for Aston Martin drivers (2021–2024).

## Project Structure
```
AMi-Fuel/
├── src/
│   ├── data_preprocessing.py    # Data cleaning, aggregation, normalization
│   ├── fuel_model.py            # Model training and evaluation
│   └── optimizer.py             # Fuel optimization strategies
├── scripts/
│   ├── fetch_fastf1_highfuel.py # Download real F1 telemetry
│   ├── generate_synth_data.py   # Generate synthetic training data
│   ├── build_proxy_and_train.py # Complete preprocessing + training pipeline
│   └── plot_results.py          # Visualization utilities
├── data/
│   ├── train_highfuel.csv       # Raw training data
│   ├── test_highfuel.csv        # Raw test data
│   └── processed/               # Preprocessed data (generated)
├── outputs/                     # Models, predictions, strategies
├── cache/                       # FastF1 cache directory
└── requirements.txt
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip

### Setup
```bash
# Clone the repository
git clone https://github.com/camifermoso/AMi-Fuel.git
cd AMi-Fuel

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run the dashboard (Streamlit)
```bash
streamlit run app.py
```
Use the sidebar to switch between Fuel Strategy Simulator, Race Fuel Debrief, and AI Model Briefing.

### Model pipeline (optional CLI)
- `scripts/build_proxy_and_train.py`: end-to-end preprocessing + training
- `src/data_preprocessing.py`: clean/aggregate/normalize telemetry
- `src/fuel_model.py`: train (random_forest/gradient_boosting) and evaluate
- `scripts/fetch_fastf1_highfuel.py`: pull FastF1 telemetry
- `scripts/generate_synth_data.py`: synthetic laps for testing

## Data Preprocessing Details

### Cleaning
- **Duplicates**: Removed exact duplicate rows
- **Missing Values**: 
  - Critical features (RPM, throttle, speed): rows dropped
  - Non-critical (gear, DRS, ERS): imputed with median or 0
- **Outliers**: Removed using both domain knowledge and IQR method
  - RPM: 5,000-14,000 range
  - Throttle: 0-100%
  - Speed: 50-350 km/h
  - Gear: 1-8
- **Data Quality**: Only green/yellow flag laps retained

### Aggregation
Multiple telemetry samples per lap are aggregated:
- **Telemetry features**: Mean aggregation
- **Speed traps**: Max value (optimal conditions)
- **Metadata**: First value (should be consistent)
- **Variability**: Standard deviation tracked for quality assessment

### Feature Engineering
Physics-inspired derived features:
- `power_estimate`: (RPM/12000) × (Throttle/100)
- `speed_per_rpm`: Speed / RPM (efficiency)
- `drs_speed_factor`: DRS × Speed (aero efficiency)
- `energy_intensity`: (ERS/5) × (Throttle/100)
- `speed_per_gear`: Speed / Gear (corner vs straight)
- `speed_variance`: Variance across speed traps
- `avg_sector_speed`: Mean of speed trap measurements

### Normalization
RobustScaler used (better for data with outliers):
- Scales features based on median and IQR
- More robust than StandardScaler for motorsport data
- Fitted on training data, applied to test data

### Fuel Proxy Target
Physics-inspired weighted combination:
```
fuel_burn_proxy = 0.48 × RPM + 0.32 × Throttle + 0.20 × ERS
```
Weights based on:
- RPM (48%): Primary fuel consumer
- Throttle (32%): Direct fuel injection control
- ERS (20%): Energy deployment efficiency

## Model Performance

### Typical Metrics (from training runs)
- **R² Score**: ~0.85-0.95
- **MAE**: ~0.02-0.04
- **Cross-validation**: Consistent across folds

### Feature Importance
Top features (typical):
1. `avg_rpm`: 25-30%
2. `avg_throttle`: 20-25%
3. `power_estimate`: 15-20%
4. `avg_speed`: 10-15%
5. `energy_intensity`: 8-12%

## Optimization Results

### Typical Fuel Savings
- **Throttle reduction**: 1-3%
- **ERS optimization**: 0.5-1.5%
- **Total savings**: 2-5% without significant lap time penalty

### Strategy Recommendations
- Slight throttle reduction in straights
- Optimized ERS deployment in acceleration zones
- Lap-by-lap adaptation based on traffic/conditions

## Outputs

### Generated Files
1. **data/processed/train_processed.csv**: Cleaned & normalized training data
2. **data/processed/test_processed.csv**: Cleaned & normalized test data
3. **data/processed/preprocessor.pkl**: Scaler state (for new data)
4. **outputs/fuel_model.pkl**: Trained model
5. **outputs/test_predictions.csv**: Model predictions on test set
6. **outputs/optimized_strategy.csv**: Optimal strategy parameters
7. **outputs/test_preds_holdout.csv**: Held-out year/circuit predictions

## Advanced Usage

### Custom Model Training
```python
from src.fuel_model import FuelModel
import pandas as pd

# Load preprocessed data
data = pd.read_csv("data/processed/train_processed.csv")

# Define features
features = ['avg_rpm', 'avg_throttle', 'avg_speed', 
            'avg_gear', 'avg_drs', 'avg_ers_mode']

X = data[features]
y = data['fuel_burn_proxy']

# Train model
model = FuelModel(model_type='gradient_boosting')
metrics = model.train(X, y, verbose=True)
model.save("outputs/custom_model.pkl")
```

### Custom Optimization
```python
from src.optimizer import FuelOptimizer

optimizer = FuelOptimizer(
    model_path="outputs/fuel_model.pkl",
    preprocessor_path="data/processed/preprocessor.pkl"
)

# Load telemetry
data = pd.read_csv("data/processed/train_processed.csv")
features = ['avg_rpm', 'avg_throttle', 'avg_speed', 
            'avg_gear', 'avg_drs', 'avg_ers_mode']

# Optimize
best_strategy, all_results = optimizer.optimize_strategy(
    data, features, 
    throttle_range=(0.90, 0.999),
    ers_range=(-0.1, 0.15),
    verbose=True
)
```

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'fastf1'`  
**Solution**: Install FastF1: `pip install fastf1`

**Issue**: Preprocessing fails with outlier warnings  
**Solution**: Check raw data quality, adjust outlier thresholds in `data_preprocessing.py`

**Issue**: Model performance is poor  
**Solution**: 
- Ensure data is preprocessed (normalized)
- Check for data leakage
- Try different model types
- Increase training data

**Issue**: Optimization finds no fuel savings  
**Solution**: 
- Widen search ranges
- Increase grid resolution
- Check if data is already optimized

## Contributing
Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License
MIT License

## Acknowledgments
- FastF1 library for F1 telemetry data access
- Scikit-learn for machine learning tools
- Formula 1 for inspiring this project

## Contact
For questions or support, please open an issue on GitHub.

---
**AMi-Fuel**: Data-driven fuel optimization for the pinnacle of motorsport. 🏎️💨
