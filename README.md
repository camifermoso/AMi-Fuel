
# AMi-Fuel
**Aston Martin Intelligent Fuel**  
Machine learning for predictive fuel optimization in Formula 1 racing.

## Description
**AMi-Fuel** is a comprehensive machine-learning-driven system that models and optimizes race-fuel consumption using real and synthetic telemetry data. By learning how driving parameters (RPM, throttle, ERS deployment, speed, gear, DRS) affect fuel flow, AMi-Fuel helps engineers balance performance and efficiency without altering car design or fuel chemistry.

### Key Features
- **Data Preprocessing Pipeline**: Comprehensive cleaning, aggregation, and normalization of F1 telemetry data
- **Feature Engineering**: Creates physics-inspired features (power estimate, efficiency metrics, energy intensity)
- **Ensemble ML Models**: Random Forest and Gradient Boosting regressors for accurate fuel prediction
- **Optimization Engine**: Grid search and lap-by-lap optimization to minimize fuel consumption
- **Real F1 Data**: Integrates with FastF1 to fetch and analyze real race telemetry
- **Validation Framework**: Cross-validation and held-out test sets for robust evaluation

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

### Complete Pipeline (Recommended)
Run the full preprocessing and training pipeline:
```bash
python scripts/build_proxy_and_train.py
```

This will:
1. Clean and preprocess the raw telemetry data
2. Aggregate multiple telemetry samples per lap
3. Engineer additional features
4. Normalize all features
5. Train a Random Forest fuel consumption model
6. Evaluate on held-out test set
7. Save all outputs

### Step-by-Step Workflow

#### 1. Fetch Real F1 Data (Optional)
Download telemetry from specific circuits (Singapore, Spain, Canada, etc.):
```bash
python scripts/fetch_fastf1_highfuel.py
```

#### 2. Preprocess Data
Clean, aggregate, and normalize the raw telemetry:
```bash
python src/data_preprocessing.py \
  --train data/train_highfuel.csv \
  --test data/test_highfuel.csv \
  --output-dir data/processed \
  --scaler robust
```

**Preprocessing Steps:**
- Remove duplicates and outliers
- Handle missing values (drop critical, impute non-critical)
- Aggregate multiple telemetry samples per lap
- Engineer features (power estimate, efficiency metrics)
- Normalize using RobustScaler (handles outliers better)
- Create fuel consumption proxy target

#### 3. Train Model
Train the fuel consumption model:
```bash
python src/fuel_model.py \
  --data data/processed/train_processed.csv \
  --model-type random_forest \
  --test-size 0.25 \
  --cv 5 \
  --save-model outputs/fuel_model.pkl
```

**Model Options:**
- `random_forest`: Ensemble of decision trees (default, robust)
- `gradient_boosting`: Sequential boosting (often more accurate)

#### 4. Optimize Strategy
Find optimal throttle and ERS deployment:
```bash
python src/optimizer.py \
  --data data/processed/train_processed.csv \
  --model outputs/fuel_model.pkl \
  --out outputs/optimized_strategy.csv
```

**Optimization Options:**
- Global optimization: Find single best strategy across all laps
- Lap-by-lap: Optimize each lap individually (use `--lap-by-lap` flag)

#### 5. Generate Synthetic Data (for testing)
```bash
python scripts/generate_synth_data.py \
  --laps 50 \
  --seed 42 \
  --out data/synth_telemetry.csv
```

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

### Typical Metrics
- **R² Score**: 0.85-0.95 (excellent predictive power)
- **MAE**: 0.02-0.04 (low average error)
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
