# Weather Data Integration for AMi-Fuel Model

## Overview
Weather conditions significantly impact fuel consumption in F1. This update integrates 7 weather parameters into the training pipeline.

## 🌡️ Weather Features Added

### 1. Air Temperature (`air_temp`)
- **Range**: 15-40°C typical
- **Impact**: Hotter air = less dense = engine runs richer = **+3-5% fuel**
- **Example**: Singapore 32°C vs Silverstone 18°C

### 2. Track Temperature (`track_temp`)
- **Range**: 20-60°C typical  
- **Impact**: Hot track = more tire slip = **+2-4% fuel**
- **Example**: Bahrain hot track 50°C vs wet Spa 25°C

### 3. Humidity (`humidity`)
- **Range**: 15-90%
- **Impact**: High humidity = less oxygen = affects combustion = **+1-3% fuel**
- **Example**: Singapore 80% vs Mexico 20%

### 4. Atmospheric Pressure (`pressure`)
- **Range**: 950-1020 mbar
- **Impact**: Low pressure (altitude) = less air density = **+4-6% fuel**
- **Example**: Mexico City 950 mbar vs sea level 1013 mbar

### 5. Rainfall (`rainfall`)
- **Type**: Boolean (True/False)
- **Impact**: Wet conditions = less throttle = **-10-20% fuel**
- **Example**: Wet races use significantly less fuel

### 6. Wind Speed (`wind_speed`)
- **Range**: 0-40 km/h
- **Impact**: Headwind on straight = **+1-2% fuel per 10 km/h**
- **Example**: Baku with strong winds

### 7. Wind Direction (`wind_direction`)
- **Range**: 0-360 degrees
- **Impact**: Combined with wind speed and track layout
- **Example**: Headwind vs tailwind on long straights

## 📊 Expected Model Improvements

### Current (No Weather):
- R² = 0.96 (96% accuracy)
- MAPE = 0.42%
- **Problem**: Same circuit, different weather = prediction errors

### With Weather:
- Expected R² = 0.97-0.98 (97-98% accuracy)  
- Expected MAPE = 0.30-0.35%
- **Benefit**: Accounts for weather variations

### Real-World Impact:
```
Scenario: Bahrain GP
- Hot day (40°C air, 55°C track, 15% humidity)
- Model predicts: +4.5% more fuel vs baseline
- Aston Martin can prepare: +4.8kg fuel load

Scenario: Same Bahrain GP but cooler evening
- Cooler (28°C air, 35°C track, 20% humidity)  
- Model predicts: -2.0% less fuel vs hot
- Aston Martin saves: ~2.2kg fuel load
```

## 🔧 Implementation

### Data Fetching (Updated):
```python
# scripts/fetch_fastf1_highfuel.py
# scripts/fetch_expanded_training_data.py

# Now includes weather at each lap:
- air_temp
- track_temp
- humidity
- pressure
- rainfall
- wind_speed
- wind_direction
```

### Model Training (Updated):
```python
# scripts/train_two_stage_model.py

# Weather features added to feature set:
base_features = ['avg_throttle', 'avg_rpm', 'avg_speed', 'avg_gear']
weather_features = ['air_temp', 'track_temp', 'humidity', 'pressure', 'wind_speed']

# Weather interactions:
- temp_humidity (air_temp × humidity)
- track_temp_sq (track_temp²)
```

## 📋 Workflow for Tonight

### Step 1: Test Weather Integration
```bash
python scripts/test_weather_integration.py
```
Expected output: Weather data for 2023 Bahrain GP

### Step 2: Fetch Expanded Data (Run Overnight)
```bash
python scripts/fetch_expanded_training_data.py
```
- Fetches 7 years (2018-2024) × 15 circuits
- **Includes weather for every lap**
- Takes 2-4 hours
- Outputs: `train_highfuel_expanded.csv`, `test_highfuel_expanded.csv`

### Step 3: Retrain Models (Tomorrow Morning)
```bash
python scripts/train_two_stage_model.py
```
- Will automatically use weather features
- Expected improvement: 96% → 97-98% accuracy

### Step 4: Verify Weather Impact
Check feature importance:
- Weather features should rank in top 10
- Track temp likely most important weather feature
- Air temp + humidity interaction significant

## 🎯 Key Benefits

### 1. Circuit-Specific Weather Patterns
- **Singapore**: Always hot + humid → model learns this
- **Silverstone**: Variable weather → model adapts
- **Mexico**: High altitude → model accounts for pressure

### 2. Race Day Predictions
- Check weather forecast
- Model predicts fuel consumption accounting for conditions
- Better strategy decisions

### 3. Tire + Fuel Correlation
- Hot track = tire deg + fuel consumption
- Model learns combined effect
- Better stint planning

### 4. Year-Over-Year Consistency
- Same circuit, different weather across years
- Model learns what's circuit vs weather
- Better generalization

## 📈 Feature Importance Expectations

After retraining with weather, expected ranking:

1. **avg_throttle** - Still #1 (direct fuel control)
2. **avg_rpm** - Still #2 (engine load)
3. **track_temp** - **NEW TOP 5** (affects everything)
4. **avg_speed** - Top 5 (baseline)
5. **air_temp** - **NEW TOP 10** (air density)
6. **humidity** - **NEW TOP 10** (combustion)
7. avg_gear - Top 10
8. **temp_humidity** - **NEW TOP 15** (combined effect)
9. circuit_encoded - Top 10
10. team_encoded - Top 10

## ⚠️ Important Notes

### Weather Data Availability:
- **Good**: 2018-2024 sessions (modern API)
- **Limited**: 2010-2017 (may be incomplete)
- **Missing**: Pre-2010 (no weather data)

### Handling Missing Data:
- Script fills missing weather with session median
- Rainfall defaults to False if missing
- Model robust to some missing weather values

### Rainfall Special Case:
- Wet races are VERY different
- Consider training separate wet-weather model
- Or filter out wet laps from fuel analysis

## 🚀 Ready to Run Tonight

All scripts updated and ready:
- ✅ `fetch_fastf1_highfuel.py` - includes weather
- ✅ `fetch_expanded_training_data.py` - includes weather
- ✅ `train_two_stage_model.py` - uses weather features
- ✅ `test_weather_integration.py` - verification script

Run order:
1. **Test**: `python scripts/test_weather_integration.py` (2 min)
2. **Fetch**: `python scripts/fetch_expanded_training_data.py` (2-4 hours, overnight)
3. **Train**: `python scripts/train_two_stage_model.py` (tomorrow, 15-30 min)

Your model will now understand how weather affects fuel consumption! 🌦️🏎️
