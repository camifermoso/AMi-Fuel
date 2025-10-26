
# AMi Fuel
**Aston Martin Intelligent Fuel**  
Machine learning for predictive fuel optimization.

## Description
**AMi Fuel** is a machine-learning-driven system that models and optimizes race-fuel consumption using telemetry data.  
By learning how driving parameters affect fuel flow and combining those insights with an optimization engine, AMi Fuel helps engineers balance performance and efficiency without altering car design or fuel chemistry.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_synth_data.py
python src/fuel_model.py --data data/synth_telemetry.csv --save-model outputs/fuel_model.pkl
python src/optimizer.py --data data/synth_telemetry.csv --model outputs/fuel_model.pkl --fuel-cap-kg 110
```
