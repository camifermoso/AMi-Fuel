# 🏎️ **AMi-Fuel — Aston Martin Intelligent Fuel**  
**Fuel Strategy Intelligence for Formula 1**

AMi-Fuel is a high-resolution fuel-consumption intelligence system built for the **Aston Martin x Aramco F1 Engineering Challenge**.  
It blends a calibrated machine-learning model with an F1-inspired professional dashboard to estimate per-lap fuel usage, analyze race stints, and support strategic decision-making with clarity and engineering rigor.

🔥 **Live Demo:**  
👉 **https://camifermoso-ami-fuel.streamlit.app**

No setup required — just click and use.

---

## 🚀 **What AMi-Fuel Does**

AMi-Fuel predicts **kg/lap fuel consumption** using lap-level telemetry, weather, and engineered interaction features.  
It helps engineers:

- Evaluate how driver behavior affects fuel burn  
- Compare laps and stints side-by-side  
- Experiment with setup levers (throttle, ERS, DRS levels)  
- Detect inefficiencies and potential savings  
- Study past races as if running an internal F1 debrief  

The dashboard consists of three core modules:

### ⛽ **Fuel Strategy Simulator**  
Live per-lap predictions, driver/circuit selection, setup adjustments, and fuel-efficiency indicators.

### 📊 **Race Fuel Debrief**  
Lap-by-lap post-race analysis using FastF1 telemetry.  
Shows predicted fuel, variation across the race, and where improvements were possible.

### 🧠 **AI Model Briefing**  
Explains the model: training coverage, decisions made, feature engineering, calibration, and limitations.

---

## 🧩 **Model Summary**

AMi-Fuel uses a **tree-based regression stack**. 

### ✔ Architecture  
- **Gradient Boosting Regressor**  
- **Random Forest Regressor**  
- **Post-hoc calibration** to ensure kg/lap consistency  
- **Circuit baseline adjustments** for track realism

### ✔ Input Features  
Lap-level aggregates of:
- RPM  
- Throttle  
- Speed  
- Gear  
- DRS  
- ERS  
- Weather variables  
- Encoded team & circuit  

Plus physics-inspired interaction features capturing nonlinear mechanical and aerodynamic relationships.

### ✔ Why tree models?  
Tree ensembles offer:
- stability across circuits  
- interpretability for engineers  
- excellent nonlinear interaction handling  
- robustness to telemetry noise  
- easy calibration for domain constraints  

---

# **Data Pipeline**

### **Cleaning**
- Duplicate removal  
- Drop missing critical values (RPM/throttle/speed)  
- Impute safe gaps  
- Remove out-of-domain outliers  
- Keep only green/yellow-flag laps

### **Aggregation**
Telemetry is aggregated per lap:
- Means for continuous signals  
- Max for speed traps  
- Standard deviation for variability  
- Sector-level summaries

### **Scaling**
Uses **RobustScaler** (median + IQR) for stability across noisy telemetry distributions.

---

# 📈 **Model Performance**

Typical behavior:
- **R²:** 0.85–0.95  
- **MAE:** 0.02–0.04  
- Consistent predictions across circuits  
- High stability across folds  

---

# 🌐 **Use AMi-Fuel Online**

### 👉 **https://camifermoso-ami-fuel.streamlit.app**

This is the recommended and primary way to use the tool.

---

# ⚙️ (Optional) **Run Locally If Needed**

If the hosted version is offline:

```bash
git clone https://github.com/camifermoso/AMi-Fuel.git
cd AMi-Fuel
pip install -r requirements.txt
streamlit run app.py
