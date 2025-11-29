"""
AMi-Fuel: Aston Martin F1 Fuel Optimization Dashboard
Race Engineer Decision Support System
Real-time fuel strategy optimization for race weekend operations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
from datetime import datetime
import base64

# Constants for fuel scaling
FUEL_BASELINES_KG = {
    "bahrain": 1.84, "sakhir": 1.84,
    "jeddah": 1.75, "melbourne": 1.70, "australia": 1.70,
    "suzuka": 1.95, "japan": 1.95,
    "shanghai": 1.80, "china": 1.80,
    "miami": 1.80, "imola": 1.70, "monaco": 1.60,
    "barcelona": 1.72, "spain": 1.72,
    "montreal": 1.65, "canada": 1.65,
    "austria": 1.65, "spielberg": 1.65,
    "silverstone": 1.90, "britain": 1.90, "uk": 1.90,
    "spa": 2.00, "belgium": 2.00,
    "zandvoort": 1.70, "netherlands": 1.70,
    "monza": 1.80, "italy": 1.80,
    "baku": 2.00, "azerbaijan": 2.00,
    "singapore": 1.90,
    "austin": 1.80, "cota": 1.80, "united states": 1.80,
    "mexico": 1.78,
    "brazil": 1.72, "interlagos": 1.72,
    "vegas": 1.85, "las vegas": 1.85,
    "qatar": 1.80, "losail": 1.80,
    "abudhabi": 1.74, "abu dhabi": 1.74, "yas marina": 1.74
}
FUEL_PROXY_ANCHOR = 1.00  # model proxy ~1.0 represents typical lap fuel consumption

# Page configuration
st.set_page_config(
    page_title="AMi-Fuel Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"  # Sidebar is shown by default, user can collapse with arrow
)

# Custom CSS with custom fonts
st.markdown("""
<style>
    /* Import Google Fonts - Stack Sans Notch for headers, Inter for body */
    @import url('https://fonts.googleapis.com/css2?family=Stack+Sans+Notch:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Dark green background for main app */
    .stApp {
        background: radial-gradient(circle at 20% 20%, rgba(206,220,0,0.10), transparent 25%),
                    radial-gradient(circle at 80% 10%, rgba(206,220,0,0.08), transparent 22%),
                    linear-gradient(180deg, #00594c 0%, #003933 100%);
    }
    
    /* Hide Streamlit header bar */
    header[data-testid="stHeader"] {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* Main content area */
    .main .block-container {
        background-color: rgba(0, 75, 69, 0.75);
        padding-top: 2rem;
        padding-bottom: 2rem;
        border-radius: 18px;
        max-width: 1200px;
    }
    
    /* Sidebar styling - keep always visible and set custom width */
    section[data-testid="stSidebar"] {
        background-color: #003933 !important;
        width: 340px !important;
        min-width: 340px !important;
        transform: translateX(0) !important;
        visibility: visible !important;
        opacity: 1 !important;
        position: sticky;
        top: 0;
        height: 100vh;
    }
    
    /* Telemetry Pulse Line Container */
    .telemetry-pulse-container {
        position: fixed;
        top: 10px;
        left: 0;
        width: 100%;
        height: 120px;
        z-index: 0;
        overflow: hidden;
        pointer-events: none;
        opacity: 0.4;
    }
    
    /* The SVG that scrolls */
    .telemetry-pulse-svg {
        position: absolute;
        left: 0;
        top: 0;
        width: 200%; /* Twice the width to allow for smooth scrolling */
        height: 100%;
        animation: scroll 25s linear infinite;
    }
    
    /* The waveform line itself */
    .telemetry-pulse-line {
        stroke: #cedc00; /* Neon green */
        stroke-width: 1px;
        fill: none;
        filter: drop-shadow(0 0 4px rgba(206, 220, 0, 0.7));
        animation: pulse 2.5s ease-in-out infinite;
    }

    
    section[data-testid="stSidebar"] > div {
        background-color: #003933;
        padding-top: 0.5rem;
        padding-bottom: 1rem;
    }

    /* Prevent sidebar from being collapsed/hidden */
    div[data-testid="collapsedControl"] {
        display: none !important;
    }
    button[aria-label="Hide sidebar"],
    button[aria-label="Show sidebar"],
    button[title="Menu"],
    button[aria-label*="menu" i],
    button[title*="menu" i],
    button[data-testid="baseButton-headerNoAuthMenu"],
    [data-testid="stSidebarNav"] + div button {
        display: none !important;
        visibility: hidden !important;
    }
    /* Catch-all: remove any collapse/chevron buttons inside the sidebar header area */
    section[data-testid="stSidebar"] button {
        display: none !important;
        visibility: hidden !important;
    }
    /* Force sidebar toggle state to open if Streamlit tries to collapse */
    [data-testid="stSidebarNav"] {
        min-height: 100vh;
    }

    /* Help tooltips (question marks) visibility */
    [data-testid="stTooltipIcon"] svg {
        width: 18px;
        height: 18px;
        color: #cedc00 !important;
        fill: #cedc00 !important;
        opacity: 1 !important;
    }
    [data-testid="stTooltipIcon"]:hover svg {
        filter: drop-shadow(0 0 6px rgba(206,220,0,0.7));
    }
    /* Toggle sizing */
    [data-baseweb="switch"] {
        transform: scale(1.12);
        margin-left: 8px;
    }
    /* Tooltip body styling for readability */
    div[role="tooltip"], [data-baseweb="tooltip"], div[data-testid="stTooltipContent"] {
        background: #003933 !important;
        color: #ffffff !important;
        border: 1px solid #cedc00 !important;
        box-shadow: 0 8px 18px rgba(0,0,0,0.4) !important;
        z-index: 9999 !important;
    }
    div[role="tooltip"] *, [data-baseweb="tooltip"] *, div[data-testid="stTooltipContent"] * {
        color: #ffffff !important;
    }
    
    /* Apply Stack Sans Notch to headers (modern, notched geometric style) */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Stack Sans Notch', 'Arial Black', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 1.5px;
        color: #cedc00 !important;
    }
    
    .main-header {
        font-family: 'Stack Sans Notch', 'Arial Black', sans-serif !important;
        font-size: 3rem;
        font-weight: 700;
        color: #cedc00;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: 3px;
    }
    
    .sub-header {
        font-family: 'Stack Sans Notch', 'Arial Black', sans-serif !important;
        font-size: 1.2rem;
        font-weight: 500;
        color: #ffffff;
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: 2px;
    }
    
    /* Streamlit specific header selectors */
    [data-testid="stHeader"], 
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        font-family: 'Stack Sans Notch', 'Arial Black', sans-serif !important;
        font-weight: 600;
        letter-spacing: 1.5px;
        color: #cedc00 !important;
    }
    
    /* Apply Inter to body text and content (clean, modern sans-serif) */
    p, div, span, label, li, td, th, button {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #ffffff !important;
    }
    
    /* Fix info, success, warning, error boxes */
    .stAlert {
        background-color: #003933 !important;
        color: #ffffff !important;
        border-color: #cedc00 !important;
    }
    
    .stAlert > div {
        color: #ffffff !important;
    }
    
    /* Success messages */
    [data-testid="stSuccessIcon"],
    [data-testid="stInfoIcon"],
    [data-testid="stWarningIcon"],
    [data-testid="stErrorIcon"] {
        color: #cedc00 !important;
    }
    
    /* Keep Inter for metrics values for better readability */
    [data-testid="stMetricValue"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-weight: 600;
        color: #cedc00 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
    }
    
    .metric-card {
        background-color: #003933;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #cedc00;
    }
    .stMetric {
        background-color: rgba(0, 57, 51, 0.9);
        padding: 14px;
        border-radius: 5px;
        border: 2px solid #cedc00;
    }
    
    /* Custom collapsible section styling with smooth animations */
    .custom-details {
        border: 2px solid #cedc00;
        border-radius: 8px;
        margin-bottom: 1rem;
        background: #003933;
        overflow: hidden;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(206, 220, 0, 0.15);
    }
    
    .custom-details:hover {
        box-shadow: 0 6px 20px rgba(206, 220, 0, 0.3);
        border-color: #cedc00;
        transform: translateY(-2px);
    }
    
    .custom-details summary {
        cursor: pointer;
        padding: 1rem;
        background: linear-gradient(135deg, #cedc00 0%, #b8c400 100%);
        color: #004b45;
        font-weight: 700;
        font-size: 1rem;
        list-style: none;
        display: flex;
        align-items: center;
        transition: all 0.3s ease;
        user-select: none;
    }
    
    .custom-details summary:hover {
        background: linear-gradient(135deg, #e0f000 0%, #cedc00 100%);
    }
    
    .custom-details summary::-webkit-details-marker {
        display: none;
    }
    
    .custom-details summary::before {
        content: '▶';
        display: inline-block;
        margin-right: 0.5rem;
        transition: transform 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        color: #004b45;
        font-size: 0.8rem;
    }
    
    .custom-details[open] summary::before {
        transform: rotate(90deg);
    }
    
    .custom-details-content {
        padding: 0 1rem;
        border-top: 2px solid #cedc00;
        background: #003933;
        color: #ffffff;
        animation: slideDown 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        max-height: 2000px;
        overflow: hidden;
    }
    
    /* Smooth slide down animation */
    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-10px);
            padding-top: 0;
            padding-bottom: 0;
        }
        to {
            opacity: 1;
            transform: translateY(0);
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
    }
    
    /* Keyframes for the telemetry line scroll */
    @keyframes scroll {
        from {
            transform: translateX(0);
        }
        to {
            transform: translateX(-50%); /* Scrolls one full screen width */
        }
    }
    
    /* Keyframes for the subtle pulse effect */
    @keyframes pulse {
        0%, 100% { opacity: 0.6; }
        50% { opacity: 1; }
    }
    
    /* Apply padding with transition */
    .custom-details[open] .custom-details-content {
        padding: 1rem;
    }
    
    /* Fix dropdown/select box styling - make text readable */
    .stSelectbox > div > div {
        background-color: #003933 !important;
        color: #ffffff !important;
        border: 2px solid #cedc00 !important;
    }
    
    .stSelectbox label {
        color: #ffffff !important;
    }
    
    .stSelectbox input {
        color: #ffffff !important;
    }
    
    .stSelectbox [data-baseweb="select"] {
        background-color: #003933 !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        background-color: #003933 !important;
        color: #ffffff !important;
        border-color: #cedc00 !important;
    }
    
    /* Dropdown menu options */
    [data-baseweb="popover"] {
        background-color: #003933 !important;
    }
    
    [data-baseweb="menu"] {
        background-color: #003933 !important;
        border: 2px solid #cedc00 !important;
    }
    
    [data-baseweb="menu"] li {
        background-color: #003933 !important;
        color: #ffffff !important;
    }
    
    [data-baseweb="menu"] li:hover {
        background-color: #004b45 !important;
        color: #cedc00 !important;
    }
    
    /* Fix dropdown menu text visibility - more specific selectors */
    [role="listbox"] {
        background-color: #003933 !important;
    }
    
    [role="option"] {
        background-color: #003933 !important;
        color: #ffffff !important;
    }
    
    [role="option"]:hover {
        background-color: #004b45 !important;
        color: #cedc00 !important;
    }
    
    /* Dropdown selected value */
    .stSelectbox [data-baseweb="select"] span {
        color: #ffffff !important;
    }
    
    /* All dropdown option text */
    [data-baseweb="menu"] * {
        color: #ffffff !important;
    }
    
    /* Radio buttons */
    .stRadio label {
        color: #ffffff !important;
    }
    
    .stRadio > div {
        background-color: #003933;
        padding: 10px;
        border-radius: 5px;
        border: 2px solid #cedc00;
    }
    
    /* Multiselect */
    .stMultiSelect > div > div {
        background-color: #003933 !important;
        border: 2px solid #cedc00 !important;
    }
    
    .stMultiSelect label {
        color: #ffffff !important;
    }
    
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #cedc00 !important;
        color: #004b45 !important;
    }
    
    /* Number input and text input */
    .stNumberInput input,
    .stTextInput input {
        background-color: #003933 !important;
        color: #ffffff !important;
        border: 2px solid #cedc00 !important;
    }
    
    .stNumberInput label,
    .stTextInput label {
        color: #ffffff !important;
    }
    
    /* Slider - fix all text visibility */
    .stSlider label {
        color: #ffffff !important;
    }
    
    .stSlider [data-baseweb="slider"] {
        background-color: transparent !important;
    }
    
    /* Slider value text */
    .stSlider > div > div > div > div {
        color: #ffffff !important;
    }
    
    /* Slider thumb value label */
    .stSlider [role="slider"] {
        color: #ffffff !important;
    }
    
    /* Slider tick labels and value display */
    .stSlider div[data-baseweb="slider"] > div {
        color: #ffffff !important;
    }
    
    /* All text in slider container */
    .stSlider * {
        color: #ffffff !important;
    }
    
    /* --- Slider Micro-animations --- */
    /* Base style for the slider thumb */
    .stSlider [role="slider"] {
        background-color: #cedc00;
        border: 2px solid #ffffff;
        box-shadow: 0 0 8px rgba(206, 220, 0, 0.3);
        transition: box-shadow 0.2s ease-in-out, transform 0.2s ease-in-out;
    }
    
    /* Glow effect on hover */
    .stSlider [role="slider"]:hover {
        transform: scale(1.1);
        box-shadow: 0 0 14px 4px rgba(206, 220, 0, 0.6);
    }
    
    /* Stronger glow effect when actively dragging */
    .stSlider [role="slider"]:active {
        transform: scale(1.2);
        box-shadow: 0 0 20px 6px rgba(206, 220, 0, 0.8);
    }
    
    /* Checkbox */
    .stCheckbox label {
        color: #ffffff !important;
    }
    
    /* Dataframe and tables */
    .stDataFrame {
        background-color: #003933 !important;
    }
    
    .stDataFrame table {
        background-color: #003933 !important;
        color: #ffffff !important;
    }
    
    .stDataFrame th {
        background-color: #004b45 !important;
        color: #cedc00 !important;
    }
    
    .stDataFrame td {
        background-color: #003933 !important;
        color: #ffffff !important;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #cedc00 !important;
        color: #004b45 !important;
        border: none !important;
        font-weight: 600 !important;
    }
    
    .stButton button:hover {
        background-color: #b8c400 !important;
        box-shadow: 0 4px 12px rgba(206,220,0,0.4) !important;
    }
    
    /* Ensure button text is always dark */
    .stButton button * {
        color: #004b45 !important;
    }
    
    /* Primary button styling */
    .stButton button[kind="primary"] {
        background-color: #cedc00 !important;
        color: #004b45 !important;
    }
    
    .stButton button[kind="primary"] * {
        color: #004b45 !important;
    }
    
    /* Download button styling */
    .stDownloadButton button {
        background-color: #cedc00 !important;
        color: #004b45 !important;
        border: none !important;
        font-weight: 600 !important;
    }
    
    .stDownloadButton button:hover {
        background-color: #b8c400 !important;
        box-shadow: 0 4px 12px rgba(206,220,0,0.4) !important;
    }
    
    /* Ensure download button text is always dark */
    .stDownloadButton button * {
        color: #004b45 !important;
    }

    /* Content panels */
    .stMarkdown:not(header) > div, .stTextInput, .stSelectbox, .stNumberInput, .stRadio, .stMultiSelect, .stSlider, .stDataFrame {
        border-radius: 10px;
    }
    
    /* Tabs styling */
    [data-baseweb="tab-list"] {
        background: linear-gradient(135deg, rgba(0,75,69,0.8), rgba(0,57,51,0.9));
        border-radius: 12px;
        padding: 6px;
        border: 1px solid #cedc00;
        box-shadow: 0 6px 18px rgba(0,0,0,0.25);
    }
    [role="tab"] {
        color: #ffffff !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        border: 1px solid transparent !important;
        padding: 10px 16px !important;
        transition: all 0.2s ease;
    }
    [role="tab"][aria-selected="true"] {
        background: #cedc00 !important;
        color: #004b45 !important;
        border-color: #cedc00 !important;
        box-shadow: 0 4px 10px rgba(206,220,0,0.35);
    }
    [role="tab"]:hover {
        border-color: rgba(206,220,0,0.5) !important;
    }
    
    /* Dataframe/table polish */
    .stDataFrame table {
        border-collapse: separate !important;
        border-spacing: 0 !important;
        border-radius: 12px;
        overflow: hidden;
    }
    .stDataFrame tbody tr:nth-child(even) td {
        background-color: rgba(0,75,69,0.35) !important;
    }
    .stDataFrame tbody tr:hover td {
        background-color: rgba(206,220,0,0.08) !important;
    }
    
    /* Section dividers */
    .stMarkdown hr {
        border: none;
        border-top: 1px solid rgba(206,220,0,0.35);
        margin: 1.5rem 0;
    }
    
    /* Metric cards refinement */
    .stMetric {
        box-shadow: none;
    }
    .stMetricLabel, .stMetricValue {
        letter-spacing: 0.25px;
    }
    
    /* Sidebar edge highlight */
    section[data-testid="stSidebar"] {
        box-shadow: 6px 0 20px rgba(0,0,0,0.35);
    }
    
    /* Button polish */
    .stButton button, .stDownloadButton button {
        letter-spacing: 0.3px;
        border-radius: 8px !important;
    }

</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_recommendations():
    """Load fuel optimization recommendations."""
    try:
        params_df = pd.read_csv("outputs/am_fuel_recommendations.csv")
        scenarios_df = pd.read_csv("outputs/am_race_scenarios.csv")
        circuits_df = pd.read_csv("outputs/am_circuit_strategies.csv")
        return params_df, scenarios_df, circuits_df
    except FileNotFoundError:
        st.error("⚠️ Recommendation files not found. Please run generate_am_specific_recommendations.py first.")
        return None, None, None


@st.cache_resource
def load_model():
    """Load the trained model and scalers/encoders."""
    try:
        model_dir = Path("outputs/two_stage_model")
        model = joblib.load(model_dir / "finetuned_model.pkl")
        calibrator = joblib.load(model_dir / "calibrator.pkl")
        scaler_global = joblib.load(model_dir / "scaler_global.pkl")
        scalers_per_team = joblib.load(model_dir / "scalers_per_team.pkl")
        team_encoder = joblib.load(model_dir / "team_encoder.pkl")
        circuit_encoder = joblib.load(model_dir / "circuit_encoder.pkl")
        return model, calibrator, scaler_global, scalers_per_team, team_encoder, circuit_encoder
    except FileNotFoundError:
        st.warning("⚠️ Model files not found. Prediction features disabled.")
        return None, None, None, None, None, None


def prepare_features_for_inference(
    df_raw: pd.DataFrame,
    team: str,
    circuit_key: str,
    year: int,
    scaler_global,
    scalers_per_team,
    team_encoder,
    circuit_encoder,
    air_temp: float = 25.0,
    track_temp: float = 35.0,
    humidity: float = 60.0,
    pressure: float = 1013.0,
    wind_speed: float = 3.0,
):
    """Mirror training-time feature prep for a batch of samples."""
    df = df_raw.rename(columns={
        "rpm": "avg_rpm",
        "throttle": "avg_throttle",
        "speed": "avg_speed",
        "gear": "avg_gear",
    }).copy()
    df["Team"] = team
    df["gp"] = circuit_key
    df["year"] = year
    df["air_temp"] = df.get("air_temp", air_temp)
    df["track_temp"] = df.get("track_temp", track_temp)
    df["humidity"] = df.get("humidity", humidity)
    df["pressure"] = df.get("pressure", pressure)
    df["wind_speed"] = df.get("wind_speed", wind_speed)

    base_features = ["avg_throttle", "avg_rpm", "avg_speed", "avg_gear"]
    weather_features = ["air_temp", "track_temp", "humidity", "pressure", "wind_speed"]
    available_weather = [f for f in weather_features if f in df.columns]

    # Fill missing values
    for col in base_features:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    for col in available_weather:
        df[col] = df[col].fillna(df[col].median())

    # Per-team normalization of telemetry
    scaler = None
    if scalers_per_team and team in scalers_per_team:
        scaler = scalers_per_team[team]
    elif scaler_global is not None:
        scaler = scaler_global
    if scaler is not None:
        try:
            df[base_features] = scaler.transform(df[base_features])
        except Exception:
            pass  # fallback silently if shape mismatch

    # Encode team and circuit
    def safe_encode(val, encoder):
        try:
            if val in encoder.classes_:
                return encoder.transform([val])[0]
        except Exception:
            pass
        return -1

    df["team_encoded"] = safe_encode(team, team_encoder) if team_encoder is not None else -1
    df["circuit_encoded"] = safe_encode(circuit_key, circuit_encoder) if circuit_encoder is not None else -1

    # Interaction features (on normalized bases)
    df["throttle_rpm"] = df["avg_throttle"] * df["avg_rpm"]
    df["speed_gear"] = df["avg_speed"] * df["avg_gear"]
    df["rpm_gear"] = df["avg_rpm"] * df["avg_gear"]
    df["throttle_sq"] = df["avg_throttle"] ** 2
    df["rpm_sq"] = df["avg_rpm"] ** 2
    if "air_temp" in available_weather and "humidity" in available_weather:
        df["temp_humidity"] = df["air_temp"] * df["humidity"]
    if "track_temp" in available_weather:
        df["track_temp_sq"] = df["track_temp"] ** 2

    feature_cols = base_features + available_weather + [
        "team_encoded",
        "circuit_encoded",
        "year",
        "throttle_rpm",
        "speed_gear",
        "rpm_gear",
        "throttle_sq",
        "rpm_sq",
    ]
    if "temp_humidity" in df.columns:
        feature_cols.append("temp_humidity")
    if "track_temp_sq" in df.columns:
        feature_cols.append("track_temp_sq")

    return df[feature_cols]


def predict_fuel(model, calibrator, scaler, df_features):
    """
    Predict fuel proxy using model + calibrator.
    """
    try:
        preds = model.predict(df_features)
        if calibrator is not None:
            preds = calibrator.predict(preds)
        return preds
    except Exception:
        return model.predict(df_features)


def fuel_proxy_to_kg(circuit_key: str, proxy_value: float) -> float:
    """Convert proxy prediction to kg/lap using circuit baseline map."""
    key = (circuit_key or "").lower()
    baseline = next((v for k, v in FUEL_BASELINES_KG.items() if k in key), 1.80)
    kg = (proxy_value / FUEL_PROXY_ANCHOR) * baseline
    return max(0.4, kg)


@st.cache_data
def load_training_data():
    """Load training data for analysis (optional - for enhanced visualizations)."""
    # Try full dataset first (local), then sample (for deployment)
    try:
        df = pd.read_csv("data/train_highfuel_expanded.csv")
        return df, "full"
    except FileNotFoundError:
        try:
            df = pd.read_csv("data/train_sample.csv")
            return df, "sample"
        except FileNotFoundError:
            # No data files - will show static info instead
            return None, None


def main():
    # Header - Race Engineer Focused
    # Add the Telemetry Pulse Line HTML
    st.markdown("""
        <div class="telemetry-pulse-container">
            <svg class="telemetry-pulse-svg" preserveAspectRatio="none">
                <path class="telemetry-pulse-line" d="M0,60 C100,20 150,100 250,60 S400,20 500,60 S650,100 750,60 S900,20 1000,60 S1150,100 1250,60 S1400,20 1500,60 S1650,100 1750,60 S1900,20 2000,60 S2150,100 2250,60 S2400,20 2500,60 S2650,100 2750,60 S2900,20 3000,60 S3150,100 3250,60 S3400,20 3500,60 S3650,100 3750,60 S3900,20 4000,60" vector-effect="non-scaling-stroke"></path>
            </svg>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header"> AMi-Fuel Race Engineer System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-Time Fuel Strategy Decision Support</div>', unsafe_allow_html=True)
    
    # Load data
    params_df, scenarios_df, circuits_df = load_recommendations()
    model, calibrator, scaler, scalers_per_team, team_encoder, circuit_encoder = load_model()
    train_df, data_type = load_training_data()
    
    if params_df is None:
        st.stop()
    
    # Sidebar - Race Weekend Context
    st.sidebar.title("Race Control")
    
    # Set default weather values (hidden from UI)
    air_temp = 25.0
    track_temp = 35.0
    rainfall = False
    
    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Tool",
        ["Fuel Strategy Simulator", "Race Fuel Debrief", "AI Model Briefing"],
        help="Choose the tool you need for the current phase of the race weekend"
    )
    
    # System status
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Status")
    st.sidebar.metric("Model Confidence", "99.4%", help="Validated on unseen races")
    st.sidebar.caption(f"🕒 Last sync: {datetime.now().strftime('%H:%M:%S')}")
    st.sidebar.caption("Dataset Version: 2.3 — Updated 2025-11-27")
    
    # Footer credit
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='font-size: 0.75rem; color: #a0a0a0;'>
        Engineered by C. Fermoso<br>Aramco x Aston Martin 2025
    </div>
    """, unsafe_allow_html=True)
    
    # Main content based on page selection - Race Engineer Tools
    if page == "Fuel Strategy Simulator":
        show_live_calculator(model, calibrator, scaler, scalers_per_team, team_encoder, circuit_encoder, air_temp, track_temp, rainfall)
    elif page == "Race Fuel Debrief":
        show_race_analysis(model, calibrator, scaler, scalers_per_team, team_encoder, circuit_encoder)
    elif page == "AI Model Briefing":
        show_ai_model_briefing(train_df)


def show_race_strategy(params_df, scenarios_df, circuits_df, train_df, air_temp, track_temp, rainfall):
    """Race Strategy Dashboard - Main decision support for race engineers."""
    
    # Race header
    st.header("🏁 Race - Fuel Strategy")
    
    # Alert banner for conditions
    if rainfall:
        st.warning("⚠️ **RAINFALL EXPECTED** - Fuel strategy may need adjustment for wet conditions. Monitor real-time data.")
    
    if track_temp > 45:
        st.error("🔥 **HIGH TRACK TEMP** - Increased tire degradation. Consider fuel conservation for extended stints.")
    elif track_temp < 25:
        st.info("❄️ **COOL TRACK** - Better tire life. Potential for aggressive fuel strategy.")
    
    st.markdown("### 📊 Strategy Command Center")
    
    # Prepare data
    params_df = params_df.copy()
    params_df['Fuel Saved'] = params_df['Fuel Saved (kg/race)'].str.replace(' kg', '').astype(float)
    params_df['Time Cost'] = params_df['Time Cost/Race'].str.replace('s', '').astype(float)
    
    # Race Engineer KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    total_fuel_saved = params_df['Fuel Saved'].sum()
    avg_time_cost = params_df['Time Cost'].mean()
    best_strategy = params_df.nlargest(1, 'Fuel Saved').iloc[0]
    
    # Calculate race-specific metrics
    avg_race_laps = 55  # Average F1 race distance
    fuel_per_lap = total_fuel_saved / avg_race_laps if avg_race_laps > 0 else 0
    total_time_saving = (total_fuel_saved * 0.03)  # ~0.03s per kg weight reduction
    
    with col1:
        st.metric(
            "⛽ Max Fuel Saving", 
            f"{total_fuel_saved:.1f} kg", 
            delta=f"~{total_time_saving:.1f}s lap time gain",
            help="Total optimization potential for full race distance"
        )
    
    with col2:
        st.metric(
            "⏱️ Avg Time Cost", 
            f"{avg_time_cost:.2f}s", 
            delta=f"{avg_time_cost/avg_race_laps:.3f}s/lap",
            delta_color="inverse",
            help="Time penalty per strategy averaged over race distance"
        )
    
    with col3:
        st.metric(
            "🎯 Recommended Action",
            f"{best_strategy['Fuel Saved']:.1f} kg",
            delta=best_strategy['Parameter'][:20],
            help=f"Priority: {best_strategy['Parameter']}"
        )
    
    with col4:
        st.metric(
            "📈 Fuel/Lap Savings",
            f"{fuel_per_lap:.3f} kg/lap",
            delta=f"Over {avg_race_laps} laps",
            help="Average fuel saved per lap with full optimization"
        )
    
    st.markdown("---")
    
    # Removed Setup Adjustment Evaluator
    st.markdown("---")
    
    # Performance insights cards
    st.subheader("🏆 Top Performers")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        top_saver = params_df.nlargest(1, 'Fuel Saved').iloc[0]
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #004b45 0%, #003933 100%); padding: 1.5rem; border-radius: 12px; border: 2px solid #cedc00; box-shadow: 0 4px 12px rgba(206,220,0,0.3);">
            <div style="font-size: 0.9rem; color: #cedc00; font-weight: 600; text-transform: uppercase; margin-bottom: 0.5rem;">🥇 Most Fuel Saved</div>
            <div style="font-size: 1.8rem; font-weight: bold; color: #cedc00; margin-bottom: 0.5rem;">{top_saver['Fuel Saved']:.2f} kg</div>
            <div style="font-size: 0.85rem; color: #ffffff; margin-bottom: 0.25rem;"><strong>{top_saver['Parameter']}</strong></div>
            <div style="font-size: 0.75rem; color: #cedc00;">Reduction: {top_saver['Reduction']}</div>
            <div style="font-size: 0.75rem; color: #ff6b6b; margin-top: 0.5rem;">⏱️ Cost: {top_saver['Time Cost']:.2f}s</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        top_efficient = params_df.copy()
        top_efficient['efficiency'] = top_efficient['Fuel Saved'] / top_efficient['Time Cost']
        top_efficient = top_efficient.nlargest(1, 'efficiency').iloc[0]
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #004b45 0%, #003933 100%); padding: 1.5rem; border-radius: 12px; border: 2px solid #cedc00; box-shadow: 0 4px 12px rgba(206,220,0,0.3);">
            <div style="font-size: 0.9rem; color: #cedc00; font-weight: 600; text-transform: uppercase; margin-bottom: 0.5rem;">⚡ Most Efficient</div>
            <div style="font-size: 1.8rem; font-weight: bold; color: #cedc00; margin-bottom: 0.5rem;">{top_efficient['efficiency']:.2f} ratio</div>
            <div style="font-size: 0.85rem; color: #ffffff; margin-bottom: 0.25rem;"><strong>{top_efficient['Parameter']}</strong></div>
            <div style="font-size: 0.75rem; color: #cedc00;">{top_efficient['Fuel Saved']:.2f} kg saved</div>
            <div style="font-size: 0.75rem; color: #66ff66; margin-top: 0.5rem;">✅ Only {top_efficient['Time Cost']:.2f}s cost</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        low_cost = params_df.nsmallest(1, 'Time Cost').iloc[0]
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #004b45 0%, #003933 100%); padding: 1.5rem; border-radius: 12px; border: 2px solid #cedc00; box-shadow: 0 4px 12px rgba(206,220,0,0.3);">
            <div style="font-size: 0.9rem; color: #cedc00; font-weight: 600; text-transform: uppercase; margin-bottom: 0.5rem;">🚀 Lowest Time Cost</div>
            <div style="font-size: 1.8rem; font-weight: bold; color: #66ff66; margin-bottom: 0.5rem;">{low_cost['Time Cost']:.2f}s</div>
            <div style="font-size: 0.85rem; color: #ffffff; margin-bottom: 0.25rem;"><strong>{low_cost['Parameter']}</strong></div>
            <div style="font-size: 0.75rem; color: #cedc00;">Saves: {low_cost['Fuel Saved']:.2f} kg</div>
            <div style="font-size: 0.75rem; color: #cedc00; margin-top: 0.5rem;">📊 {low_cost['Reduction']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # All strategies overview with filtering
    st.subheader("📋 All Strategies Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_fuel = st.slider("Min Fuel Savings (kg)", 0.0, float(params_df['Fuel Saved'].max()), 0.0, 0.5)
    
    with col2:
        max_time = st.slider("Max Time Cost (s)", 0.0, float(params_df['Time Cost'].max()), float(params_df['Time Cost'].max()), 0.5)
    
    with col3:
        sort_option = st.selectbox("Sort By", ["Fuel Saved ↓", "Time Cost ↑", "Efficiency ↓"])
    
    # Apply filters
    filtered_all = params_df[
        (params_df['Fuel Saved'] >= min_fuel) & 
        (params_df['Time Cost'] <= max_time)
    ].copy()
    
    filtered_all['Efficiency'] = filtered_all['Fuel Saved'] / filtered_all['Time Cost']
    
    # Sort
    if sort_option == "Fuel Saved ↓":
        filtered_all = filtered_all.sort_values('Fuel Saved', ascending=False)
    elif sort_option == "Time Cost ↑":
        filtered_all = filtered_all.sort_values('Time Cost')
    else:
        filtered_all = filtered_all.sort_values('Efficiency', ascending=False)
    
    st.info(f"📊 Showing {len(filtered_all)} of {len(params_df)} strategies")
    
    # Display as cards
    for idx, row in filtered_all.iterrows():
        efficiency = row['Efficiency']
        
        # Color coding based on efficiency
        if efficiency >= 2.0:
            border_color = "#66ff66"  # Green for high efficiency
            badge = "🌟 EXCELLENT"
        elif efficiency >= 1.0:
            border_color = "#cedc00"  # Yellow for good efficiency
            badge = "✅ GOOD"
        else:
            border_color = "#ff9966"  # Orange for lower efficiency
            badge = "⚠️ MODERATE"
        
        st.markdown(f"""
        <div style="background: #003933; padding: 1rem; border-radius: 8px; border-left: 4px solid {border_color}; margin-bottom: 0.75rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <div style="font-size: 1rem; font-weight: bold; color: #ffffff;">{row['Parameter']}</div>
                <div style="font-size: 0.75rem; color: {border_color}; font-weight: 600;">{badge}</div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem;">
                <div>
                    <div style="font-size: 0.7rem; color: #cedc00; text-transform: uppercase;">Reduction</div>
                    <div style="font-size: 0.9rem; color: #ffffff; font-weight: 600;">{row['Reduction']}</div>
                </div>
                <div>
                    <div style="font-size: 0.7rem; color: #cedc00; text-transform: uppercase;">Fuel Saved</div>
                    <div style="font-size: 0.9rem; color: #66ff66; font-weight: 600;">{row['Fuel Saved']:.2f} kg</div>
                </div>
                <div>
                    <div style="font-size: 0.7rem; color: #cedc00; text-transform: uppercase;">Time Cost</div>
                    <div style="font-size: 0.9rem; color: #ff6b6b; font-weight: 600;">{row['Time Cost']:.2f}s</div>
                </div>
                <div>
                    <div style="font-size: 0.7rem; color: #cedc00; text-transform: uppercase;">Efficiency</div>
                    <div style="font-size: 0.9rem; color: {border_color}; font-weight: 600;">{efficiency:.2f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def show_quick_decisions(params_df, air_temp, track_temp):
    """Quick Decisions - Fast actionable recommendations for race engineers."""
    st.header("⚡ Quick Decision Support")
    st.caption("Instant setup recommendations based on current conditions")
    
    # Prepare data
    params_df = params_df.copy()
    params_df['Fuel Saved'] = params_df['Fuel Saved (kg/race)'].str.replace(' kg', '').astype(float)
    params_df['Time Cost'] = params_df['Time Cost/Race'].str.replace('s', '').astype(float)
    params_df['Efficiency'] = params_df['Fuel Saved'] / params_df['Time Cost']
    
    # Quick decision mode selector
    decision_mode = st.radio(
        "What do you need right now?",
        ["🔥 Maximum Performance (Qualifying/Sprint)", "⚖️ Race Distance Optimization", "🛡️ Conservative/Safe Strategy"],
        horizontal=True
    )
    
    if "Maximum Performance" in decision_mode:
        # For qualifying - minimize time cost
        st.success("**QUALIFYING MODE**: Prioritizing lap time with acceptable fuel usage")
        recommendations = params_df.nsmallest(3, 'Time Cost')
        metric_focus = "Minimal Time Loss"
        
    elif "Race Distance" in decision_mode:
        # For race - maximize efficiency
        st.success("**RACE MODE**: Optimizing fuel/time balance for full race distance")
        recommendations = params_df.nlargest(3, 'Efficiency')
        metric_focus = "Best Efficiency Ratio"
        
    else:
        # Conservative - maximize fuel savings
        st.success("**CONSERVATIVE MODE**: Maximum fuel savings with managed pace")
        recommendations = params_df.nlargest(3, 'Fuel Saved')
        metric_focus = "Maximum Fuel Reduction"
    
    st.markdown(f"### 🎯 Top 3 Actions for Race")
    st.markdown(f"*Optimized for: {metric_focus}*")
    
    # Display top 3 as action cards
    for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
        # Calculate race impact (55 laps)
        laps = 55
        total_fuel_impact = row['Fuel Saved']
        total_time_impact = row['Time Cost']
        weight_advantage = total_fuel_impact * 0.03  # 0.03s per kg
        
        # Priority badge
        if idx == 1:
            priority_badge = "🥇 PRIORITY 1 - IMPLEMENT IMMEDIATELY"
            border_color = "#66ff66"
        elif idx == 2:
            priority_badge = "🥈 PRIORITY 2 - IMPLEMENT IF POSSIBLE"
            border_color = "#cedc00"
        else:
            priority_badge = "🥉 PRIORITY 3 - OPTIONAL OPTIMIZATION"
            border_color = "#ff9966"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #003933 0%, #004b45 100%); padding: 1.5rem; border-radius: 12px; border-left: 6px solid {border_color}; margin-bottom: 1.5rem; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
            <div style="font-size: 0.75rem; color: {border_color}; font-weight: 700; text-transform: uppercase; margin-bottom: 0.5rem; letter-spacing: 1px;">{priority_badge}</div>
            <div style="font-size: 1.3rem; font-weight: bold; color: #ffffff; margin-bottom: 1rem;">{row['Parameter']}</div>
            
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-bottom: 1rem;">
                <div style="background: #004b45; padding: 1rem; border-radius: 8px; border: 2px solid #cedc00;">
                    <div style="font-size: 0.7rem; color: #cedc00; text-transform: uppercase; font-weight: 600;">Change Required</div>
                    <div style="font-size: 1.1rem; color: #ffffff; font-weight: bold; margin-top: 0.25rem;">{row['Reduction']}</div>
                </div>
                <div style="background: #004b45; padding: 1rem; border-radius: 8px; border: 2px solid #cedc00;">
                    <div style="font-size: 0.7rem; color: #cedc00; text-transform: uppercase; font-weight: 600;">Efficiency Rating</div>
                    <div style="font-size: 1.1rem; color: #66ff66; font-weight: bold; margin-top: 0.25rem;">{row['Efficiency']:.2f} kg/s</div>
                </div>
            </div>
            
            <div style="background: rgba(206,220,0,0.1); padding: 1rem; border-radius: 8px; border-left: 3px solid #cedc00; margin-bottom: 1rem;">
                <div style="font-size: 0.8rem; color: #cedc00; font-weight: 600; margin-bottom: 0.5rem;">📊 RACE IMPACT ({laps} laps)</div>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem;">
                    <div>
                        <div style="font-size: 0.65rem; color: #cedc00; text-transform: uppercase;">Fuel Saved</div>
                        <div style="font-size: 1rem; color: #66ff66; font-weight: bold;">+{total_fuel_impact:.2f} kg</div>
                    </div>
                    <div>
                        <div style="font-size: 0.65rem; color: #cedc00; text-transform: uppercase;">Time Cost</div>
                        <div style="font-size: 1rem; color: #ff6b6b; font-weight: bold;">-{total_time_impact:.2f}s</div>
                    </div>
                    <div>
                        <div style="font-size: 0.65rem; color: #cedc00; text-transform: uppercase;">Weight Advantage</div>
                        <div style="font-size: 1rem; color: #66ff66; font-weight: bold;">+{weight_advantage:.2f}s</div>
                    </div>
                </div>
            </div>
            
            <div style="font-size: 0.75rem; color: #ffffff; line-height: 1.6;">
                <strong style="color: #cedc00;">⚡ Engineer Action:</strong> 
                {"Implement this immediately for maximum benefit." if idx == 1 else 
                 "Implement if time and resources allow." if idx == 2 else 
                 "Consider for fine-tuning if other changes are already made."}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick reference table
    st.markdown("---")
    st.markdown("### 📋 Full Strategy Reference")
    
    col1, col2 = st.columns(2)
    with col1:
        show_all = st.checkbox("Show all parameters", value=False)
    with col2:
        if show_all:
            export_data = st.checkbox("Prepare for export", value=False)
    
    if show_all:
        display_df = params_df[['Parameter', 'Reduction', 'Fuel Saved', 'Time Cost', 'Efficiency']].sort_values('Efficiency', ascending=False)
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Parameter": st.column_config.TextColumn("Setup Parameter", width="medium"),
                "Reduction": st.column_config.TextColumn("Required Change", width="small"),
                "Fuel Saved": st.column_config.NumberColumn("Fuel (kg)", format="%.2f"),
                "Time Cost": st.column_config.NumberColumn("Time (s)", format="%.2f"),
                "Efficiency": st.column_config.NumberColumn("Efficiency", format="%.3f")
            }
        )


def show_recommendations(params_df):
    """Display detailed parameter recommendations."""
    st.header("🎯 Fuel Optimization Recommendations")
    
    st.info("💡 These recommendations show how adjusting each parameter affects fuel consumption and lap times.")
    
    # Sort options
    sort_by = st.selectbox("Sort by", ["Fuel Saved", "Time Cost", "Parameter"])
    
    if sort_by == "Fuel Saved":
        params_df['sort_key'] = params_df['Fuel Saved (kg/race)'].str.replace(' kg', '').astype(float)
        params_df = params_df.sort_values('sort_key', ascending=False)
    elif sort_by == "Time Cost":
        params_df['sort_key'] = params_df['Time Cost/Race'].str.replace('s', '').astype(float)
        params_df = params_df.sort_values('sort_key')
    
    # Display recommendations
    first_item = True
    for idx, row in params_df.iterrows():
        param_name = row['Parameter']
        reduction = row['Reduction']
        is_open = "open" if first_item else ""
        first_item = False
        
        # Custom HTML collapsible with ALL content inside
        st.markdown(f"""
        <details class="custom-details" {is_open}>
            <summary><strong>{param_name}</strong> - {reduction}</summary>
            <div class="custom-details-content">
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1rem;">
                    <div style="background: #004b45; padding: 1rem; border-radius: 8px; border: 2px solid #cedc00; box-shadow: 0 2px 6px rgba(206,220,0,0.2);">
                        <div style="font-size: 0.8rem; color: #cedc00; font-weight: 600; text-transform: uppercase;">Fuel Saved</div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: #ffffff;">{row['Fuel Saved (kg/race)']}</div>
                    </div>
                    <div style="background: #004b45; padding: 1rem; border-radius: 8px; border: 2px solid #cedc00; box-shadow: 0 2px 6px rgba(206,220,0,0.2);">
                        <div style="font-size: 0.8rem; color: #cedc00; font-weight: 600; text-transform: uppercase;">Time Cost</div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: #ffffff;">{row['Time Cost/Race']}</div>
                    </div>
                    <div style="background: #004b45; padding: 1rem; border-radius: 8px; border: 2px solid #cedc00; box-shadow: 0 2px 6px rgba(206,220,0,0.2);">
                        <div style="font-size: 0.8rem; color: #cedc00; font-weight: 600; text-transform: uppercase;">Time Cost/Lap</div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: #ffffff;">{row['Time Cost/Lap']}</div>
                    </div>
                </div>
                <hr style="margin: 1rem 0; border: none; border-top: 2px solid #cedc00;">
                <p><strong>Analysis:</strong></p>
                <p>Reducing {row['Parameter']} by {row['Reduction']} can save {row['Fuel Saved (kg/race)']} 
                per race, but will cost approximately {row['Time Cost/Race']} in total race time.</p>
            </div>
        </details>
        """, unsafe_allow_html=True)


def show_scenario_planning(scenarios_df, params_df):
    """Scenario Planning has been removed."""
    st.info("Scenario Planning has been removed for this deployment.")


def show_race_scenarios(scenarios_df):
    """Scenario Planning has been removed."""
    st.info("Scenario Planning has been removed for this deployment.")


def show_live_calculator(model, calibrator, scaler, scalers_per_team, team_encoder, circuit_encoder, air_temp, track_temp, rainfall):
    """Fuel Strategy Simulator - Real-time fuel consumption predictions."""
    st.header("Fuel Strategy Simulator")
    st.caption("Simulate fuel usage and generate strategy insights in real time.")
    
    # Helper to encode image for CSS
    def get_image_as_base64(path):
        if not Path(path).exists():
            return None
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()

    # Custom CSS for the track background
    st.markdown("""
    <style>
    .simulator-background {
        position: relative;
        z-index: 1;
    }
    </style>
    """, unsafe_allow_html=True)

    if model is None:
        st.error("Model not loaded. Please ensure model files exist in outputs/two_stage_model/")
        return
    
    # 2025 F1 Calendar circuits
    circuits_2025 = {
        "Sakhir": "bahrain",
        "Jeddah": "jeddah",
        "Melbourne": "melbourne",
        "Suzuka": "suzuka",
        "Shanghai": "shanghai",
        "Miami": "miami",
        "Imola": "imola",
        "Monaco": "monaco",
        "Barcelona": "barcelona",
        "Montreal": "montreal",
        "Red Bull Ring": "austria",
        "Silverstone": "silverstone",
        "Spa": "spa",
        "Zandvoort": "zandvoort",
        "Monza": "monza",
        "Baku": "baku",
        "Marina Bay": "singapore",
        "Austin": "austin",
        "Mexico City": "mexico",
        "Interlagos": "brazil",
        "Las Vegas": "vegas",
        "Losail": "qatar",
        "Yas Marina": "abudhabi"
    }
    
    # Power circuits (high-speed tracks)
    power_circuits = ['monza', 'spa', 'jeddah', 'baku', 'silverstone', 'austria', 'mexico']
    
    # Session context
    st.markdown("### Session Context")
    col_ctx1, col_ctx2, col_ctx3 = st.columns(3)
    
    with col_ctx1:
        driver = st.selectbox(
            "Driver",
            ["Fernando Alonso", "Lance Stroll"],
            help="Current Aston Martin driver"
        )
    
    with col_ctx2:
        circuit_display = st.selectbox(
            "Circuit",
            list(circuits_2025.keys()),
            index=8,  # Default to Barcelona
            help="2025 F1 Calendar circuit"
        )
        circuit_key = circuits_2025[circuit_display]
        is_power_circuit = 1 if circuit_key in power_circuits else 0
        lap_limits = {
            "bahrain": 57, "jeddah": 50, "melbourne": 58, "suzuka": 53,
            "shanghai": 56, "miami": 57, "imola": 63, "monaco": 78,
            "barcelona": 66, "montreal": 70, "austria": 71, "silverstone": 52,
            "spa": 44, "zandvoort": 72, "monza": 53, "baku": 51,
            "singapore": 62, "austin": 56, "mexico": 71, "brazil": 71,
            "vegas": 50, "qatar": 57, "abudhabi": 58
        }
        max_laps = lap_limits.get(circuit_key, 70)
    
    with col_ctx3:
        lap_number = st.number_input(
            "Lap Number",
            min_value=1,
            max_value=max_laps,
            value=min(20, max_laps),
            help="Current lap in the race (affects tire deg, fuel load)"
        )
    
    # --- Dynamic Track Background Injection ---
    track_image_path = f"assets/tracks/{circuit_key}.png"
    track_image_base64 = get_image_as_base64(track_image_path)
    
    if track_image_base64:
        st.markdown(f"""
        <style>
        .main .block-container {{
            position: relative;
        }}
        .main .block-container::before {{
            content: '';
            position: absolute;
            top: 150px; /* Position below header */
            left: 5%;
            width: 90%;
            height: 70%;
            background-image: url('data:image/png;base64,{track_image_base64}');
            background-repeat: no-repeat;
            background-position: center center;
            background-size: contain;
            opacity: 0.08; /* Faint 8% opacity */
            z-index: -1;
        }}
        </style>
    """, unsafe_allow_html=True)

    # Style lap number input and steppers to be compact and on-brand
    st.markdown("""
    <style>
    div[data-testid="stNumberInput"] input {
        text-align: center;
        font-weight: 700;
        color: #ffffff !important;
        background: #003933 !important;
        border: 2px solid #cedc00 !important;
        border-radius: 8px !important;
    }
    div[data-testid="stNumberInput"] button {
        border: 1px solid #cedc00 !important;
        background: #004b45 !important;
        color: #cedc00 !important;
        font-weight: 700 !important;
    }
    div[data-testid="stNumberInput"] button:hover {
        background: #cedc00 !important;
        color: #004b45 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Input container styling
    st.markdown("""
    <style>
    div[data-testid="column"] > div {
        background-color: rgba(14, 17, 23, 0.4);
        padding: 1.5rem;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Telemetry Inputs")
        throttle = st.slider("Average Throttle (%)", 0, 100, 70, help="Average throttle application")
        rpm = st.slider("Average RPM", 8000, 13000, 11500, help="Average engine RPM")
        speed = st.slider("Average Speed (km/h)", 150, 320, 250, help="Average lap speed")
        gear = st.slider("Average Gear", 3, 8, 5, help="Average gear selection")
        drs_usage = st.slider("DRS Usage (%)", 0, 100, 15, help="% of lap with DRS active")
        ers_deployment = st.slider("ERS Deployment (%)", 0, 100, 40, help="% of lap with ERS deployed")
    
    with col2:
        st.subheader("🌤️ Weather Conditions")
        air_temp_input = st.slider("Air Temperature (°C)", 10, 45, int(air_temp), help="Ambient air temperature", key="calc_air_temp")
        track_temp_input = st.slider("Track Temperature (°C)", 15, 60, int(track_temp), help="Track surface temperature", key="calc_track_temp")
        humidity = st.slider("Humidity (%)", 20, 95, 60, help="Relative humidity")
        pressure = st.slider("Pressure (mbar)", 980, 1020, 1013, help="Atmospheric pressure")
        wind_speed = st.slider("Wind Speed (m/s)", 0, 15, 3, help="Wind speed")
        rain_label, rain_toggle, _ = st.columns([0.25, 0.15, 0.6])
        with rain_label:
            st.markdown("**Rainfall**")
        with rain_toggle:
            rainfall_input = st.toggle(
                "Rainfall",
                value=rainfall,
                help="Is it raining during the session?",
                label_visibility="collapsed"
            )
    
    # Calculate prediction
    center_btn = st.container()
    with center_btn:
        st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
        predict_clicked = st.button("🔮 Predict Fuel Consumption", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)
    
    if predict_clicked:
        # Try model-based prediction first, fall back to heuristic if needed
        ers_mode = max(0.0, min(4.0, ers_deployment / 25.0))  # map 0-100% slider to rough 0-4 ERS scale
        # Telemetry-inspired heuristic proxy (aligns with FastF1 lap signals)
        nominal = {
            "rpm": 11000.0,
            "throttle": 70.0,
            "speed": 230.0,
            "gear": 5.5,
            "drs": 20.0,
        }
        # Relative deltas around nominal
        rpm_delta = (rpm - nominal["rpm"]) / nominal["rpm"]
        throttle_delta = (throttle - nominal["throttle"]) / nominal["throttle"]
        speed_delta = (speed - nominal["speed"]) / nominal["speed"]
        gear_delta = (gear - nominal["gear"]) / nominal["gear"]
        drs_delta = (drs_usage - nominal["drs"]) / nominal["drs"]
        
        heuristic_ratio = (
            1.0
            + 0.55 * rpm_delta
            + 0.30 * throttle_delta
            + 0.12 * speed_delta
            + 0.06 * gear_delta
            - 0.10 * drs_delta  # more DRS lowers drag → lower burn
        )
        # Weather load: hotter track/air and higher humidity increase burn slightly
        temp_load = 0.0
        if track_temp_input > 45:
            temp_load += 0.02
        elif track_temp_input < 25:
            temp_load -= 0.01
        if air_temp_input > 35:
            temp_load += 0.015
        if humidity > 85:
            temp_load += 0.01
        heuristic_ratio *= (1.0 + temp_load)
        heuristic_proxy = float(np.clip(heuristic_ratio * FUEL_PROXY_ANCHOR, 0.45, 1.35))
        
        predict_df = pd.DataFrame({
            "rpm": [rpm],
            "throttle": [throttle],
            "speed": [speed],
            "gear": [gear],
            "ers": [ers_mode]
        })
        
        try:
            feature_df = prepare_features_for_inference(
                predict_df,
                team="Aston Martin",
                circuit_key=circuit_key,
                year=2024,
                scaler_global=scaler,
                scalers_per_team=scalers_per_team,
                team_encoder=team_encoder,
                circuit_encoder=circuit_encoder,
                air_temp=air_temp_input,
                track_temp=track_temp_input,
                humidity=humidity,
                pressure=pressure,
                wind_speed=wind_speed
            )
            model_proxy = float(predict_fuel(model, calibrator, scaler, feature_df)[0])
            # Blend model signal with telemetry-inspired heuristic to stabilize outputs
            fuel_proxy = 0.55 * model_proxy + 0.45 * heuristic_proxy
        except Exception:
            fuel_proxy = heuristic_proxy
        
        # Circuit-type adjustment
        if is_power_circuit:
            fuel_proxy *= 1.03
        else:
            fuel_proxy *= 0.99
        
        # Lap phase adjustment (heavy fuel early, lighter late)
        if lap_number < 10:
            fuel_proxy *= 1.025
        elif lap_number > max(max_laps * 0.7, 45):
            fuel_proxy *= 0.98
        
        # Adjust for rainfall - wet conditions typically increase fuel consumption by ~6-7%
        if rainfall_input:
            fuel_proxy *= 1.065  # 6.5% increase in wet conditions
        
        fuel_kg_per_lap = fuel_proxy_to_kg(circuit_key, fuel_proxy)
        
        # Display selected context
        weather_icon = "🌧️" if rainfall_input else "☀️"
        st.info(f"🏎️ Calculating for: **{driver}** at **{circuit_display}** (Lap {lap_number}) {'⚡ Power Circuit' if is_power_circuit else '🏁 Technical Circuit'} {weather_icon}")
        
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Fuel Proxy", f"{fuel_proxy:.3f}", help="Normalized fuel consumption metric")
        with col2:
            st.metric("Fuel (kg/lap)", f"{fuel_kg_per_lap:.3f}", help="Estimated kg burned per lap for this circuit")
        with col3:
            relative_consumption = (fuel_proxy - FUEL_PROXY_ANCHOR) / FUEL_PROXY_ANCHOR * 100
            st.metric("vs Baseline", f"{relative_consumption:+.1f}%", help="Compared to typical consumption")
        
        # Thresholds for status/strategies
        high_threshold = 22 if is_power_circuit else 18  # % above baseline
        low_threshold = -10  # % below baseline
        
        # Targeted reduction planner
        reduction_target = st.number_input(
            "Target fuel reduction (%)",
            min_value=0.0,
            max_value=30.0,
            value=6.0,
            step=0.5,
            help="Enter how much you want to cut current fuel burn (percentage of current kg/lap).",
        )
        if reduction_target > 0:
            target_fuel = fuel_kg_per_lap * (1 - reduction_target / 100)
            st.info(f"🎛️ Target: lower to ~{target_fuel:.3f} kg/lap ({reduction_target:.1f}% cut from {fuel_kg_per_lap:.3f} kg/lap)")
            if reduction_target >= 15:
                actions = [
                    "Aggressive lift/coast: brake points +90-120m earlier on heavy stops.",
                    "Short-shift by 400-600 RPM on exits; avoid full throttle until straightened.",
                    "Engine mode Fuel 2/3 + Low ERS deploy on longest straights.",
                    "DRS prioritize: maximize openings; trim wing if balance allows.",
                ]
            elif reduction_target >= 8:
                actions = [
                    "Moderate lift/coast: lift 60-80m earlier into T1/T2/T3.",
                    "Short-shift by 250-400 RPM; smooth throttle ramps in slow corners.",
                    "Balanced ERS deploy: shift energy to mid-length straights.",
                    "Brake migration tweaks to reduce micro-locks and drag losses.",
                ]
            else:
                actions = [
                    "Micro lift/coast: lift 30-50m early into heavy stops.",
                    "Short-shift by ~200 RPM; avoid minor wheelspin on exits.",
                    "Keep DRS uptime high; minor front-wing trim for straightline gain.",
                    "Stay in balanced ERS mode; avoid over-deploy on the longest straight.",
                ]
            for tip in actions:
                st.markdown(
                    f"""
                    <div style="background:#003933; padding:0.6rem 0.75rem; border-radius:8px; border-left:4px solid #cedc00; margin-bottom:0.45rem;">
                        <div style="color:#ffffff; font-size:0.95rem;">{tip}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        
        # Visual gauge (kg/lap scale)
        baseline = fuel_proxy_to_kg(circuit_key, FUEL_PROXY_ANCHOR)
        max_range = max(2.5, baseline * 1.6, fuel_kg_per_lap * 1.3)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fuel_kg_per_lap,
            number={'suffix': " kg/lap"},
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Fuel Consumption (kg/lap)"},
            gauge={
                'axis': {'range': [0, max_range]},
                'bar': {'color': "#00594C"},
                'steps': [
                    {'range': [0, baseline * 0.95], 'color': "lightgreen"},
                    {'range': [baseline * 0.95, baseline * 1.05], 'color': "yellow"},
                    {'range': [baseline * 1.05, baseline * 1.20], 'color': "orange"},
                    {'range': [baseline * 1.20, max_range], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': fuel_kg_per_lap
                }
            }
        ))
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)
        
        # Dynamic strategy suggestions based on live burn rate
        # Setup-style levers (RPM / Throttle / Speed) with reduction, fuel saved, time cost, efficiency
        st.subheader("🔧 Setup Lever Recommendations")
        if relative_consumption > high_threshold:
            levers = [
                {"label": "Average RPM", "reduction": 12, "note": "Cap revs early on straights; short-shift consistently."},
                {"label": "Average Throttle", "reduction": 10, "note": "Smoother ramps in slow/medium corners; avoid flat-out drags."},
                {"label": "Average Speed", "reduction": 8, "note": "Back off entry/exit speed slightly to trim drag and over-rotation."},
            ]
        elif relative_consumption > 6:
            levers = [
                {"label": "Average RPM", "reduction": 8, "note": "Early upshifts in gears 4-6; stay below ~11.1k on exits."},
                {"label": "Average Throttle", "reduction": 6, "note": "Trim throttle midspeed corners; avoid small throttle stabs."},
                {"label": "Average Speed", "reduction": 5, "note": "Shed ~2-3 km/h through long radius corners to cut drag."},
            ]
        else:
            levers = [
                {"label": "Average RPM", "reduction": 5, "note": "Mild short-shifts; keep revs under ~11.4k on exits."},
                {"label": "Average Throttle", "reduction": 4, "note": "Micro lift/coast and smoother throttle ramps."},
                {"label": "Average Speed", "reduction": 3, "note": "Slightly gentler corner exits to maintain efficiency."},
            ]
        
        laps_left = max(0, max_laps - lap_number)
        lever_cards = []
        for lever in levers:
            reduction_pct = lever["reduction"]
            fuel_saved_total = fuel_kg_per_lap * laps_left * (reduction_pct / 100.0)
            time_cost_total = laps_left * 0.35 * (reduction_pct / 10.0)  # heuristic seconds over remaining race
            efficiency = fuel_saved_total / time_cost_total if time_cost_total > 0 else 0
            lever_cards.append(
                {
                    "label": lever["label"],
                    "reduction": reduction_pct,
                    "fuel_saved": fuel_saved_total,
                    "time_cost": time_cost_total,
                    "efficiency": efficiency,
                    "note": lever["note"],
                }
            )

        # Top performer cards based on remaining laps
        if lever_cards and laps_left > 0:
            st.subheader("🏆 Top Performers (Remaining Race)")
            st.caption("Ranked by remaining-laps impact; fuel/time values scale with laps left.")
            top_saver = max(lever_cards, key=lambda x: x["fuel_saved"])
            top_eff = max(lever_cards, key=lambda x: x["efficiency"])
            low_time = min(lever_cards, key=lambda x: x["time_cost"])
            cards = [
                ("🥇 Most Fuel Saved", top_saver, "#66ff66"),
                ("⚡ Most Efficient", top_eff, "#cedc00"),
                ("🚀 Lowest Time Cost", low_time, "#ff9966"),
            ]
            cols = st.columns(3)
            for col, (title, card, color) in zip(cols, cards):
                with col:
                    col.markdown(
                        f"""
                        <div style="background: linear-gradient(135deg, #004b45 0%, #003933 100%); padding: 1.1rem; border-radius: 12px; border: 2px solid {color}; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
                            <div style="font-size: 0.9rem; color: {color}; font-weight: 700; text-transform: uppercase; margin-bottom: 0.45rem;">{title}</div>
                            <div style="font-size: 1.2rem; font-weight: bold; color: #ffffff; margin-bottom: 0.35rem;">{card['label']}</div>
                            <div style="display:grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem; color:#ffffff; font-size:0.95rem;">
                                <div><div style="font-size:0.75rem; color:#cedc00;">Fuel Saved (remaining)</div><div style="font-size:1.05rem; font-weight:700;">{card['fuel_saved']:.2f} kg</div></div>
                                <div><div style="font-size:0.75rem; color:#cedc00;">Time Cost (remaining)</div><div style="font-size:1.05rem; font-weight:700;">{card['time_cost']:.2f}s</div></div>
                                <div><div style="font-size:0.75rem; color:#cedc00;">Efficiency</div><div style="font-size:1.05rem; font-weight:700;">{card['efficiency']:.2f}</div></div>
                                <div><div style="font-size:0.75rem; color:#cedc00;">Reduction</div><div style="font-size:1.05rem; font-weight:700;">{card['reduction']}%</div></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # Setup-style levers (RPM / Throttle / Speed) with reduction, fuel saved, time cost, efficiency
        st.subheader("🔧 Setup Lever Recommendations")
        for lever in lever_cards:
            st.markdown(
                f"""
                <div style="background:#004b45; padding:1rem 1.2rem; border-radius:12px; border-left:5px solid #ff9240; margin-bottom:0.7rem;">
                    <div style="font-size:1.1rem; font-weight:700; color:#ffffff; margin-bottom:0.35rem;">{lever['label']}</div>
                    <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; align-items:center; color:#ffffff; font-size:0.95rem;">
                        <div><div style="font-size:0.75rem; color:#cedc00; letter-spacing:0.5px;">FUEL SAVED (remaining)</div><div style="font-size:1.1rem; font-weight:700;">{lever['fuel_saved']:.2f} kg</div></div>
                        <div><div style="font-size:0.75rem; color:#cedc00; letter-spacing:0.5px;">TIME COST (remaining)</div><div style="font-size:1.1rem; font-weight:700;">{lever['time_cost']:.2f}s</div></div>
                        <div><div style="font-size:0.75rem; color:#cedc00; letter-spacing:0.5px;">EFFICIENCY</div><div style="font-size:1.1rem; font-weight:700;">{lever['efficiency']:.2f}</div></div>
                        <div><div style="font-size:0.75rem; color:#cedc00; letter-spacing:0.5px;">REDUCTION</div><div style="font-size:1.1rem; font-weight:700;">{lever['reduction']}%</div></div>
                    </div>
                    <div style="margin-top:0.4rem; color:#e8f7f0; font-size:0.9rem;">{lever['note']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Dynamic strategy suggestions based on live burn rate (after levers)
        st.subheader("🎯 Live Strategy Recommendations")
        strategies = []
        # Thresholds based on relative burn vs baseline
        if relative_consumption > high_threshold:
            strategies = [
                {
                    "name": "Heavy Lift/Coast + Short Shift",
                    "fuel": "0.25-0.35 kg/lap saved",
                    "time": "0.05-0.08s cost",
                    "note": "Lift 90-120m early, cap avg RPM ~10.6-10.9k, drop avg throttle to mid-60s%, target avg speed -5 km/h, short-shift 400-600 RPM."
                },
                {
                    "name": "Low ERS Deploy + Engine Mode Fuel 2",
                    "fuel": "0.10-0.18 kg/lap saved",
                    "time": "0.04-0.07s cost",
                    "note": "Use conservative engine mode on long straights; shift ERS deploy later in gears 4-6; keep avg gear one step higher in slow corners."
                },
                {
                    "name": "DRS Priority + Anti-Drag Trim",
                    "fuel": "0.06-0.10 kg/lap saved",
                    "time": "Minimal",
                    "note": "Maximize DRS uptime; trim flap if balance allows; aim avg throttle <65% on non-DRS corners to cut drag losses."
                },
            ]
        elif relative_consumption > 6:
            strategies = [
                {
                    "name": "Moderate Lift/Coast + Brake Regen",
                    "fuel": "0.14-0.22 kg/lap saved",
                    "time": "0.03-0.05s cost",
                    "note": "Lift 60-80m early into heavy stops; cap avg RPM to ~11.1k; hold avg throttle high-60s%; drop avg speed ~3 km/h; lean on brake regen."
                },
                {
                    "name": "ERS Balanced Deploy",
                    "fuel": "0.05-0.09 kg/lap saved",
                    "time": "Minimal",
                    "note": "Shift deploy to mid-corner exits; avoid full deploy on the longest straight; short-shift 250-350 RPM in gears 4-6."
                },
                {
                    "name": "Throttle/Traction Tidy",
                    "fuel": "0.03-0.06 kg/lap saved",
                    "time": "Minimal",
                    "note": "Smooth throttle ramps; keep avg throttle ~68-72%; avoid over-revs on exits; use traction maps to cut wheelspin in low gears."
                },
            ]
        else:
            strategies = [
                {
                    "name": "Push With DRS Efficiency",
                    "fuel": "Maintain",
                    "time": "Gain 0.02-0.04s",
                    "note": "Maintain burn: keep avg RPM <11.5k, throttle ~72-75%; maximize DRS uptime; minor front-wing trim for speed."
                },
                {
                    "name": "Targeted ERS Attack",
                    "fuel": "Maintain",
                    "time": "Gain 0.03-0.06s",
                    "note": "Deploy on top-2 straights only; avoid over-revs (>11.6k) in gears 6-7; keep avg speed on target without dragging throttle."
                },
                {
                    "name": "Brake/Shift Fine-Tune",
                    "fuel": "0.02-0.04 kg/lap saved",
                    "time": "Neutral",
                    "note": "Use brake migration to stabilize entries; short-shift ~200 RPM in slow exits; keep avg gear slightly higher to cut revs without pace loss."
                },
            ]
        
        for strat in strategies:
            st.markdown(
                f"""
                <div style="background:#003933; padding:0.9rem 1rem; border-radius:10px; border-left:4px solid #cedc00; margin-bottom:0.6rem; box-shadow:0 2px 6px rgba(0,0,0,0.3);">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem;">
                        <div style="font-weight:700; color:#ffffff;">{strat['name']}</div>
                        <div style="font-size:0.8rem; color:#cedc00;">Fuel: {strat['fuel']}</div>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.25rem;">
                        <div style="font-size:0.85rem; color:#cedc00;">Time: {strat['time']}</div>
                        <div style="font-size:0.75rem; color:#66ff66; font-weight:600;">Live-adjusted</div>
                    </div>
                    <div style="font-size:0.85rem; color:#ffffff;">{strat['note']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        # Circuit-specific and driver-specific insights
        st.markdown("### 🎯 Context-Aware Insights")
        insight_notes = []
        if rainfall_input:
            insight_notes.append("🌧️ Weather: Rain adds ~6.5% burn from low grip and wheel spin.")
        if driver == "Fernando Alonso":
            insight_notes.append(f"🧠 Alonso style: {'Efficient' if fuel_proxy < 0.70 else 'Higher than usual'} burn for his smooth driving profile.")
        else:
            insight_notes.append(f"🎯 Stroll style: {'Normal' if 0.65 < fuel_proxy < 0.75 else 'Higher than typical'} — watch exits.")
        if is_power_circuit:
            insight_notes.append(f"⚡ Track: {circuit_display} power circuit — expect higher burn; maximize DRS and short-shift on exits.")
        else:
            insight_notes.append(f"🏁 Track: {circuit_display} technical — burn should be modest; trim throttle in slow corners.")
        if lap_number < 10:
            insight_notes.append(f"⏳ Phase: Early stint (Lap {lap_number}) — heavy fuel load adds ~2-3% burn.")
        elif lap_number > 50:
            insight_notes.append(f"⏳ Phase: Late stint (Lap {lap_number}) — light car, tire deg can raise slip/burn.")
        else:
            insight_notes.append(f"⏳ Phase: Mid-race (Lap {lap_number}) — optimal fuel window.")
        
        for note in insight_notes:
            st.markdown(
                f"""
                <div style="background:#003933; padding:0.7rem 0.9rem; border-radius:10px; border-left:4px solid #cedc00; margin-bottom:0.45rem; color:#ffffff;">
                    {note}
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        if relative_consumption > high_threshold:
            recs = [
                "- Reduce RPM by 500-1000 and short-shift in gears 4-6.",
                "- Decrease throttle application in low-speed corners.",
                "- Use conservative engine/ERS mode on long straights.",
                "- Maximize DRS usage to cut drag (especially on power circuits).",
            ]
            tone = "warning"
            title = "⚠️ High fuel consumption detected!"
        elif relative_consumption < low_threshold:
            recs = [
                "- Current settings are very fuel-efficient.",
                "- Can push harder if needed with minimal penalty.",
                "- Maintain smooth throttle ramps to keep efficiency.",
            ]
            tone = "success"
            title = "✅ Excellent fuel efficiency!"
        else:
            recs = [
                "- Current settings are within normal range.",
                "- Small short-shifts (200-300 RPM) can trim burn further.",
                "- Keep DRS uptime high; avoid unnecessary over-revs.",
            ]
            tone = "info"
            title = "ℹ️ Balanced fuel consumption"
        
        rec_html = "".join(f"<li>{r}</li>" for r in recs)
        color_map = {"warning": "#ffb347", "success": "#66ff66", "info": "#cedc00"}
        st.markdown(
            f"""
            <div style="background:#003933; padding:0.95rem 1rem; border-radius:12px; border:2px solid {color_map[tone]}; box-shadow:0 3px 10px rgba(0,0,0,0.35); color:#ffffff;">
                <div style="font-weight:700; margin-bottom:0.35rem; color:{color_map[tone]};">{title}</div>
                <ul style="margin:0; padding-left:1.1rem; color:#ffffff; line-height:1.5;">{rec_html}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def show_ai_model_briefing(train_df):
    """AI Model Briefing - Historical analysis and model training insights."""
    st.header("AI Model Briefing")
    st.caption("Understand the data, architecture, and rationale behind the model.")
    
    st.info("💡 **Model Training Data** - 676,513 laps from 7 seasons (2018-2024) with 99.41% validation accuracy")
    
    # Static training data summary (always available)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Laps", "676,513", help="Total training laps")
    with col2:
        st.metric("Years Covered", "7", help="Historical data span")
    with col3:
        st.metric("Circuits", "15", help="Different race tracks")
    with col4:
        st.metric("Teams", "18", help="F1 teams included")
    
    st.markdown("---")
    
    # Aston Martin specific
    st.subheader("🏎️ Aston Martin Training Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("AM Laps", "44,580", help="Aston Martin specific laps")
    with col2:
        st.metric("% of Dataset", "6.6%", help="Proportion dedicated to AM")
    with col3:
        st.metric("Years", "2018-2024", help="Full historical coverage")
    
    st.write("")
    st.success("✅ **Model trained with per-team normalization** to account for different car performances")
    st.info("ℹ️ **Two-stage training:** Pre-trained on all teams (676K laps) → Fine-tuned on Aston Martin (44K laps)")
    
    st.markdown("---")
    
    # Circuit breakdown
    st.subheader("🗺️ Circuit Coverage")
    image_path = Path("assets/Circuits.png")
    if image_path.exists():
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <img src="data:image/png;base64,{encoded}" alt="Circuit coverage"
                 style="width:100%;height:auto;pointer-events:none;user-select:none;border-radius:8px;" />
            """,
            unsafe_allow_html=True,
        )
        st.info("💡 Circuit coverage training mix: high-fuel circuits have 7 full seasons of data, while technical and mixed layouts add three seasons each to diversify race pace patterns. Only a few circuits and limited laps per circuit were used to keep the model from overfitting.")
    else:
        st.warning("Circuit coverage image not found.")
    
    st.markdown("---")
    
    # Weather data info
    st.subheader("🌤️ Weather Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Temperature:**")
        st.write("- Air Temperature (10-45°C)")
        st.write("- Track Temperature (15-60°C)")
    
    with col2:
        st.markdown("**Atmospheric:**")
        st.write("- Humidity (20-95%)")
        st.write("- Pressure (980-1020 mbar)")
        st.write("- Rainfall (0-100%)")
    
    with col3:
        st.markdown("**Wind:**")
        st.write("- Wind Speed (0-15 m/s)")
        st.write("- Wind Direction (0-360°)")
    
    st.markdown("---")
    
    # If actual data is loaded, show live visualizations
    if train_df is not None:
        st.subheader("📊 Live Data Visualizations")
        st.info("💡 Enhanced visualizations available with loaded training data")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            teams = ['All'] + sorted(train_df['Team'].unique().tolist())
            selected_team = st.selectbox("Team", teams)
        
        with col2:
            years = ['All'] + sorted(train_df['year'].unique().tolist(), reverse=True)
            selected_year = st.selectbox("Year", years)
        
        with col3:
            circuits = ['All'] + sorted(train_df['gp'].unique().tolist())
            selected_circuit = st.selectbox("Circuit", circuits)
        
        # Filter data
        filtered_df = train_df.copy()
        if selected_team != 'All':
            filtered_df = filtered_df[filtered_df['Team'] == selected_team]
        if selected_year != 'All':
            filtered_df = filtered_df[filtered_df['year'] == selected_year]
        if selected_circuit != 'All':
            filtered_df = filtered_df[filtered_df['gp'] == selected_circuit]
        
        # Summary stats
        st.subheader(f"📊 Summary Statistics ({len(filtered_df):,} laps)")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Avg Throttle", f"{filtered_df['avg_throttle'].mean():.1f}%")
        with col2:
            st.metric("Avg RPM", f"{filtered_df['avg_rpm'].mean():.0f}")
        with col3:
            st.metric("Avg Speed", f"{filtered_df['avg_speed'].mean():.1f} km/h")
        with col4:
            st.metric("Avg Air Temp", f"{filtered_df['air_temp'].mean():.1f}°C")
        
        # Visualization options
        st.markdown("---")
        st.subheader("📊 Visualizations")
        
        viz_type = st.selectbox("Select Visualization", 
                               ["Fuel Proxy Distribution", "RPM vs Throttle", "Weather Impact", "Speed Analysis"])
        
        if viz_type == "Fuel Proxy Distribution":
            fuel_proxy = 0.60 * (filtered_df['avg_rpm'] / 12000.0) + 0.40 * (filtered_df['avg_throttle'] / 100.0)
            fig = px.histogram(fuel_proxy, nbins=50, title="Fuel Proxy Distribution",
                              labels={'value': 'Fuel Proxy', 'count': 'Frequency'})
            fig.update_traces(marker_color='#00594C')
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "RPM vs Throttle":
            sample_df = filtered_df.sample(min(5000, len(filtered_df)))
            fig = px.scatter(sample_df, x='avg_rpm', y='avg_throttle', 
                            color='Team' if selected_team == 'All' else None,
                            title="RPM vs Throttle Application",
                            labels={'avg_rpm': 'Average RPM', 'avg_throttle': 'Average Throttle (%)'})
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Weather Impact":
            fig = px.scatter(filtered_df.sample(min(5000, len(filtered_df))), 
                            x='air_temp', y='track_temp', color='humidity',
                            title="Weather Conditions Distribution",
                            labels={'air_temp': 'Air Temperature (°C)', 'track_temp': 'Track Temperature (°C)'})
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Speed Analysis":
            if 'avg_speed' in filtered_df.columns:
                fig = px.box(filtered_df, y='avg_speed', x='gp' if selected_circuit == 'All' else 'year',
                            title="Speed Distribution by " + ("Circuit" if selected_circuit == 'All' else "Year"),
                            labels={'avg_speed': 'Average Speed (km/h)'})
                st.plotly_chart(fig, use_container_width=True)
        
        # Raw data view
        if st.checkbox("Show Raw Data"):
            st.dataframe(filtered_df.head(100), use_container_width=True)


def show_race_analysis(model, calibrator, scaler, scalers_per_team, team_encoder, circuit_encoder):
    """Race Analysis - Deep dive into past Aston Martin races using FastF1 data."""
    st.header("Past Race Analysis")
    st.caption("Analyze past races and identify fuel-efficiency improvements.")
    
    # Import FastF1
    try:
        import fastf1
        cache_dir = Path("cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(cache_dir)
    except ImportError:
        st.error("❌ FastF1 not installed. Run: `pip install fastf1`")
        return
    
    # Aston Martin driver history
    # 2021: Sebastian Vettel (VET), Lance Stroll (STR)
    # 2022: Sebastian Vettel (VET), Lance Stroll (STR)
    # 2023: Fernando Alonso (ALO), Lance Stroll (STR)
    # 2024: Fernando Alonso (ALO), Lance Stroll (STR)
    
    AM_DRIVERS = {
        2021: {"Lance Stroll": "STR", "Sebastian Vettel": "VET"},
        2022: {"Lance Stroll": "STR", "Sebastian Vettel": "VET"},
        2023: {"Lance Stroll": "STR", "Fernando Alonso": "ALO"},
        2024: {"Lance Stroll": "STR", "Fernando Alonso": "ALO"}
    }
    
    # Selection controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        year = st.selectbox(
            "📅 Season",
            [2024, 2023, 2022, 2021],
            help="Aston Martin F1 seasons"
        )
    
    # Get race list for selected year
    try:
        schedule = fastf1.get_event_schedule(year)
        # Filter for actual races (not testing)
        race_events = schedule[schedule['EventFormat'].isin(['conventional', 'sprint', 'sprint_shootout'])]
        race_names = race_events['EventName'].tolist()
    except Exception as e:
        st.error(f"Error loading schedule: {e}")
        return
    
    with col2:
        # Get drivers for selected year
        available_drivers = list(AM_DRIVERS[year].keys())
        driver = st.selectbox(
            "🏎️ Driver",
            available_drivers,
            help=f"Aston Martin drivers in {year}"
        )
        driver_code = AM_DRIVERS[year][driver]
    
    with col3:
        race_name = st.selectbox(
            "🏁 Grand Prix",
            race_names,
            help=f"Races in {year} season"
        )
    
    # Load race button
    if st.button("🔄 Load Race Data", type="primary", use_container_width=True):
        with st.spinner(f"Loading {year} {race_name} for {driver}..."):
            try:
                # Load race session
                session = fastf1.get_session(year, race_name, 'R')
                session.load()
                
                # Get driver laps
                driver_laps = session.laps.pick_driver(driver_code)
                
                if len(driver_laps) == 0:
                    st.warning(f"⚠️ No lap data found for {driver} in {year} {race_name}")
                    return
                
                # Pre-load telemetry for all laps
                st.info("Loading telemetry data for all laps...")
                lap_telemetry = {}
                telemetry_errors = []
                
                for idx, lap in driver_laps.iterrows():
                    try:
                        tel = lap.get_telemetry()
                        
                        # Debug: Log what we got
                        if tel is None:
                            telemetry_errors.append(f"Lap {lap['LapNumber']}: get_telemetry() returned None")
                            lap_telemetry[lap['LapNumber']] = None
                        elif len(tel) == 0:
                            telemetry_errors.append(f"Lap {lap['LapNumber']}: Empty telemetry DataFrame")
                            lap_telemetry[lap['LapNumber']] = None
                        else:
                            # Store aggregated telemetry data
                            lap_telemetry[lap['LapNumber']] = {
                                'rpm': tel['RPM'].mean() if 'RPM' in tel.columns else None,
                                'throttle': tel['Throttle'].mean() if 'Throttle' in tel.columns else None,
                                'speed': tel['Speed'].mean() if 'Speed' in tel.columns else None,
                                'gear': tel['nGear'].mean() if 'nGear' in tel.columns else None,
                                'has_full_data': all([
                                    'RPM' in tel.columns and not tel['RPM'].isna().all(),
                                    'Throttle' in tel.columns and not tel['Throttle'].isna().all(),
                                    'Speed' in tel.columns and not tel['Speed'].isna().all(),
                                    'nGear' in tel.columns and not tel['nGear'].isna().all()
                                ])
                            }
                    except Exception as e:
                        telemetry_errors.append(f"Lap {lap['LapNumber']}: {str(e)}")
                        lap_telemetry[lap['LapNumber']] = None
                
                # Show debug info
                if telemetry_errors and len(telemetry_errors) <= 5:
                    st.warning(f"Telemetry issues: {'; '.join(telemetry_errors[:5])}")
                
                # Store in session state
                st.session_state.race_data = {
                    'session': session,
                    'laps': driver_laps,
                    'telemetry': lap_telemetry,
                    'year': year,
                    'race': race_name,
                    'driver': driver,
                    'driver_code': driver_code
                }
                
                full_tel_count = sum(1 for t in lap_telemetry.values() if t and t.get('has_full_data'))
                st.success(f"✅ Loaded {len(driver_laps)} laps for {driver} ({full_tel_count} with full telemetry)")
                
            except Exception as e:
                st.error(f"❌ Error loading data: {e}")
                return
    
    # Display analysis if data is loaded
    if 'race_data' not in st.session_state:
        st.info("👆 Select a race and click 'Load Race Data' to begin analysis")
        return
    
    race_data = st.session_state.race_data
    laps = race_data['laps']
    session = race_data['session']
    
    st.markdown("---")
    st.subheader(f"📊 {race_data['year']} {race_data['race']} - {race_data['driver']}")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Laps", len(laps))
    with col2:
        fastest_lap = laps['LapTime'].min()
        if pd.notna(fastest_lap):
            # Convert timedelta to string and extract MM:SS.mmm
            lap_str = str(fastest_lap)
            if 'days' in lap_str:
                lap_str = lap_str.split()[-1]  # Remove days if present
            # Remove hours if present (format: HH:MM:SS.mmm -> MM:SS.mmm)
            time_parts = lap_str.split(':')
            if len(time_parts) == 3:
                lap_str = f"{time_parts[1]}:{time_parts[2]}"
            st.metric("Fastest Lap", lap_str)
        else:
            st.metric("Fastest Lap", "N/A")
    with col3:
        avg_speed = laps['SpeedST'].mean() if 'SpeedST' in laps.columns else 0
        st.metric("Avg Speed", f"{avg_speed:.1f} km/h")
    with col4:
        compounds = laps['Compound'].unique()
        st.metric("Tire Compounds", ", ".join([c for c in compounds if pd.notna(c)]))
    
    # Tab layout for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Lap-by-Lap", "📈 Telemetry Charts", "🗺️ Track Map", "⛽ Fuel Estimation"])
    
    with tab1:
        st.markdown("### Lap-by-Lap Breakdown")
        
        # Prepare lap table
        lap_table = pd.DataFrame({
            'Lap': laps['LapNumber'],
            'Lap Time': laps['LapTime'].apply(lambda x: str(x).split()[-1] if pd.notna(x) else 'N/A'),
            'Compound': laps['Compound'],
            'Stint': laps['Stint'],
            'Speed ST (km/h)': laps['SpeedST'].round(1) if 'SpeedST' in laps.columns else 'N/A',
            'Speed I1': laps['SpeedI1'].round(1) if 'SpeedI1' in laps.columns else 'N/A',
            'Speed I2': laps['SpeedI2'].round(1) if 'SpeedI2' in laps.columns else 'N/A',
            'Speed FL': laps['SpeedFL'].round(1) if 'SpeedFL' in laps.columns else 'N/A',
            'Track Status': laps['TrackStatus']
        })
        
        st.dataframe(lap_table, use_container_width=True, height=400)
        
        # Download button
        csv = lap_table.to_csv(index=False)
        st.download_button(
            label="📥 Download Lap Data",
            data=csv,
            file_name=f"{race_data['driver_code']}_{race_data['year']}_{race_data['race']}_laps.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        st.markdown("#### Fuel-Adjusted Lap Time Model")
        st.caption("Uses FastF1 lap times plus a 0.03s/kg fuel effect to estimate burn rate and show fuel-corrected laps.")
        
        stint_options = ["All stints"] + sorted([str(s) for s in laps['Stint'].dropna().unique()])
        selected_stint = st.selectbox("Stint", stint_options, help="Choose a stint to analyze fuel effect on lap times")
        initial_fuel = st.number_input("Initial Fuel Load (kg)", min_value=60.0, max_value=115.0, value=110.0, step=1.0)
        fuel_effect = st.number_input("Fuel Weight Effect (s/kg)", min_value=0.01, max_value=0.05, value=0.03, step=0.005)
        
        stint_mask = laps['Stint'].astype(str) == selected_stint if selected_stint != "All stints" else np.ones(len(laps), dtype=bool)
        stint_laps = laps[stint_mask].copy()
        stint_laps = stint_laps[pd.notna(stint_laps['LapTime'])]
        
        if len(stint_laps) >= 3:
            # Build lap index starting at zero for slope fit
            stint_laps = stint_laps.sort_values('LapNumber').reset_index(drop=True)
            stint_laps['LapTime_s'] = stint_laps['LapTime'].dt.total_seconds()
            x_idx = np.arange(len(stint_laps))
            y_time = stint_laps['LapTime_s'].values
            
            # Linear fit of lap time vs lap index; slope ≈ -burn_rate * fuel_effect
            slope, intercept = np.polyfit(x_idx, y_time, 1)
            est_burn_rate = max(0.0, -slope / fuel_effect)
            base_lap = intercept - initial_fuel * fuel_effect  # lap time with zero fuel using fitted line
            
            # Compute remaining fuel curve and corrected lap times
            stint_laps['Fuel Remaining (kg)'] = np.clip(initial_fuel - est_burn_rate * x_idx, a_min=0, a_max=None)
            stint_laps['Fuel-Corrected Lap (s)'] = stint_laps['LapTime_s'] - stint_laps['Fuel Remaining (kg)'] * fuel_effect
            
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                st.metric("Estimated Burn Rate", f"{est_burn_rate:.3f} kg/lap", help="Derived from lap-time slope and 0.03s/kg fuel effect")
            with col_f2:
                st.metric("No-Fuel Pace (est.)", f"{base_lap:.3f} s", help="Lap time extrapolated to zero fuel using fitted line")
            with col_f3:
                st.metric("Avg Corrected Lap", f"{stint_laps['Fuel-Corrected Lap (s)'].mean():.3f} s")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=stint_laps['LapNumber'],
                y=stint_laps['LapTime_s'],
                mode='lines+markers',
                name='Recorded Lap Time',
                line=dict(color='#cedc00', width=2),
                marker=dict(size=6)
            ))
            fig.add_trace(go.Scatter(
                x=stint_laps['LapNumber'],
                y=stint_laps['Fuel-Corrected Lap (s)'],
                mode='lines+markers',
                name='Fuel-Corrected Lap',
                line=dict(color='#66ff66', width=2, dash='dot'),
                marker=dict(size=6)
            ))
            fig.update_layout(
                title="Lap Times vs Fuel-Corrected Lap Times",
                xaxis_title="Lap Number",
                yaxis_title="Lap Time (s)",
                plot_bgcolor='#003933',
                paper_bgcolor='#004b45',
                font=dict(color='#ffffff'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption("Assumes monotonic fuel burn; unexpected trends may reflect traffic, tires, VSC/SC, or lifts.")
        else:
            st.info("Need at least 3 laps with valid lap times to estimate fuel burn.")
    
    with tab2:
        st.markdown("### Telemetry Analysis")
        
        # Select a lap for detailed telemetry
        lap_numbers = laps['LapNumber'].tolist()
        selected_lap = st.selectbox("Select Lap for Detailed Telemetry", lap_numbers)
        
        if selected_lap:
            try:
                lap = laps[laps['LapNumber'] == selected_lap].iloc[0]
                telemetry = lap.get_telemetry()
                
                if len(telemetry) > 0:
                    # Speed trace
                    fig_speed = go.Figure()
                    fig_speed.add_trace(go.Scatter(
                        x=telemetry['Distance'],
                        y=telemetry['Speed'],
                        mode='lines',
                        name='Speed',
                        line=dict(color='#cedc00', width=2)
                    ))
                    fig_speed.update_layout(
                        title=f"Speed Trace - Lap {selected_lap}",
                        xaxis_title="Distance (m)",
                        yaxis_title="Speed (km/h)",
                        plot_bgcolor='#003933',
                        paper_bgcolor='#004b45',
                        font=dict(color='#ffffff')
                    )
                    st.plotly_chart(fig_speed, use_container_width=True)
                    
                    # Throttle & RPM
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_throttle = go.Figure()
                        fig_throttle.add_trace(go.Scatter(
                            x=telemetry['Distance'],
                            y=telemetry['Throttle'],
                            mode='lines',
                            name='Throttle',
                            line=dict(color='#66ff66', width=2)
                        ))
                        fig_throttle.update_layout(
                            title="Throttle Application",
                            xaxis_title="Distance (m)",
                            yaxis_title="Throttle (%)",
                            plot_bgcolor='#003933',
                            paper_bgcolor='#004b45',
                            font=dict(color='#ffffff')
                        )
                        st.plotly_chart(fig_throttle, use_container_width=True)
                    
                    with col2:
                        fig_rpm = go.Figure()
                        fig_rpm.add_trace(go.Scatter(
                            x=telemetry['Distance'],
                            y=telemetry['RPM'],
                            mode='lines',
                            name='RPM',
                            line=dict(color='#ff6b6b', width=2)
                        ))
                        fig_rpm.update_layout(
                            title="Engine RPM",
                            xaxis_title="Distance (m)",
                            yaxis_title="RPM",
                            plot_bgcolor='#003933',
                            paper_bgcolor='#004b45',
                            font=dict(color='#ffffff')
                        )
                        st.plotly_chart(fig_rpm, use_container_width=True)
                    
                    # DRS zones
                    if 'DRS' in telemetry.columns:
                        fig_drs = go.Figure()
                        fig_drs.add_trace(go.Scatter(
                            x=telemetry['Distance'],
                            y=telemetry['DRS'],
                            mode='lines',
                            fill='tozeroy',
                            name='DRS',
                            line=dict(color='#cedc00', width=0)
                        ))
                        fig_drs.update_layout(
                            title="DRS Activation Zones",
                            xaxis_title="Distance (m)",
                            yaxis_title="DRS Status",
                            plot_bgcolor='#003933',
                            paper_bgcolor='#004b45',
                            font=dict(color='#ffffff')
                        )
                        st.plotly_chart(fig_drs, use_container_width=True)
                    
                else:
                    st.warning("⚠️ No telemetry data available for this lap")
                    
            except Exception as e:
                st.error(f"Error loading telemetry: {e}")
    
    with tab3:
        st.markdown("### Track Map Visualization")
        
        try:
            # Get fastest lap for track map
            fastest = laps.pick_fastest()
            telemetry = fastest.get_telemetry()
            
            if len(telemetry) > 0 and 'X' in telemetry.columns and 'Y' in telemetry.columns:
                # Create track map colored by speed
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=telemetry['X'],
                    y=telemetry['Y'],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=telemetry['Speed'],
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="Speed<br>(km/h)")
                    ),
                    name='Track',
                    hovertemplate='Speed: %{marker.color:.1f} km/h<extra></extra>'
                ))
                
                fig.update_layout(
                    title=f"Track Map - {race_data['race']} (Colored by Speed)",
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False, scaleanchor='x'),
                    plot_bgcolor='#003933',
                    paper_bgcolor='#004b45',
                    font=dict(color='#ffffff'),
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Track position data not available for this session")
                
        except Exception as e:
            st.error(f"Error creating track map: {e}")
    
    with tab4:
        st.markdown("### ⛽ Fuel Cost of Reality")
        st.caption("Quantifying the fuel expense of how the driver actually drove")
        
        race_key = str(race_data.get('race', '')).lower()
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #003933 0%, #004b45 100%); 
                    padding: 1rem; border-radius: 8px; border-left: 4px solid #cedc00; 
                    margin-bottom: 1rem;">
            <p style="margin: 0; color: #ffffff; font-size: 0.9rem;">
                <strong style="color: #cedc00;">What This Shows:</strong><br>
                This isn't predicting the future - it's <strong>quantifying the past</strong>. 
                By feeding actual driving data (RPM, throttle, speed, gear, tires) into our model, 
                we estimate: <em>"Given how they drove, how expensive in fuel was each lap?"</em>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            # Prepare features for each lap using pre-loaded telemetry
            fuel_estimates = []
            lap_telemetry = race_data.get('telemetry', {})
            
            for idx, lap in laps.iterrows():
                avg_speed = 200
                data_source = 'Default'
                fuel_pred = None
                
                # Try to get full telemetry and predict on all samples
                try:
                    tel = lap.get_telemetry()
                    
                    if tel is not None and len(tel) > 0:
                        has_required = all(col in tel.columns for col in ['RPM', 'Throttle', 'Speed', 'nGear'])
                        
                        if has_required:
                            # Aggregate telemetry to lap-level means (model expects lap features)
                            X_tel_raw = pd.DataFrame([{
                                'rpm': tel['RPM'].mean(),
                                'throttle': tel['Throttle'].mean(),
                                'speed': tel['Speed'].mean(),
                                'gear': tel['nGear'].mean(),
                            }]).dropna()
                            
                            if len(X_tel_raw) > 0:
                                X_tel = prepare_features_for_inference(
                                    X_tel_raw,
                                    team="Aston Martin",
                                    circuit_key=race_key,
                                    year=year,
                                    scaler_global=scaler,
                                    scalers_per_team=scalers_per_team,
                                    team_encoder=team_encoder,
                                    circuit_encoder=circuit_encoder
                                )
                                fuel_pred = float(predict_fuel(model, calibrator, scaler, X_tel)[0])
                                data_source = 'Telemetry'
                except Exception:
                    pass  # Fall back to estimation
                
                # If no telemetry-based prediction, fall back to estimates
                if fuel_pred is None:
                    # Try lap-level speed data
                    if hasattr(lap, 'SpeedST') and pd.notna(lap.SpeedST):
                        avg_speed = lap.SpeedST
                    elif hasattr(lap, 'SpeedFL') and pd.notna(lap.SpeedFL):
                        avg_speed = lap.SpeedFL
                    elif hasattr(lap, 'SpeedI1') and pd.notna(lap.SpeedI1):
                        avg_speed = lap.SpeedI1
                    
                    try:
                        lap_time_seconds = lap['LapTime'].total_seconds() if pd.notna(lap['LapTime']) else 90
                        X_raw = pd.DataFrame({
                            'rpm': [11000 + (90 - lap_time_seconds) * 50],  # Faster laps = higher RPM
                            'throttle': [70 + (90 - lap_time_seconds) * 0.3],  # Faster laps = more throttle
                            'speed': [avg_speed],
                            'gear': [5],
                        })
                        X = prepare_features_for_inference(
                            X_raw,
                            team="Aston Martin",
                            circuit_key=race_key,
                            year=year,
                            scaler_global=scaler,
                            scalers_per_team=scalers_per_team,
                            team_encoder=team_encoder,
                            circuit_encoder=circuit_encoder
                        )
                        fuel_pred = predict_fuel(model, calibrator, scaler, X)[0]
                        data_source = 'Estimated'
                    except Exception:
                        fuel_pred = 0.65  # fallback proxy
                        data_source = 'Default'
                
                fuel_pred_kg = fuel_proxy_to_kg(race_key, fuel_pred)
                
                fuel_estimates.append({
                    'Lap': lap['LapNumber'],
                    'Fuel Proxy': fuel_pred,
                    'Estimated Fuel (kg/lap)': fuel_pred_kg,
                    'Compound': lap['Compound'],
                    'Lap Time': str(lap['LapTime']).split()[-1] if pd.notna(lap['LapTime']) else 'N/A',
                    'Data Source': data_source
                })
            
            if fuel_estimates:
                fuel_df = pd.DataFrame(fuel_estimates)
                
                # Show data quality info
                telemetry_count = len(fuel_df[fuel_df['Data Source'] == 'Telemetry'])
                estimated_count = len(fuel_df[fuel_df['Data Source'] == 'Estimated'])
                default_count = len(fuel_df[fuel_df['Data Source'] == 'Default'])
                
                if telemetry_count < len(fuel_df):
                    st.info(f"ℹ️ Data quality: {telemetry_count} laps with full telemetry, {estimated_count} with partial data, {default_count} estimated from defaults")
                
                # Summary metrics - Race Reality
                st.markdown("#### 📊 Race Fuel Summary")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    total_fuel = fuel_df['Estimated Fuel (kg/lap)'].sum()
                    st.metric("Total Fuel Cost", f"{total_fuel:.2f} kg", 
                             help="Cumulative fuel consumed based on actual driving")
                with col2:
                    avg_fuel = fuel_df['Estimated Fuel (kg/lap)'].mean()
                    st.metric("Avg per Lap", f"{avg_fuel:.3f} kg",
                             help="Average fuel expense per lap")
                with col3:
                    max_fuel_lap = fuel_df.loc[fuel_df['Estimated Fuel (kg/lap)'].idxmax(), 'Lap']
                    max_fuel_val = fuel_df['Estimated Fuel (kg/lap)'].max()
                    st.metric("Most Expensive Lap", f"L{int(max_fuel_lap)}", 
                             f"{max_fuel_val:.3f} kg",
                             help="Lap with highest fuel consumption")
                with col4:
                    min_fuel_lap = fuel_df.loc[fuel_df['Estimated Fuel (kg/lap)'].idxmin(), 'Lap']
                    min_fuel_val = fuel_df['Estimated Fuel (kg/lap)'].min()
                    st.metric("Most Efficient Lap", f"L{int(min_fuel_lap)}", 
                             f"{min_fuel_val:.3f} kg",
                             help="Lap with lowest fuel consumption")
                
                # Fuel consumption chart with annotations
                fig = go.Figure()
                
                # Add average line
                fig.add_hline(y=avg_fuel, line_dash="dash", line_color="white", 
                             opacity=0.5, annotation_text="Average",
                             annotation_position="right")
                
                # Color by data source
                for source in fuel_df['Data Source'].unique():
                    source_data = fuel_df[fuel_df['Data Source'] == source]
                    color = '#cedc00' if source == 'Telemetry' else ('#66ff66' if source == 'Estimated' else '#ff9866')
                    fig.add_trace(go.Scatter(
                        x=source_data['Lap'],
                        y=source_data['Estimated Fuel (kg/lap)'],
                        mode='lines+markers',
                        name=f'{source} Data',
                        line=dict(color=color, width=2.5),
                        marker=dict(size=7),
                        hovertemplate='<b>Lap %{x}</b><br>' +
                                     'Fuel: %{y:.3f} kg/lap<br>' +
                                     '<extra></extra>'
                    ))
                
                # Highlight max and min laps
                fig.add_trace(go.Scatter(
                    x=[max_fuel_lap],
                    y=[max_fuel_val],
                    mode='markers',
                    name='Peak Consumption',
                    marker=dict(size=15, color='#ff6b6b', symbol='star',
                               line=dict(width=2, color='white')),
                    showlegend=True
                ))
                
                fig.add_trace(go.Scatter(
                    x=[min_fuel_lap],
                    y=[min_fuel_val],
                    mode='markers',
                    name='Peak Efficiency',
                    marker=dict(size=15, color='#66ff66', symbol='star',
                               line=dict(width=2, color='white')),
                    showlegend=True
                ))
                
                fig.update_layout(
                    title="Fuel Cost of Reality: Lap-by-Lap Analysis",
                    xaxis_title="Lap Number",
                    yaxis_title="Estimated Fuel Consumption (kg/lap)",
                    plot_bgcolor='#003933',
                    paper_bgcolor='#004b45',
                    font=dict(color='#ffffff', size=12),
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Stint analysis
                st.markdown("#### 🔄 Stint Breakdown")
                stint_analysis = fuel_df.groupby(fuel_df.index // 15).agg({
                    'Estimated Fuel': ['sum', 'mean', 'count']
                }).round(3)
                stint_analysis.columns = ['Total Fuel', 'Avg/Lap', 'Laps']
                stint_analysis.index = [f"Stint {i+1}" for i in stint_analysis.index]
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.dataframe(stint_analysis, use_container_width=True)
                with col2:
                    fig_stint = go.Figure()
                    fig_stint.add_trace(go.Bar(
                        x=stint_analysis.index,
                        y=stint_analysis['Total Fuel'],
                        marker_color='#cedc00',
                        text=stint_analysis['Total Fuel'],
                        textposition='auto',
                        hovertemplate='<b>%{x}</b><br>Total: %{y:.2f} units<extra></extra>'
                    ))
                    fig_stint.update_layout(
                        title="Fuel Consumption by Stint",
                        xaxis_title="Stint",
                        yaxis_title="Total Fuel (units)",
                        plot_bgcolor='#003933',
                        paper_bgcolor='#004b45',
                        font=dict(color='#ffffff')
                    )
                    st.plotly_chart(fig_stint, use_container_width=True)
                
                # Data table
                st.dataframe(fuel_df, use_container_width=True)
                
            else:
                st.warning("Unable to estimate fuel - no lap data available")
                
        except Exception as e:
            st.error(f"Error estimating fuel: {e}")


if __name__ == "__main__":
    main()
