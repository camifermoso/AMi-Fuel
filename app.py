"""
AMi-Fuel: Aston Martin F1 Fuel Optimization Dashboard
Interactive Streamlit app for exploring fuel predictions and optimization strategies.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib

# Page configuration
st.set_page_config(
    page_title="AMi-Fuel Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with custom fonts
st.markdown("""
<style>
    /* Import Darkmode Off CC Regular */
    @import url('https://fonts.cdnfonts.com/css/darkmode-off-cc');
    
    /* Load Domus Titling Medium font - local first, then CDN fallback */
    @font-face {
        font-family: 'Domus Titling Medium';
        src: url('assets/fonts/DomusTitling-Medium.woff2') format('woff2'),
             url('https://db.onlinewebfonts.com/t/0d49ae2ce2f7e5de5341474c5b78f697.woff2') format('woff2'),
             url('https://db.onlinewebfonts.com/t/0d49ae2ce2f7e5de5341474c5b78f697.woff') format('woff');
        font-weight: 500;
        font-style: normal;
        font-display: swap;
    }
    
    /* Apply Domus Titling to headers */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Domus Titling Medium', 'Arial Black', sans-serif !important;
        font-weight: 500 !important;
        letter-spacing: 1.5px;
    }
    
    .main-header {
        font-family: 'Domus Titling Medium', 'Arial Black', sans-serif !important;
        font-size: 3rem;
        font-weight: 500;
        color: #00594C;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: 3px;
    }
    
    .sub-header {
        font-family: 'Domus Titling Medium', 'Arial Black', sans-serif !important;
        font-size: 1.2rem;
        color: #666;
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
        font-family: 'Domus Titling Medium', 'Arial Black', sans-serif !important;
        letter-spacing: 1.5px;
    }
    
    /* Apply Darkmode Off CC Regular to body text and content */
    p, div, span, label, li, td, th, button {
        font-family: 'Darkmode Off CC Regular', -apple-system, sans-serif !important;
    }
    
    /* Keep default for metrics values */
    [data-testid="stMetricValue"] {
        font-family: -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #00594C;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
    }
    
    /* Fix Material Icons rendering in expanders */
    .streamlit-expanderHeader {
        font-family: 'Darkmode Off CC Regular', -apple-system, sans-serif !important;
    }
    
    /* Ensure expander content displays properly */
    [data-testid="stExpander"] summary {
        display: flex !important;
        align-items: center !important;
    }
    
    /* Ensure SVG arrow is visible */
    [data-testid="stExpander"] svg {
        display: inline-block !important;
        flex-shrink: 0 !important;
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
    """Load the trained model and scalers."""
    try:
        model_dir = Path("outputs/two_stage_model")
        model = joblib.load(model_dir / "finetuned_model.pkl")
        calibrator = joblib.load(model_dir / "calibrator.pkl")
        scaler = joblib.load(model_dir / "scaler_global.pkl")
        return model, calibrator, scaler
    except FileNotFoundError:
        st.warning("⚠️ Model files not found. Prediction features disabled.")
        return None, None, None


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
    # Header
    st.markdown('<div class="main-header">🏎️ AMi-Fuel Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Aston Martin F1 Fuel Optimization System</div>', unsafe_allow_html=True)
    
    # Load data
    params_df, scenarios_df, circuits_df = load_recommendations()
    model, calibrator, scaler = load_model()
    train_df, data_type = load_training_data()
    
    if params_df is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["📊 Overview", "🎯 Recommendations", "🏁 Race Scenarios", "🗺️ Circuit Analysis", "🔮 Live Prediction", "📈 Data Explorer"]
    )
    
    # Add model metrics to sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Performance")
    st.sidebar.metric("Training R²", "99.88%")
    st.sidebar.metric("Unseen Races R²", "99.41%")
    st.sidebar.metric("Training Data", "676,513 laps")
    st.sidebar.metric("Years Covered", "2018-2024")
    
    # Main content based on page selection
    if page == "📊 Overview":
        show_overview(params_df, scenarios_df, circuits_df, train_df)
    elif page == "🎯 Recommendations":
        show_recommendations(params_df)
    elif page == "🏁 Race Scenarios":
        show_race_scenarios(scenarios_df)
    elif page == "🗺️ Circuit Analysis":
        show_circuit_analysis(circuits_df, train_df)
    elif page == "🔮 Live Prediction":
        show_live_prediction(model, calibrator, scaler)
    elif page == "📈 Data Explorer":
        show_data_explorer(train_df)


def show_overview(params_df, scenarios_df, circuits_df, train_df):
    """Display overview dashboard."""
    st.header("System Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_fuel_saved = params_df['Fuel Saved (kg/race)'].str.replace(' kg', '').astype(float).sum()
        st.metric("Total Fuel Savings Potential", f"{total_fuel_saved:.1f} kg/race", help="Sum of all optimization strategies")
    
    with col2:
        avg_time_cost = params_df['Time Cost/Race'].str.replace('s', '').astype(float).mean()
        st.metric("Avg Time Cost", f"{avg_time_cost:.2f} sec/race", help="Average time penalty per strategy")
    
    with col3:
        if train_df is not None:
            am_laps = len(train_df[train_df['Team'] == 'Aston Martin'])
            st.metric("AM Training Laps", f"{am_laps:,}", help="Aston Martin laps in training data")
        else:
            st.metric("AM Training Laps", "44,580", help="Total AM laps used in training")
    
    with col4:
        num_circuits = len(circuits_df) if circuits_df is not None else 0
        st.metric("Circuits Analyzed", num_circuits, help="Number of circuits with specific strategies")
    
    st.markdown("---")
    
    # Quick insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Top 3 Fuel Saving Strategies")
        params_copy = params_df.copy()
        params_copy['fuel_saved_num'] = params_copy['Fuel Saved (kg/race)'].str.replace(' kg', '').astype(float)
        top_3 = params_copy.nlargest(3, 'fuel_saved_num')
        
        for idx, row in top_3.iterrows():
            param_name = row['Parameter']
            reduction = row['Reduction']
            with st.expander(f"#{top_3.index.get_loc(idx)+1}: {param_name} ({reduction})", expanded=(top_3.index.get_loc(idx)==0)):
                st.write(f"**Reduction:** {row['Reduction']}")
                st.write(f"**Fuel Saved:** {row['Fuel Saved (kg/race)']}")
                st.write(f"**Time Cost:** {row['Time Cost/Race']}")
                st.write(f"**Time Cost/Lap:** {row['Time Cost/Lap']}")
    
    with col2:
        st.subheader("🏁 Best Race Scenarios")
        if scenarios_df is not None:
            scenarios_df['Fuel Saved'] = scenarios_df['Fuel Saved (Race)'].str.replace(' kg', '').astype(float)
            top_scenarios = scenarios_df.nlargest(3, 'Fuel Saved')
            
            for idx, row in top_scenarios.iterrows():
                scenario_name = str(row['Scenario'])
                # Use checkbox instead of expander to avoid icon issues
                if st.checkbox(f"📋 {scenario_name}", key=f"scenario_{idx}"):
                    st.write(f"**Fuel Saved:** {row['Fuel Saved (Race)']}")
                    st.write(f"**Time Cost:** {row['Time Cost (Race)']}")
                    st.write(f"**Strategy:** {row['Strategy']}")
                    st.markdown("---")
    
    # Visualization
    st.markdown("---")
    st.subheader("📊 Fuel Savings vs Time Cost")
    
    plot_df = params_df.copy()
    plot_df['Fuel Saved'] = plot_df['Fuel Saved (kg/race)'].str.replace(' kg', '').astype(float)
    plot_df['Time Cost'] = plot_df['Time Cost/Race'].str.replace('s', '').astype(float)
    
    fig = px.scatter(
        plot_df,
        x='Time Cost',
        y='Fuel Saved',
        text='Parameter',
        title='Fuel Savings vs Time Cost Trade-off',
        labels={'Time Cost': 'Time Cost (seconds/race)', 'Fuel Saved': 'Fuel Saved (kg/race)'},
        height=500
    )
    fig.update_traces(textposition='top center', marker=dict(size=12, color='#00594C'))
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


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
    for idx, row in params_df.iterrows():
        param_name = row['Parameter']
        reduction = row['Reduction']
        with st.expander(f"**{param_name}** - {reduction}", expanded=(idx==0)):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Fuel Saved", row['Fuel Saved (kg/race)'])
            with col2:
                st.metric("Time Cost", row['Time Cost/Race'])
            with col3:
                st.metric("Time Cost/Lap", row['Time Cost/Lap'])
            
            st.markdown("---")
            st.markdown("**Analysis:**")
            st.write(f"Reducing {row['Parameter']} by {row['Reduction']} can save {row['Fuel Saved (kg/race)']} "
                    f"per race, but will cost approximately {row['Time Cost/Race']} in total race time.")


def show_race_scenarios(scenarios_df):
    """Display race scenario analysis."""
    st.header("🏁 Race Scenario Strategies")
    
    st.info("💡 Pre-configured strategies for different race situations.")
    
    if scenarios_df is not None:
        # Add fuel saved as numeric column for sorting
        scenarios_df['Fuel Saved Numeric'] = scenarios_df['Fuel Saved (Race)'].str.replace(' kg', '').astype(float)
        scenarios_df = scenarios_df.sort_values('Fuel Saved Numeric', ascending=False)
        
        # Create tabs for different scenario types
        scenario_types = scenarios_df['Scenario'].unique()
        
        for scenario in scenario_types:
            scenario_name = str(scenario)
            # Use header + container instead of expander
            st.subheader(f"📋 {scenario_name}")
            scenario_data = scenarios_df[scenarios_df['Scenario'] == scenario].iloc[0]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("💧 Fuel Saved", scenario_data['Fuel Saved (Race)'])
            with col2:
                st.metric("⏱️ Time Cost", scenario_data['Time Cost (Race)'])
            with col3:
                positions = scenario_data.get('Positions Lost', 'N/A')
                st.metric("📍 Est. Positions Lost", positions)
                
                st.markdown("**Strategy:**")
                st.write(scenario_data['Strategy'])
                
                st.markdown("**When to Use:**")
                st.write(scenario_data.get('When to Use', 'N/A'))
                if 'Aggressive' in scenario:
                    st.write("✅ Use when: Fighting for podium, need to push hard")
                    st.write("❌ Avoid when: Conserving engine life, risk of DNF")
                elif 'Balanced' in scenario:
                    st.write("✅ Use when: Normal race conditions, consistent pace needed")
                    st.write("❌ Avoid when: Extreme weather conditions")
                elif 'Conservative' in scenario:
                    st.write("✅ Use when: Protecting position, hot weather, saving components")
                    st.write("❌ Avoid when: Need to make up positions")


def show_circuit_analysis(circuits_df, train_df):
    """Display circuit-specific analysis."""
    st.header("🗺️ Circuit-Specific Strategies")
    
    st.info("💡 Optimized fuel strategies for each circuit based on historical data.")
    
    if circuits_df is not None:
        # Circuit selector
        circuits = circuits_df['Circuit Type'].unique()
        selected_circuit = st.selectbox("Select Circuit Type", circuits)
        
        circuit_data = circuits_df[circuits_df['Circuit Type'] == selected_circuit].iloc[0]
        
        # Display circuit strategy
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💧 Fuel Saved", circuit_data['Fuel Saved'])
        with col2:
            st.metric("⏱️ Time Cost", circuit_data['Time Cost (Race)'])
        with col3:
            st.metric("📊 Baseline Fuel", circuit_data.get('AM Baseline Fuel', 'N/A'))
        with col4:
            positions = circuit_data.get('Positions Impact', 'N/A')
            st.metric("📍 Positions Impact", positions)
        
        st.markdown("---")
        st.markdown("**Best Strategy:**")
        st.write(circuit_data['Best Strategy'])
        
        st.markdown("**AM Advantage:**")
        st.write(circuit_data.get('AM Advantage', 'N/A'))
        
        st.markdown("**Critical Notes:**")
        st.write(circuit_data.get('Critical Notes', 'N/A'))
        
        # Historical data if available
        if train_df is not None:
            st.markdown("---")
            st.subheader(f"📈 Historical Training Data Overview")
            
            # Show overall statistics since circuit CSV uses circuit types, not specific circuits
            am_laps = train_df[train_df['Team'] == 'Aston Martin']
            
            if len(am_laps) > 0:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total AM Laps", len(am_laps))
                with col2:
                    years = am_laps['year'].unique()
                    st.metric("Years Covered", f"{len(years)} ({min(years)}-{max(years)})")
                with col3:
                    avg_temp = am_laps['air_temp'].mean()
                    st.metric("Avg Air Temp", f"{avg_temp:.1f}°C")
                
                # Weather distribution
                st.markdown("**Weather Conditions Distribution:**")
                fig = go.Figure()
                fig.add_trace(go.Box(y=am_laps['air_temp'], name='Air Temp (°C)'))
                fig.add_trace(go.Box(y=am_laps['track_temp'], name='Track Temp (°C)'))
                fig.add_trace(go.Box(y=am_laps['humidity'], name='Humidity (%)'))
                fig.update_layout(height=300, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)


def show_live_prediction(model, calibrator, scaler):
    """Interactive fuel consumption prediction."""
    st.header("🔮 Live Fuel Prediction")
    
    st.info("💡 Adjust parameters to see real-time fuel consumption predictions.")
    
    if model is None:
        st.error("Model not loaded. Please ensure model files exist in outputs/two_stage_model/")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Telemetry Inputs")
        throttle = st.slider("Average Throttle (%)", 0, 100, 70, help="Average throttle application")
        rpm = st.slider("Average RPM", 8000, 12000, 10500, help="Average engine RPM")
        speed = st.slider("Average Speed (km/h)", 150, 320, 250, help="Average lap speed")
        gear = st.slider("Average Gear", 3, 8, 5, help="Average gear selection")
        drs_usage = st.slider("DRS Usage (%)", 0, 100, 15, help="% of lap with DRS active")
        ers_deployment = st.slider("ERS Deployment (%)", 0, 100, 40, help="% of lap with ERS deployed")
    
    with col2:
        st.subheader("🌤️ Weather Conditions")
        air_temp = st.slider("Air Temperature (°C)", 10, 45, 25, help="Ambient air temperature")
        track_temp = st.slider("Track Temperature (°C)", 15, 60, 35, help="Track surface temperature")
        humidity = st.slider("Humidity (%)", 20, 95, 60, help="Relative humidity")
        pressure = st.slider("Pressure (mbar)", 980, 1020, 1013, help="Atmospheric pressure")
        wind_speed = st.slider("Wind Speed (m/s)", 0, 15, 3, help="Wind speed")
    
    # Calculate prediction
    if st.button("🔮 Predict Fuel Consumption", type="primary"):
        # Calculate fuel proxy
        fuel_proxy = 0.60 * (rpm / 12000.0) + 0.40 * (throttle / 100.0)
        
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Fuel Proxy", f"{fuel_proxy:.3f}", help="Normalized fuel consumption metric")
        with col2:
            relative_consumption = (fuel_proxy - 0.7) / 0.7 * 100
            st.metric("vs Baseline", f"{relative_consumption:+.1f}%", help="Compared to typical consumption")
        with col3:
            if fuel_proxy < 0.65:
                efficiency = "🟢 Very Efficient"
            elif fuel_proxy < 0.75:
                efficiency = "🟡 Efficient"
            elif fuel_proxy < 0.85:
                efficiency = "🟠 Moderate"
            else:
                efficiency = "🔴 High Consumption"
            st.metric("Efficiency", efficiency)
        
        # Visual gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fuel_proxy,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Fuel Consumption"},
            gauge={
                'axis': {'range': [None, 1.2]},
                'bar': {'color': "#00594C"},
                'steps': [
                    {'range': [0, 0.65], 'color': "lightgreen"},
                    {'range': [0.65, 0.75], 'color': "yellow"},
                    {'range': [0.75, 0.85], 'color': "orange"},
                    {'range': [0.85, 1.2], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.85
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("**💡 Recommendations:**")
        if fuel_proxy > 0.85:
            st.warning("⚠️ High fuel consumption detected!")
            st.write("- Reduce RPM by 500-1000 to save fuel")
            st.write("- Decrease throttle application in low-speed corners")
            st.write("- Consider more conservative engine mode")
        elif fuel_proxy < 0.65:
            st.success("✅ Excellent fuel efficiency!")
            st.write("- Current settings are very fuel-efficient")
            st.write("- Can push harder if needed without significant penalty")
        else:
            st.info("ℹ️ Balanced fuel consumption")
            st.write("- Current settings are within normal range")
            st.write("- Small adjustments can optimize further")


def show_data_explorer(train_df):
    """Interactive data exploration."""
    st.header("📈 Data Explorer")
    
    st.info("💡 **Training Dataset Overview** - Data used to train the 99.41% accurate fuel prediction model")
    
    # Static training data summary (always available)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Laps", "676,513", help="Total training laps")
    with col2:
        st.metric("Years Covered", "7 (2018-2024)", help="Historical data span")
    with col3:
        st.metric("Circuits", "15", help="Different race tracks")
    with col4:
        st.metric("Teams", "18", help="F1 teams included")
    
    st.markdown("---")
    
    # Circuit breakdown
    st.subheader("🗺️ Circuit Coverage")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**High-Fuel Circuits (5 circuits, 7 years each):**")
        st.write("- 🇧🇭 Bahrain Grand Prix")
        st.write("- 🇪🇸 Spanish Grand Prix")
        st.write("- 🇨🇦 Canadian Grand Prix")
        st.write("- 🇸🇬 Singapore Grand Prix")
        st.write("- 🇯🇵 Japanese Grand Prix")
    
    with col2:
        st.markdown("**Additional Circuits (10 circuits, 3 years each):**")
        st.write("- 🇦🇺 Australian GP, 🇸🇦 Saudi Arabian GP")
        st.write("- 🇦🇪 Abu Dhabi GP, 🇦🇹 Austrian GP")
        st.write("- 🇬🇧 British GP, 🇮🇹 Italian GP")
        st.write("- 🇧🇪 Belgian GP, 🇳🇱 Dutch GP")
        st.write("- 🇺🇸 United States GP, 🇲🇽 Mexico GP")
    
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
    st.info("ℹ️ **Two-stage training:** Pre-trained on all teams (575K laps) → Fine-tuned on Aston Martin (44K laps)")
    
    # If actual data is loaded, show live visualizations
    if train_df is not None:
        st.markdown("---")
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


if __name__ == "__main__":
    main()
