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
    /* Import Google Fonts - Stack Sans Notch for headers, Inter for body */
    @import url('https://fonts.googleapis.com/css2?family=Stack+Sans+Notch:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Dark green background for main app */
    .stApp {
        background-color: #004b45;
    }
    
    /* Hide Streamlit header bar */
    header[data-testid="stHeader"] {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* Main content area */
    .main .block-container {
        background-color: #004b45;
        padding-top: 2rem;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #003933 !important;
    }
    
    section[data-testid="stSidebar"] > div {
        background-color: #003933;
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
        box-shadow: 0 1px 3px rgba(206, 220, 0, 0.2);
    }
    .stMetric {
        background-color: #003933;
        padding: 10px;
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
    # Header - Race Engineer Focused
    st.markdown('<div class="main-header">🏎️ AMi-Fuel Race Engineer System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-Time Fuel Strategy Decision Support</div>', unsafe_allow_html=True)
    
    # Load data
    params_df, scenarios_df, circuits_df = load_recommendations()
    model, calibrator, scaler = load_model()
    train_df, data_type = load_training_data()
    
    if params_df is None:
        st.stop()
    
    # Sidebar - Race Weekend Context
    st.sidebar.title("🏁 Race Control")
    
    # Race conditions
    st.sidebar.markdown("### 🌡️ Live Conditions")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        air_temp = st.number_input("Air Temp (°C)", 15.0, 45.0, 25.0, 1.0, key="air_temp")
    with col2:
        track_temp = st.number_input("Track Temp (°C)", 20.0, 60.0, 35.0, 1.0, key="track_temp")
    
    rainfall = st.sidebar.checkbox("Rainfall Expected", value=False)
    
    # Navigation
    st.sidebar.markdown("---")
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.radio(
        "Select Tool",
        ["🎯 Race Strategy", "⚡ Quick Decisions", "🏁 Scenario Planning", "🗺️ Circuit Intel", "🔮 Live Calculator", "📊 Performance Data"],
        help="Choose the tool you need for the current phase of the race weekend"
    )
    
    # System status
    st.sidebar.markdown("---")
    st.sidebar.subheader("✅ System Status")
    st.sidebar.metric("Model Confidence", "99.4%", help="Validated on unseen races")
    st.sidebar.metric("Data Currency", "2024 Season", help="Last updated: 2024")
    st.sidebar.caption(f"🕒 Last sync: {datetime.now().strftime('%H:%M:%S')}")
    
    # Main content based on page selection - Race Engineer Tools
    if page == "🎯 Race Strategy":
        show_race_strategy(params_df, scenarios_df, circuits_df, train_df, air_temp, track_temp, rainfall)
    elif page == "⚡ Quick Decisions":
        show_quick_decisions(params_df, air_temp, track_temp)
    elif page == "🏁 Scenario Planning":
        show_scenario_planning(scenarios_df, params_df)
    elif page == "🗺️ Circuit Intel":
        show_circuit_intel(circuits_df, train_df)
    elif page == "🔮 Live Calculator":
        show_live_calculator(model, calibrator, scaler, air_temp, track_temp, rainfall)
    elif page == "� Performance Data":
        show_performance_data(train_df)


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
            "💧 Max Fuel Saving", 
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
    
    # Interactive Strategy Selector - Race Engineer Decision Tool
    st.subheader("⚙️ Setup Adjustment Evaluator")
    st.caption("Compare setup changes and their impact on fuel consumption vs lap time")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_strategies = st.multiselect(
            "Select setup parameters to evaluate",
            options=params_df['Parameter'].tolist(),
            default=params_df.nlargest(3, 'Fuel Saved')['Parameter'].tolist()[:3],
            max_selections=5,
            help="Choose up to 5 parameters to compare their race impact"
        )
    
    with col2:
        chart_type = st.radio(
            "View Type",
            ["Comparison", "Trade-off", "Multi-Axis"],
            horizontal=False,
            help="Comparison=bars, Trade-off=scatter, Multi-Axis=radar"
        )
    
    with col3:
        priority = st.radio(
            "Priority",
            ["Fuel Focus", "Time Focus", "Balanced"],
            help="Filter recommendations based on race strategy priority"
        )
    
    if selected_strategies:
        filtered_df = params_df[params_df['Parameter'].isin(selected_strategies)].copy()
        
        # Apply priority filter
        if priority == "Fuel Focus":
            filtered_df = filtered_df.nlargest(min(5, len(filtered_df)), 'Fuel Saved')
            st.info("🎯 Showing strategies with maximum fuel savings")
        elif priority == "Time Focus":
            filtered_df = filtered_df.nsmallest(min(5, len(filtered_df)), 'Time Cost')
            st.info("⚡ Showing strategies with minimum time penalty")
        else:
            filtered_df['balance_score'] = (filtered_df['Fuel Saved'] / filtered_df['Fuel Saved'].max()) - (filtered_df['Time Cost'] / filtered_df['Time Cost'].max())
            filtered_df = filtered_df.nlargest(min(5, len(filtered_df)), 'balance_score')
            st.info("⚖️ Showing balanced fuel/time trade-off strategies")
        
        # Dynamic chart based on selection
        if chart_type == "Comparison":
            fig = go.Figure()
            
            # Fuel Saved bars
            fig.add_trace(go.Bar(
                name='Fuel Saved (kg)',
                x=filtered_df['Parameter'],
                y=filtered_df['Fuel Saved'],
                marker_color='#cedc00',
                text=filtered_df['Fuel Saved'].round(2),
                textposition='outside',
                yaxis='y'
            ))
            
            # Time Cost bars (on secondary axis)
            fig.add_trace(go.Bar(
                name='Time Cost (s)',
                x=filtered_df['Parameter'],
                y=filtered_df['Time Cost'],
                marker_color='#ff6b6b',
                text=filtered_df['Time Cost'].round(2),
                textposition='outside',
                yaxis='y2',
                opacity=0.7
            ))
            
            fig.update_layout(
                title="Race Setup Analysis: Fuel vs Time Impact",
                xaxis=dict(title="Setup Parameter", tickangle=-45),
                yaxis=dict(title="Fuel Saved (kg/race)", side='left', color='#cedc00'),
                yaxis2=dict(title="Time Cost (s/race)", side='right', overlaying='y', color='#ff6b6b'),
                height=500,
                hovermode='x unified',
                plot_bgcolor='#003933',
                paper_bgcolor='#004b45',
                font=dict(color='#ffffff', family='Inter'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Trade-off":
            fig = px.scatter(
                filtered_df,
                x='Time Cost',
                y='Fuel Saved',
                text='Parameter',
                size='Fuel Saved',
                color='Fuel Saved',
                color_continuous_scale=['#004b45', '#cedc00'],
                title='⚖️ Performance Trade-off Matrix',
                labels={'Time Cost': 'Lap Time Penalty (s)', 'Fuel Saved': 'Fuel Reduction (kg)'},
                height=500
            )
            
            # Add quadrant lines
            mean_time = filtered_df['Time Cost'].mean()
            mean_fuel = filtered_df['Fuel Saved'].mean()
            
            fig.add_hline(y=mean_fuel, line_dash="dot", line_color="#cedc00", opacity=0.5, annotation_text="Avg Fuel")
            fig.add_vline(x=mean_time, line_dash="dot", line_color="#ff6b6b", opacity=0.5, annotation_text="Avg Time")
            
            fig.update_traces(
                textposition='top center',
                marker=dict(line=dict(width=2, color='#ffffff'))
            )
            fig.update_layout(
                plot_bgcolor='#003933',
                paper_bgcolor='#004b45',
                font=dict(color='#ffffff', family='Inter'),
                showlegend=False,
                annotations=[
                    dict(text="🌟 IDEAL ZONE<br>(Low Time, High Fuel)", x=mean_time*0.5, y=mean_fuel*1.2, 
                         showarrow=False, font=dict(color='#66ff66', size=10), opacity=0.7)
                ]
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:  # Multi-Axis (Radar)
            # Normalize values for radar chart
            max_fuel = filtered_df['Fuel Saved'].max()
            max_time = filtered_df['Time Cost'].max()
            
            fig = go.Figure()
            
            for idx, row in filtered_df.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[
                        (row['Fuel Saved'] / max_fuel) * 100,
                        (1 - row['Time Cost'] / max_time) * 100,  # Invert time cost (lower is better)
                        (row['Fuel Saved'] / row['Time Cost'] if row['Time Cost'] > 0 else 0) * 10
                    ],
                    theta=['Fuel Savings', 'Time Efficiency', 'Overall Efficiency'],
                    fill='toself',
                    name=row['Parameter'][:30]
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], color='#cedc00'),
                    bgcolor='#003933'
                ),
                title="Multi-Dimensional Strategy Comparison",
                height=500,
                paper_bgcolor='#004b45',
                font=dict(color='#ffffff', family='Inter'),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Comparison table
        st.markdown("### 📊 Detailed Comparison")
        comparison_data = filtered_df[['Parameter', 'Reduction', 'Fuel Saved', 'Time Cost']].copy()
        comparison_data['Efficiency'] = (comparison_data['Fuel Saved'] / comparison_data['Time Cost']).round(3)
        comparison_data = comparison_data.sort_values('Fuel Saved', ascending=False)
        
        # Style the dataframe
        st.dataframe(
            comparison_data,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Parameter": st.column_config.TextColumn("Strategy", width="medium"),
                "Reduction": st.column_config.TextColumn("Reduction", width="small"),
                "Fuel Saved": st.column_config.NumberColumn("Fuel Saved (kg)", format="%.2f"),
                "Time Cost": st.column_config.NumberColumn("Time Cost (s)", format="%.2f"),
                "Efficiency": st.column_config.NumberColumn("Efficiency Ratio", format="%.3f", help="kg saved per second")
            }
        )
    
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
    """Scenario Planning - Race strategy simulation."""
    st.header("🏁 Race Scenario Planning")
    st.caption("Simulate and compare race strategies under different conditions")
    
    # Scenario builder
    st.markdown("### 🎮 Custom Scenario Builder")
    col1, col2, col3 = st.columns(3)
    with col1:
        strategy = st.select_slider("Strategy Type", ["Conservative", "Balanced", "Aggressive"], value="Balanced")
    with col2:
        safety_cars = st.number_input("Expected Safety Cars", 0, 3, 1)
    with col3:
        tire_stops = st.selectbox("Tire Strategy", ["1-Stop", "2-Stop", "3-Stop"])
    
    st.markdown("---")
    show_race_scenarios(scenarios_df)


def show_race_scenarios(scenarios_df):
    """Display race scenario analysis."""
    st.markdown("### � Pre-Configured Race Scenarios")
    
    st.info("💡 Proven strategies for different race situations")
    
    if scenarios_df is not None:
        # Add fuel saved as numeric column for sorting
        scenarios_df['Fuel Saved Numeric'] = scenarios_df['Fuel Saved (Race)'].str.replace(' kg', '').astype(float)
        scenarios_df = scenarios_df.sort_values('Fuel Saved Numeric', ascending=False)
        
        # Create tabs for different scenario types
        scenario_types = scenarios_df['Scenario'].unique()
        
        for idx, scenario in enumerate(scenario_types):
            scenario_name = str(scenario)
            scenario_data = scenarios_df[scenarios_df['Scenario'] == scenario].iloc[0]
            is_open = "open" if idx == 0 else ""
            
            # Build the content HTML
            positions = scenario_data.get('Positions Lost', 'N/A')
            strategy = scenario_data['Strategy']
            when_to_use = scenario_data.get('When to Use', 'N/A')
            
            # Determine usage tips
            usage_tips = ""
            if 'MINIMAL' in scenario or 'Minimal' in scenario:
                usage_tips = """
                <p>✅ Use when: Managing fuel to finish comfortably</p>
                <p>❌ Avoid when: Need to push hard for positions</p>
                """
            elif 'BALANCED' in scenario or 'Balanced' in scenario:
                usage_tips = """
                <p>✅ Use when: Normal race conditions, consistent pace needed</p>
                <p>❌ Avoid when: Extreme weather conditions</p>
                """
            elif 'CRITICAL' in scenario or 'Critical' in scenario:
                usage_tips = """
                <p>✅ Use when: Must save fuel to finish race</p>
                <p>❌ Avoid when: Fighting for positions</p>
                """
            elif 'TIRE' in scenario or 'Tire' in scenario:
                usage_tips = """
                <p>✅ Use when: Extending stint, managing both resources</p>
                <p>❌ Avoid when: Need maximum pace</p>
                """
            
            # Custom HTML collapsible with ALL content inside
            st.markdown(f"""
            <details class="custom-details" {is_open}>
                <summary>📋 {scenario_name}</summary>
                <div class="custom-details-content">
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1rem;">
                        <div style="background: #004b45; padding: 1rem; border-radius: 8px; border: 2px solid #cedc00; box-shadow: 0 2px 6px rgba(206,220,0,0.2);">
                            <div style="font-size: 0.8rem; color: #cedc00; font-weight: 600; text-transform: uppercase;">💧 Fuel Saved</div>
                            <div style="font-size: 1.5rem; font-weight: bold; color: #ffffff;">{scenario_data['Fuel Saved (Race)']}</div>
                        </div>
                        <div style="background: #004b45; padding: 1rem; border-radius: 8px; border: 2px solid #cedc00; box-shadow: 0 2px 6px rgba(206,220,0,0.2);">
                            <div style="font-size: 0.8rem; color: #cedc00; font-weight: 600; text-transform: uppercase;">⏱️ Time Cost</div>
                            <div style="font-size: 1.5rem; font-weight: bold; color: #ffffff;">{scenario_data['Time Cost (Race)']}</div>
                        </div>
                        <div style="background: #004b45; padding: 1rem; border-radius: 8px; border: 2px solid #cedc00; box-shadow: 0 2px 6px rgba(206,220,0,0.2);">
                            <div style="font-size: 0.8rem; color: #cedc00; font-weight: 600; text-transform: uppercase;">📍 Est. Positions Lost</div>
                            <div style="font-size: 1.5rem; font-weight: bold; color: #ffffff;">{positions}</div>
                        </div>
                    </div>
                    <p><strong>Strategy:</strong> {strategy}</p>
                    <p><strong>When to Use:</strong> {when_to_use}</p>
                    {usage_tips}
                </div>
            </details>
            """, unsafe_allow_html=True)


def show_circuit_intel(circuits_df, train_df):
    """Circuit Intelligence - Track-specific fuel strategies."""
    st.header("🗺️ Circuit Intelligence Database")
    st.caption("Track-specific fuel strategies and historical performance data")
    
    st.info("💡 Pre-race preparation: Review circuit-specific fuel optimization strategies")
    
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


def show_live_calculator(model, calibrator, scaler, air_temp, track_temp, rainfall):
    """Live Calculator - Real-time fuel consumption predictions."""
    st.header("🔮 Live Fuel Calculator")
    st.caption("Real-time fuel consumption calculator for current session conditions")
    
    st.success(f"🌡️ Using current conditions: Air {air_temp}°C | Track {track_temp}°C | Rainfall: {'Yes' if rainfall else 'No'}")
    
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
        st.caption("Using live conditions from sidebar. Adjust if needed:")
        air_temp_input = st.slider("Air Temperature (°C)", 10, 45, int(air_temp), help="Ambient air temperature", key="calc_air_temp")
        track_temp_input = st.slider("Track Temperature (°C)", 15, 60, int(track_temp), help="Track surface temperature", key="calc_track_temp")
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


def show_performance_data(train_df):
    """Performance Data - Historical analysis and model training insights."""
    st.header("� Performance Data & Model Insights")
    st.caption("Historical race data and model training information")
    
    st.info("💡 **Model Training Data** - 676,513 laps from 7 seasons (2018-2024) with 99.41% validation accuracy")
    
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
