"""
Streamlit dashboard for AMi-Fuel
Displays model metrics, top features, prediction diagnostics, and comparison plot.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="AMi-Fuel Dashboard", layout="wide")

st.title("AMi-Fuel — Engineering Dashboard")

DATA_DIR = Path("outputs")

# Load metrics summary
metrics_path = DATA_DIR / "metrics_summary.txt"
feat_path = DATA_DIR / "feature_importance.csv"
pred_path = DATA_DIR / "test_predictions_enhanced.csv"
comp_path = DATA_DIR / "model_comparison.png"
eng_report = DATA_DIR / "engineering_report.txt"

# Helper loader with graceful errors
def safe_read_text(p: Path):
    if p.exists():
        return p.read_text()
    return "(file not found: {})".format(p)

# Left column: metrics and report
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Model Summary")
    if metrics_path.exists():
        st.text(metrics_path.read_text())
    else:
        st.info("No metrics summary found. Run the training pipeline first.")

    st.markdown("---")
    st.header("Engineering Report")
    if eng_report.exists():
        st.download_button("Download report", eng_report.read_bytes(), file_name="engineering_report.txt")
        st.text(eng_report.read_text())
    else:
        st.info("No engineering report found. Generate it with scripts/generate_engineering_report.py")

with col2:
    st.header("Top Features")
    if feat_path.exists():
        df_feat = pd.read_csv(feat_path)
        st.dataframe(df_feat.head(20))

        # Bar chart of top features
        fig = px.bar(df_feat.head(20).iloc[::-1], x='importance', y='feature', orientation='h',
                     labels={'importance':'Importance','feature':'Feature'}, title='Top 20 Feature Importances')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No feature importance file found.")

    st.markdown("---")
    st.header("Predictions Diagnostics")
    if pred_path.exists():
        df_pred = pd.read_csv(pred_path)
        # Basic metrics
        r2 = df_pred.get('r2') if 'r2' in df_pred.columns else None
        st.metric("Test samples", len(df_pred))
        st.metric("MAE (mean abs)", f"{df_pred['abs_error'].mean():.4f}" if 'abs_error' in df_pred.columns else "n/a")

        # Scatter plot
        if 'y_true' in df_pred.columns and 'y_pred' in df_pred.columns:
            fig2 = px.scatter(df_pred.sample(min(2000, len(df_pred))), x='y_true', y='y_pred',
                              labels={'y_true':'True', 'y_pred':'Predicted'}, title='Predicted vs True (sample)')
            fig2.add_shape(type='line', x0=df_pred['y_true'].min(), x1=df_pred['y_true'].max(), y0=df_pred['y_true'].min(), y1=df_pred['y_true'].max(), line=dict(dash='dash'))
            st.plotly_chart(fig2, use_container_width=True)

            # Residual histogram
            df_pred['residual'] = df_pred['y_true'] - df_pred['y_pred']
            fig3 = px.histogram(df_pred, x='residual', nbins=40, title='Residual Distribution')
            st.plotly_chart(fig3, use_container_width=True)

        # Allow download
        st.download_button("Download predictions CSV", df_pred.to_csv(index=False).encode('utf-8'), file_name='test_predictions_enhanced.csv')
    else:
        st.info("No predictions found. Run the improved training pipeline first.")

st.markdown("---")

st.header("Model Comparison")
if comp_path.exists():
    st.image(str(comp_path), use_column_width=True)
else:
    st.info("No comparison plot found (outputs/model_comparison.png)")

st.markdown("---")

st.header("How to run locally")
st.code("""# From the project root
pip install -r requirements.txt
streamlit run scripts/dashboard.py --server.port 8501
""", language='bash')

st.caption("Tips: use --server.headless true on headless servers and set --server.port to an open port.")
