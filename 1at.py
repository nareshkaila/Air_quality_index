# app.py - Attractive Streamlit AQI explorer + anomaly detection
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------- USER SETTINGS -----------------
DATA_PATH = "/Users/kailanaresh/Downloads/data_date.csv"  # <- adjust if needed
PAGE_TITLE = "AQI Monitoring — Interactive Dashboard"
# -------------------------------------------------

st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state="expanded")

# small custom header style
st.markdown(
    """
    <style>
    .header {
        background: linear-gradient(90deg,#ff7e00,#ffb347);
        padding: 18px;
        border-radius: 12px;
        color: white;
        text-align: left;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .small {font-size:13px; color: rgba(255,255,255,0.95)}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(f'<div class="header"><h1 style="margin:0">{PAGE_TITLE}</h1>'
            f'<div class="small">Interactive exploration, anomaly detection (IsolationForest), and quick downloads</div></div>',
            unsafe_allow_html=True)

# ----------------- Load data -----------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    # attempt to parse typical date-like columns
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower():
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Data file not found at: {DATA_PATH}. Change DATA_PATH at top of the script.")
    st.stop()

# Harmonize column names (trim)
df.columns = [c.strip() for c in df.columns]

# Attempt to detect common column names
possible_date = [c for c in df.columns if "date" in c.lower() or "time" in c.lower() or "timestamp" in c.lower()]
possible_aqi = [c for c in df.columns if "aqi" in c.lower()]
possible_country = [c for c in df.columns if "country" in c.lower()]

if not possible_date:
    st.warning("No date-like column detected. Time series features will be limited.")
date_col = possible_date[0] if possible_date else None
aqi_col = possible_aqi[0] if possible_aqi else None
country_col = possible_country[0] if possible_country else None

# quick conversions
if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(date_col)

# Sidebar controls
st.sidebar.header("Filters & Options")
if date_col:
    min_date, max_date = df[date_col].min(), df[date_col].max()
    date_range = st.sidebar.date_input("Date range", value=(min_date.date(), max_date.date()), 
                                       min_value=min_date.date(), max_value=max_date.date())
else:
    date_range = None

country_list = sorted(df[country_col].dropna().unique().tolist()) if country_col else []
sel_countries = st.sidebar.multiselect("Country (filter)", options=country_list, default=country_list[:3] if country_list else None)

st.sidebar.markdown("---")
run_if = st.sidebar.checkbox("Run IsolationForest anomaly detection", value=True)
contam = st.sidebar.slider("IF contamination (approx)", 0.001, 0.1, 0.01, step=0.001)
feature_choice = st.sidebar.multiselect("Features to use for IF (numeric)", options=df.select_dtypes(include=np.number).columns.tolist(),
                                       default=[c for c in df.select_dtypes(include=np.number).columns.tolist() if c==aqi_col or c])

# Filter dataframe according to sidebar
df_filtered = df.copy()
if date_col and date_range:
    start_dt = pd.to_datetime(date_range[0])
    end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df_filtered = df_filtered[(df_filtered[date_col] >= start_dt) & (df_filtered[date_col] <= end_dt)]
if country_col and sel_countries:
    df_filtered = df_filtered[df_filtered[country_col].isin(sel_countries)]

# Top metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows (filtered)", f"{len(df_filtered):,}")
col2.metric("Unique countries", f"{df_filtered[country_col].nunique() if country_col else 'N/A'}")
if aqi_col:
    col3.metric("AQI mean", f"{df_filtered[aqi_col].dropna().mean():.2f}")
    col4.metric("AQI max", f"{df_filtered[aqi_col].dropna().max():.0f}")
else:
    col3.metric("AQI mean", "N/A")
    col4.metric("AQI max", "N/A")

st.markdown("----")

# Main layout: left time-series + right summary and histogram
left, right = st.columns((3,1))

with left:
    st.subheader("AQI Time Series")
    if date_col and aqi_col:
        # Resample per day if very dense
        to_plot = df_filtered.copy()
        # if data is too dense, allow aggregation by hour/day
        granularity = st.selectbox("Granularity", ["Auto", "Hourly", "Daily", "Raw"], index=0)
        if granularity == "Hourly":
            to_plot = to_plot.set_index(date_col).resample("H").mean().reset_index()
        elif granularity == "Daily":
            to_plot = to_plot.set_index(date_col).resample("D").mean().reset_index()
        # Build interactive plotly chart
        fig = px.line(to_plot, x=date_col, y=aqi_col, color=country_col if country_col else None,
                      labels={date_col: "Date", aqi_col: "AQI"},
                      title="AQI over time (interactive)")
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No date or AQI column detected for time-series visualization.")

    # If user ran IF, show anomalies on time series
    if run_if and feature_choice and len(feature_choice) > 0:
        numeric_df = df_filtered[feature_choice].dropna()
        if numeric_df.shape[0] < 10:
            st.warning("Not enough numeric rows to run IsolationForest reliably.")
        else:
            with st.spinner("Running IsolationForest..."):
                scaler = StandardScaler()
                X = scaler.fit_transform(numeric_df)
                if_model = IsolationForest(contamination=contam, random_state=42)
                preds = if_model.fit_predict(X)
                # map predictions back to df_filtered index
                anomaly_idx = numeric_df.index[preds == -1]
                df_filtered = df_filtered.copy()
                df_filtered["anomaly_if"] = False
                df_filtered.loc[anomaly_idx, "anomaly_if"] = True

            st.success(f"IsolationForest found {df_filtered['anomaly_if'].sum()} anomalies (in filtered rows).")

            # Plot with anomalies highlighted
            if date_col and aqi_col:
                fig2 = px.scatter(df_filtered.reset_index(), x=date_col, y=aqi_col,
                                  color=df_filtered.reset_index()["anomaly_if"].map({True: "anomaly", False:"normal"}),
                                  title="AQI with IsolationForest anomalies highlighted",
                                  labels={date_col: "Date", aqi_col: "AQI"})
                st.plotly_chart(fig2, use_container_width=True)

with right:
    st.subheader("Distribution & Quick Stats")
    if aqi_col:
        fig_h = px.histogram(df_filtered, x=aqi_col, nbins=40, marginal="box", title="AQI Distribution")
        st.plotly_chart(fig_h, use_container_width=True)
        st.write(df_filtered[aqi_col].describe().to_frame().T)
    else:
        st.info("No AQI numeric column detected.")

    # show top countries or status breakdown
    if country_col:
        st.subheader("Country counts (top 10)")
        topc = df_filtered[country_col].value_counts().nlargest(10).reset_index()
        topc.columns = [country_col, "count"]
        fig_bar = px.bar(topc, x=country_col, y="count", title="Top Countries (filtered)")
        st.plotly_chart(fig_bar, use_container_width=True)

# Data preview and download
st.markdown("---")
st.subheader("Preview & Export")
st.dataframe(df_filtered.head(200))

csv_bytes = df_filtered.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered data (CSV)", csv_bytes, file_name="aqi_filtered.csv", mime="text/csv")

# Footer
st.markdown(
    """
    <div style="padding:12px 0 20px 0; color: #6b6b6b;">
    Built with ❤️ &dash; Interactive plots: Plotly, anomaly detection: IsolationForest. 
    Tips: try different contamination values, or add rolling-window features (lag, mean) to the IF feature set.
    </div>
    """,
    unsafe_allow_html=True,
)
