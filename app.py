import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
import tempfile
import joblib
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from datetime import timedelta

# Page config
st.set_page_config(page_title="Road Roughness Detection", layout="wide")

# --- Styling ---
st.markdown("""
    <style>
        .centered-title {
            text-align: center;
            font-size:30px !important;
            color: #006699;
            margin-top: -30px;
        }
        .subtext {
            text-align: center;
            color: gray;
            font-size: 15px;
            margin-bottom: 25px;
        }
        .stMetric {
            text-align: center !important;
        }
        .metric-container {
            padding: 10px;
            background-color: #f9f9f9;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='centered-title'>🚗 Road Surface Roughness Detection App</div>", unsafe_allow_html=True)
st.markdown("<div class='subtext'>Upload your sensor data (.zip) to visualize road conditions in a beautiful map and summary dashboard.</div>", unsafe_allow_html=True)

# --- Upload ---
st.markdown("### 📥 Upload Your Sensor ZIP File")
st.markdown("Ensure your `.zip` contains `Accelerometer.csv`, `Gyroscope.csv`, and `Location.csv`.")

zip_file = st.file_uploader("Choose a ZIP file", type="zip")

if zip_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.read())

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        # Find sensor files
        def find_file(keywords):
            for root, _, files in os.walk(tmpdir):
                for file in files:
                    fname = file.lower()
                    if all(k in fname for k in keywords):
                        return os.path.join(root, file)
            return None

        accel_path = find_file(["accelerometer"])
        gyro_path = find_file(["gyroscope"])
        gps_path = find_file(["location"])

        if not accel_path or not gyro_path or not gps_path:
            st.error("One or more required files not found in the ZIP.")
        else:
            # Merge files
            accel = pd.read_csv(accel_path)
            gyro = pd.read_csv(gyro_path)
            gps = pd.read_csv(gps_path)

            for df in [accel, gyro, gps]:
                df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
                df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
                if df["timestamp"].max() > 1e15:
                    df["timestamp"] = df["timestamp"] / 1_000_000
                df.sort_values("timestamp", inplace=True)
                df.drop(columns=[col for col in df.columns if "elapsed" in col.lower()], inplace=True)

            merged = pd.merge_asof(accel, gyro, on="timestamp", direction="nearest", tolerance=50)
            merged = pd.merge_asof(merged, gps, on="timestamp", direction="nearest", tolerance=200)
            merged.dropna(inplace=True)
            merged.insert(0, "readable_time", pd.to_datetime(merged["timestamp"], unit='ms') + timedelta(hours=8))
            raw_df = merged.copy()

            # Feature extraction
            df = raw_df.rename(columns={
                'x_x': 'accel_x', 'y_x': 'accel_y', 'z_x': 'accel_z',
                'x_y': 'gyro_x', 'y_y': 'gyro_y', 'z_y': 'gyro_z'
            }).dropna().sort_values("timestamp").reset_index(drop=True)

            segments = []
            window_size = 200
            for i in range(0, len(df) - window_size, window_size):
                window = df.iloc[i:i+window_size]
                if window['speed'].mean() >= 5:
                    y_filtered = window['accel_y'].rolling(window=5, center=True).mean().bfill().ffill()
                    features = {
                        'start_time': window['timestamp'].iloc[0],
                        'end_time': window['timestamp'].iloc[-1],
                        'mean_accel_y': y_filtered.mean(),
                        'std_accel_y': y_filtered.std(),
                        'rms_accel_y': np.sqrt(np.mean(y_filtered**2)),
                        'peak2peak_accel_y': y_filtered.max() - y_filtered.min(),
                        'mean_speed': window['speed'].mean() * 3.6,
                        'elevation_change': window['altitude'].iloc[-1] - window['altitude'].iloc[0] if 'altitude' in window else 0,
                        'gyro_y_std': window['gyro_y'].std(),
                        'gyro_x_std': window['gyro_x'].std(),
                        'latitude': window['latitude'].mean(),
                        'longitude': window['longitude'].mean()
                    }
                    segments.append(features)

            features_df = pd.DataFrame(segments)
            if features_df.empty:
                st.warning("No valid segments extracted. Try with another ZIP.")
                st.stop()

            # Prediction
            model = joblib.load("iri_rf_classifier_cleaned.pkl")
            X_pred = features_df.drop(columns=["start_time", "end_time", "latitude", "longitude"], errors="ignore")
            features_df["prediction"] = model.predict(X_pred)

            # Add readable time
            if "readable_time" in raw_df.columns:
                time_lookup = raw_df[["timestamp", "readable_time"]].drop_duplicates()
                features_df = features_df.merge(time_lookup, left_on="start_time", right_on="timestamp", how="left")
                features_df.drop(columns=["timestamp"], inplace=True)

            # Sidebar options
            st.sidebar.title("🧭 Display Options")
            show_data = st.sidebar.checkbox("📋 Show Segment Data", value=True)
            show_pie = st.sidebar.checkbox("📊 Show Pie Chart", value=True)
            show_map = st.sidebar.checkbox("🗺️ Show Map View", value=True)

            # Summary Metrics
            st.markdown("### 📈 Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Segments", len(features_df))
            col2.metric("Smooth", sum(features_df["prediction"] == "Smooth"))
            col3.metric("Rough", sum(features_df["prediction"] == "Rough"))

            # Show data
            if show_data:
                st.markdown("### 📋 Segment Predictions")
                st.dataframe(features_df[["readable_time", "mean_speed", "prediction"]])

            # Pie chart
            if show_pie:
                st.markdown("### 📎 Road Condition Distribution")
                fig, ax = plt.subplots()
                features_df["prediction"].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, startangle=90)
                ax.set_ylabel('')
                st.pyplot(fig)

            # Map
            if show_map and "latitude" in features_df.columns and "longitude" in features_df.columns:
                st.markdown("### 🗺️ Road Segment Map")
                color_map = {"Smooth": "green", "Fair": "blue", "Rough": "red"}

                m = folium.Map(
                    location=[features_df["latitude"].mean(), features_df["longitude"].mean()],
                    zoom_start=15,
                    tiles="OpenStreetMap"  # Light tile
                )

                for _, row in features_df.iterrows():
                    folium.CircleMarker(
                        location=[row["latitude"], row["longitude"]],
                        radius=5,
                        color=color_map.get(row["prediction"], "gray"),
                        fill=True,
                        fill_opacity=0.8
                    ).add_to(m)

                st_folium(m, width=1100, height=500)

            # Optional: show raw data in expander
            with st.expander("🔍 See Raw Merged Sensor Data"):
                st.dataframe(raw_df.head())
