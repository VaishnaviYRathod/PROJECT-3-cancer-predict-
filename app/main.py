import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import os

# Get the absolute path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model", "feature_names.pkl")

# Load model, scaler, and feature names
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    FEATURE_NAMES = joblib.load(FEATURES_PATH)
except FileNotFoundError:
    st.error("Model files not found. Please ensure the model is trained first.")
    st.stop()

def get_feature_range(feature_name):
    """Define reasonable ranges for each feature based on typical values"""
    ranges = {
        'radius_mean': (6.0, 30.0, 15.0),
        'texture_mean': (9.0, 40.0, 20.0),
        'perimeter_mean': (40.0, 190.0, 90.0),
        'area_mean': (140.0, 2500.0, 650.0),
        'smoothness_mean': (0.05, 0.16, 0.1),
        'compactness_mean': (0.02, 0.35, 0.1),
        'concavity_mean': (0.0, 0.5, 0.1),
        'concave points_mean': (0.0, 0.2, 0.05),
        'symmetry_mean': (0.1, 0.3, 0.2),
        'fractal_dimension_mean': (0.05, 0.1, 0.06)
    }
    return ranges.get(feature_name, (0.0, 50.0, 25.0))

def main():
    st.title("Breast Cancer Prediction App")
    st.markdown("""
    ### Input Cell Nuclei Measurements
    Adjust the sliders to input cell characteristics and get a prediction for tumor malignancy.
    Each measurement represents different aspects of the cell nuclei observed in the sample.
    """)

    # Create two columns for the sliders
    col1, col2 = st.columns(2)
    
    # Create sliders for each feature
    input_values = {}
    for i, feature in enumerate(FEATURE_NAMES):
        min_val, max_val, default_val = get_feature_range(feature)
        if i % 2 == 0:
            with col1:
                input_values[feature] = st.slider(
                    f"{feature.replace('_', ' ').title()}", 
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(default_val),
                    help=f"Input value for {feature}"
                )
        else:
            with col2:
                input_values[feature] = st.slider(
                    f"{feature.replace('_', ' ').title()}", 
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(default_val),
                    help=f"Input value for {feature}"
                )

    # Convert input to array in the correct order
    input_array = np.array([input_values[feature] for feature in FEATURE_NAMES]).reshape(1, -1)

    # Scale the input
    input_scaled = scaler.transform(input_array)

    # Predict
    prediction = model.predict(input_scaled)[0]
    prediction_prob = model.predict_proba(input_scaled)[0]

    # Display prediction
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"The tumor is predicted to be **Malignant** with {prediction_prob[1]*100:.2f}% confidence.")
    else:
        st.success(f"The tumor is predicted to be **Benign** with {prediction_prob[0]*100:.2f}% confidence.")

    # Radar Chart Visualization
    st.subheader("Feature Distribution")
    
    # Prepare data for radar chart
    features = FEATURE_NAMES.copy()
    features.append(FEATURE_NAMES[0])  # Add first feature again to close the loop
    values = [input_values[feature] for feature in FEATURE_NAMES]
    values.append(values[0])  # Add first value again to close the loop

    # Create radar chart using graph objects
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=features,
        fill='toself',
        name='Features'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                showticklabels=True
            )
        ),
        showlegend=False,
        title="Feature Distribution Radar Chart"
    )
    
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()