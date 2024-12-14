import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib

# Load the model
model = joblib.load('xgboost.pkl')

# Define feature names
feature_names = [
    "baseline_LDH", "C2_ALC", "C2_NLR", "C2_PLR", "C3_ALC", 
    "ΔC2_ALC", "ΔC2_Eosinophil", "ΔC3_LMR", "ΔC3_ALB", "ΔC3_ALP"
]

# Streamlit user interface
st.title("Immune Checkpoint Inhibitor Therapeutic Efficacy Predictor")

# ldh: categorical selection
baseline_LDH = st.selectbox("baseline_LDH ≥ 182.214 U/L:", options=[0, 1], format_func=lambda x: 'False (0)' if x == 0 else 'True (1)')

# alc1: categorical selection
C2_ALC = st.selectbox("C2_ALC ≥ 1.661*10^9/L:", options=[0, 1], format_func=lambda x: 'False (0)' if x == 0 else 'True (1)')

# nlr1: categorical selection
C2_NLR = st.selectbox("C2_NLR ≥ 1.286:", options=[0, 1], format_func=lambda x: 'False (0)' if x == 0 else 'True (1)')

# plr1: categorical selection
C2_PLR = st.selectbox("C2_PLR ≥ 187.923:", options=[0, 1], format_func=lambda x: 'False (0)' if x == 0 else 'True (1)')

# alc2: categorical selection
C3_ALC = st.selectbox("C3_ALC ≥ 1.693*10^9/L:", options=[0, 1], format_func=lambda x: 'False (0)' if x == 0 else 'True (1)')

# dalc1: categorical selection
ΔC2_ALC = st.selectbox("ΔC2_ALC ≥ -0.035:", options=[0, 1], format_func=lambda x: 'False (0)' if x == 0 else 'True (1)')

# eo: categorical selection
ΔC2_Eosinophil = st.selectbox("ΔC2_Eosinophil ≥ -0.008:", options=[0, 1], format_func=lambda x: 'False (0)' if x == 0 else 'True (1)')

# dlmr2: categorical selection
ΔC3_LMR = st.selectbox("ΔC3_LMR ≥ -0.104:", options=[0, 1], format_func=lambda x: 'False (0)' if x == 0 else 'True (1)')

# dalb2: categorical selection
ΔC3_ALB = st.selectbox("ΔC3_ALB ≥ -0.098:", options=[0, 1], format_func=lambda x: 'False (0)' if x == 0 else 'True (1)')

# dalp2: categorical selection
ΔC3_ALP = st.selectbox("ΔC3_ALP ≥ 0.576:", options=[0, 1], format_func=lambda x: 'False (0)' if x == 0 else 'True (1)')

# Process inputs and make predictions
feature_values = [baseline_LDH, C2_ALC, C2_NLR, C2_PLR, C3_ALC, ΔC2_ALC, ΔC2_Eosinophil, ΔC3_LMR, ΔC3_ALB, ΔC3_ALP]
features = np.array([feature_values])

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    
    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")
    
    # Generate advice based on prediction results    
    probability = predicted_proba[predicted_class] * 100
    
    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of non-durable benefit (NDB). "
            f"The model predicts that your probability of having NDB is {probability:.1f}%. "
            )
                
    else:
        advice = (
            f"According to our model, you have a low risk of non-durable benefit (NDB). "
            f"The model predicts that your probability of not having NDB is {probability:.1f}%. "
            )
        
    st.write(advice)
    
    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    
    st.image("shap_force_plot.png")