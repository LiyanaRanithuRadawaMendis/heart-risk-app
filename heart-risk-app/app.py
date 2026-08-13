import os
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib

# 1. Path setup and Model Loading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, 'heart_disease_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))

# 2. Page Configuration & Custom CSS for a Premium UI
st.set_page_config(page_title="AI Cardiovascular Engine", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .main { background-color: #0b0f19; color: #e2e8f0; font-family: 'Inter', sans-serif; }
    h1 { color: #38bdf8; text-align: center; margin-bottom: 0px; padding-bottom: 0px; }
    p { text-align: center; color: #94a3b8; font-size: 1.1rem; }
    .stButton>button { width: 100%; border-radius: 8px; background-color: #0ea5e9; color: white; border: none; padding: 12px; font-weight: bold; font-size: 16px; transition: all 0.3s; margin-top: 20px; }
    .stButton>button:hover { background-color: #0284c7; transform: translateY(-2px); box-shadow: 0 4px 12px rgba(14, 165, 233, 0.4); }
    div[data-testid="stForm"] { border: 1px solid #1e293b; background-color: #0f172a; padding: 20px; border-radius: 12px; }
    </style>
""", unsafe_allow_html=True)

st.title("🫀 AI Cardiovascular Diagnostic Engine")
st.write("Enter clinical vitals to initiate a predictive health scan.")

# 3. Central Animation Display (The "3D" Hologram Heart)
animation_placeholder = st.empty()

# Default Scanning State Animation
default_html = """
<div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 350px; width: 100%; background: radial-gradient(circle, #1e293b 0%, #0b0f19 70%); border-radius: 12px; border: 1px solid #334155; position: relative; overflow: hidden;">
    <div style="position: absolute; width: 200%; height: 2px; background: rgba(56, 189, 248, 0.5); box-shadow: 0 0 15px #38bdf8; animation: scan 3s linear infinite;"></div>
    <!-- Replace the src URL below with a link to your own custom 2D anatomical asset if desired -->
    <img id="heart" src="https://cdn-icons-png.flaticon.com/512/873/873295.png" style="width: 180px; filter: drop-shadow(0 0 15px rgba(56, 189, 248, 0.6)) grayscale(80%) sepia(20%) hue-rotate(180deg); animation: pulse 1.5s infinite;">
    <div style="color: #38bdf8; margin-top: 20px; font-weight: bold; letter-spacing: 2px; text-transform: uppercase;">Awaiting Vitals...</div>
    <style>
        @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.05); } 100% { transform: scale(1); } }
        @keyframes scan { 0% { top: -10%; } 100% { top: 110%; } }
    </style>
</div>
"""
animation_placeholder.components.v1.html(default_html, height=360)

st.markdown("---")

# 4. Data Entry Section using a Form (Prevents app from reloading on every single click)
with st.form("clinical_form"):
    st.subheader("Patient Clinical Attributes")
    
    # We use columns to organize the inputs neatly under the central heart
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", 1, 120, 50, help="The patient's age in years.")
        
        sex_map = {"Female": 0, "Male": 1}
        sex_choice = st.selectbox("Biological Sex", list(sex_map.keys()), help="Biological sex of the patient.")
        
        cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
        cp_choice = st.selectbox("Chest Pain Type", list(cp_map.keys()), help="Type of chest pain reported by the patient.")
        
        trestbps = st.number_input("Resting Blood Pressure", 80.0, 200.0, 120.0, help="Resting blood pressure in mm Hg upon admission.")
        
        chol = st.number_input("Serum Cholesterol", 100.0, 600.0, 200.0, help="Serum cholesterol in mg/dl. (Values over 200 may indicate concern).")

    with col2:
        fbs_map = {"Under 120 mg/dl (Normal)": 0, "Over 120 mg/dl (Elevated)": 1}
        fbs_choice = st.selectbox("Fasting Blood Sugar", list(fbs_map.keys()), help="Is fasting blood sugar > 120 mg/dl?")
        
        restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
        restecg_choice = st.selectbox("Resting ECG Results", list(restecg_map.keys()), help="Results from the resting electrocardiogram.")
        
        thalch = st.number_input("Maximum Heart Rate", 60.0, 220.0, 150.0, help="Maximum heart rate achieved during exercise testing.")
        
        exang_map = {"No": 0, "Yes": 1}
        exang_choice = st.selectbox("Exercise Induced Angina", list(exang_map.keys()), help="Did exercise induce angina (chest pain)?")

    with col3:
        oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1, help="ST depression induced by exercise relative to rest.")
        
        slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
        slope_choice = st.selectbox("ST Segment Slope", list(slope_map.keys()), help="The slope of the peak exercise ST segment.")
        
        ca_map = {"0 Vessels": 0, "1 Vessel": 1, "2 Vessels": 2, "3 Vessels": 3}
        ca_choice = st.selectbox("Fluoroscopy Colored Vessels", list(ca_map.keys()), help="Number of major vessels colored by fluoroscopy.")
        
        thal_map = {"Normal": 0, "Fixed Defect": 1, "Reversable Defect": 2}
        thal_choice = st.selectbox("Thalassemia", list(thal_map.keys()), help="Blood disorder status.")

    # Submit button for the form
    submit_button = st.form_submit_button("Run Diagnostic Prediction")

# 5. Prediction Logic and Dynamic Animation Trigger
if submit_button:
    # Convert text selections back to integers for the model
    input_data = pd.DataFrame([[
        age, sex_map[sex_choice], cp_map[cp_choice], trestbps, chol, 
        fbs_map[fbs_choice], restecg_map[restecg_choice], thalch, 
        exang_map[exang_choice], oldpeak, slope_map[slope_choice], 
        ca_map[ca_choice], thal_map[thal_choice]
    ]], columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]
    
    if prediction[0] == 1:
        st.error(f"⚠️ **High Cardiovascular Risk Detected** (Model Confidence: {probability:.2%})")
        
        # High Risk Animation: Aggressive pulse, red alerts, highlighting text
        high_risk_html = """
        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 350px; width: 100%; background: radial-gradient(circle, #450a0a 0%, #0b0f19 70%); border-radius: 12px; border: 1px solid #7f1d1d; position: relative; overflow: hidden;">
            <div style="position: absolute; top: 15%; left: 10%; background: rgba(0,0,0,0.8); padding: 8px; border-radius: 4px; color: #f87171; border-left: 3px solid #ef4444; font-size: 12px;">Arterial Blockage Warning</div>
            <div style="position: absolute; bottom: 15%; right: 10%; background: rgba(0,0,0,0.8); padding: 8px; border-radius: 4px; color: #f87171; border-left: 3px solid #ef4444; font-size: 12px;">Ventricular Stress High</div>
            <img id="heart" src="https://cdn-icons-png.flaticon.com/512/873/873295.png" style="width: 220px; filter: drop-shadow(0 0 30px #ef4444); animation: fastPulse 0.6s infinite;">
            <div style="color: #ef4444; margin-top: 20px; font-weight: bold; letter-spacing: 2px; text-transform: uppercase;">Critical Anomaly Detected</div>
            <style>
                @keyframes fastPulse { 0% { transform: scale(1); } 50% { transform: scale(1.15); } 100% { transform: scale(1); } }
            </style>
        </div>
        """
        animation_placeholder.components.v1.html(high_risk_html, height=360)
        
    else:
        st.success(f"✅ **Low Risk / Normal Function** (Model Confidence: {1 - probability:.2%})")
        
        # Low Risk Animation: Smooth, calm green pulse
        low_risk_html = """
        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 350px; width: 100%; background: radial-gradient(circle, #064e3b 0%, #0b0f19 70%); border-radius: 12px; border: 1px solid #065f46; position: relative; overflow: hidden;">
            <img id="heart" src="https://cdn-icons-png.flaticon.com/512/873/873295.png" style="width: 180px; filter: drop-shadow(0 0 20px #10b981) hue-rotate(90deg); animation: calmPulse 2s infinite;">
            <div style="color: #10b981; margin-top: 20px; font-weight: bold; letter-spacing: 2px; text-transform: uppercase;">Stable Rhythm</div>
            <style>
                @keyframes calmPulse { 0% { transform: scale(1); } 50% { transform: scale(1.03); } 100% { transform: scale(1); } }
            </style>
        </div>
        """
        animation_placeholder.components.v1.html(low_risk_html, height=360)
