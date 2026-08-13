import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load the saved model and scaler
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')

# 2. Set Page Config & UI Design
st.set_page_config(page_title="Heart Risk Predictor", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #f4f6f9; }
    h1 { color: #1e3a8a; text-align: center; font-family: 'Times New Roman', Times, serif; }
    .stButton>button { width: 100%; border-radius: 8px; background-color: #2563eb; color: white; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("🫀 AI Heart Disease Risk Prediction System")
st.write("Enter the patient's clinical details below to predict cardiovascular risk.")

# 3. Create input fields for the required clinical attributes
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", value=120.0)
    chol = st.number_input("Serum Cholesterol (mg/dl)", value=200.0)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (0 = False, 1 = True)", [0, 1])

with col2:
    restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
    thalch = st.number_input("Maximum Heart Rate Achieved", value=150.0)
    exang = st.selectbox("Exercise Induced Angina (0 = No, 1 = Yes)", [0, 1])
    oldpeak = st.number_input("ST Depression Induced by Exercise", value=1.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (0 = Normal, 1 = Fixed, 2 = Reversable)", [0, 1, 2])

# 4. Prediction Logic
if st.button("Predict Risk"):
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak, slope, ca, thal]], 
                              columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
    
    # Scale the input data using the saved scaler
    input_scaled = scaler.transform(input_data)
    
    # Generate prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]
    
    st.markdown("---")
    if prediction[0] == 1:
        st.error(f"⚠️ **High Risk Detected:** The model predicts a presence of heart disease. (Confidence: {probability:.2%})")
    else:
        st.success(f"✅ **Low Risk:** The model predicts no presence of heart disease. (Confidence: {1 - probability:.2%})")
