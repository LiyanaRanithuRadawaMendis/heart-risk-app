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

# 2. Page Configuration & Dark Red/Black Theme
st.set_page_config(page_title="AI Cardiovascular Engine", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    /* Main Background & Text */
    .main { background-color: #050000; color: #fca5a5; font-family: 'Inter', sans-serif; }
    h1 { color: #ef4444; text-align: center; margin-bottom: 0px; padding-bottom: 0px; letter-spacing: 1px; }
    p { text-align: center; color: #f87171; font-size: 1.1rem; }
    
    /* Input Form Styling */
    div[data-testid="stForm"] { border: 1px solid #450a0a; background-color: #0f0202; padding: 25px; border-radius: 12px; box-shadow: 0 0 20px rgba(239, 68, 68, 0.05); }
    
    /* Submit Button Styling */
    .stButton>button { width: 100%; border-radius: 8px; background-color: #b91c1c; color: white; border: 1px solid #f87171; padding: 12px; font-weight: bold; font-size: 16px; transition: all 0.3s; margin-top: 20px; text-transform: uppercase; letter-spacing: 2px; }
    .stButton>button:hover { background-color: #7f1d1d; transform: translateY(-2px); box-shadow: 0 0 15px rgba(239, 68, 68, 0.5); border-color: #ef4444; }
    </style>
""", unsafe_allow_html=True)

st.title("🩸 CARDIAC TELEMETRY AI")
st.write("Enter patient vitals to initiate predictive scanning sequence.")

# 3. Central Animation Display (The EKG Redline)
animation_placeholder = st.empty()

# Default State: Standby EKG
default_ekg = """
<div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 350px; width: 100%; background: #000000; border-radius: 12px; border: 2px solid #450a0a; position: relative; overflow: hidden;">
    
    <!-- EKG Grid Background -->
    <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; background-image: linear-gradient(rgba(239, 68, 68, 0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(239, 68, 68, 0.1) 1px, transparent 1px); background-size: 20px 20px;"></div>
    
    <!-- Top Right BPM Display -->
    <div style="position: absolute; top: 15px; right: 25px; color: #ef4444; font-family: monospace; font-size: 32px; font-weight: bold;">
        ❤️ 72 BPM
    </div>
    
    <!-- Animated EKG Line (Pure CSS/SVG) -->
    <div style="width: 100%; height: 150px; background-image: url('data:image/svg+xml;utf8,<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"300\" height=\"150\"><path d=\"M0,75 L100,75 L110,40 L125,130 L140,75 L300,75\" fill=\"none\" stroke=\"%23ef4444\" stroke-width=\"4\" stroke-linejoin=\"round\"/></svg>'); background-repeat: repeat-x; animation: monitorSlide 3s linear infinite; filter: drop-shadow(0 0 8px #ef4444);"></div>
    
    <div style="color: #ef4444; margin-top: 20px; font-weight: bold; letter-spacing: 4px; font-family: monospace; font-size: 18px; animation: blink 1.5s infinite;">SYSTEM STANDBY - AWAITING INPUT</div>
    
    <style>
        @keyframes monitorSlide { from { background-position: 0 0; } to { background-position: -300px 0; } }
        @keyframes blink { 0% { opacity: 0.3; } 50% { opacity: 1; } 100% { opacity: 0.3; } }
    </style>
</div>
"""
with animation_placeholder:
    components.html(default_ekg, height=360)

st.markdown("---")

# 4. Data Entry Section using a Form
with st.form("clinical_form"):
    st.subheader("Patient Clinical Attributes")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", 1, 120, 50)
        sex_map = {"Female": 0, "Male": 1}
        sex_choice = st.selectbox("Biological Sex", list(sex_map.keys()))
        cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
        cp_choice = st.selectbox("Chest Pain Type", list(cp_map.keys()))
        trestbps = st.number_input("Resting Blood Pressure", 80.0, 200.0, 120.0)
        chol = st.number_input("Serum Cholesterol", 100.0, 600.0, 200.0)

    with col2:
        fbs_map = {"Under 120 mg/dl (Normal)": 0, "Over 120 mg/dl (Elevated)": 1}
        fbs_choice = st.selectbox("Fasting Blood Sugar", list(fbs_map.keys()))
        restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
        restecg_choice = st.selectbox("Resting ECG Results", list(restecg_map.keys()))
        thalch = st.number_input("Maximum Heart Rate", 60.0, 220.0, 150.0)
        exang_map = {"No": 0, "Yes": 1}
        exang_choice = st.selectbox("Exercise Induced Angina", list(exang_map.keys()))

    with col3:
        oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)
        slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
        slope_choice = st.selectbox("ST Segment Slope", list(slope_map.keys()))
        ca_map = {"0 Vessels": 0, "1 Vessel": 1, "2 Vessels": 2, "3 Vessels": 3}
        ca_choice = st.selectbox("Fluoroscopy Colored Vessels", list(ca_map.keys()))
        thal_map = {"Normal": 0, "Fixed Defect": 1, "Reversable Defect": 2}
        thal_choice = st.selectbox("Thalassemia", list(thal_map.keys()))

    submit_button = st.form_submit_button("INITIATE SCAN")

# 5. Prediction Logic and Dynamic EKG Update
if submit_button:
    input_data = pd.DataFrame([[
        age, sex_map[sex_choice], cp_map[cp_choice], trestbps, chol, 
        fbs_map[fbs_choice], restecg_map[restecg_choice], thalch, 
        exang_map[exang_choice], oldpeak, slope_map[slope_choice], 
        ca_map[ca_choice], thal_map[thal_choice]
    ]], columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]
    
    # Grab the user's inputted Max Heart Rate to display on the monitor!
    display_bpm = int(thalch)
    
    if prediction[0] == 1:
        st.error(f"⚠️ **CRITICAL ALERT:** Elevated risk of cardiovascular disease. (Confidence: {probability:.2%})")
        
        # High Risk EKG: Fast, erratic line with issue description
        high_risk_ekg = f"""
        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 350px; width: 100%; background: #1a0000; border-radius: 12px; border: 3px solid #ef4444; position: relative; overflow: hidden; animation: bgFlash 0.8s infinite;">
            <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; background-image: linear-gradient(rgba(239, 68, 68, 0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(239, 68, 68, 0.1) 1px, transparent 1px); background-size: 20px 20px;"></div>
            
            <div style="position: absolute; top: 15px; right: 25px; color: #ef4444; font-family: monospace; font-size: 32px; font-weight: bold; text-shadow: 0 0 10px #ef4444;">
                ⚠️ {display_bpm} BPM
            </div>
            
            <!-- Warning Description Box -->
            <div style="position: absolute; top: 15px; left: 25px; background: rgba(0,0,0,0.8); padding: 10px 15px; border-left: 5px solid #ef4444; color: #fca5a5; font-family: sans-serif; font-size: 14px; max-width: 300px;">
                <b style="color: #ef4444; font-size: 16px;">Vascular Anomaly Detected</b><br>
                Model indicates elevated probability of coronary artery disease or compromised ventricular function based on clinical indicators.
            </div>

            <!-- Faster, Erratic EKG Line -->
            <div style="width: 100%; height: 150px; background-image: url('data:image/svg+xml;utf8,<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"200\" height=\"150\"><path d=\"M0,75 L40,75 L50,10 L70,140 L90,75 L200,75\" fill=\"none\" stroke=\"%23ef4444\" stroke-width=\"5\" stroke-linejoin=\"round\"/></svg>'); background-repeat: repeat-x; animation: monitorSlide 0.7s linear infinite; filter: drop-shadow(0 0 12px #ef4444);"></div>
            
            <div style="color: #ef4444; margin-top: 20px; font-weight: 900; letter-spacing: 5px; font-family: monospace; font-size: 24px; text-shadow: 0 0 15px #ef4444;">CRITICAL ABNORMALITY</div>
            
            <style>
                @keyframes monitorSlide {{ from {{ background-position: 0 0; }} to {{ background-position: -200px 0; }} }}
                @keyframes bgFlash {{ 0% {{ background-color: #1a0000; }} 50% {{ background-color: #3f0000; }} 100% {{ background-color: #1a0000; }} }}
            </style>
        </div>
        """
        with animation_placeholder:
            components.html(high_risk_ekg, height=360)
        
    else:
        st.success(f"✅ **NORMAL FUNCTION:** No critical risk detected. (Confidence: {1 - probability:.2%})")
        
        # Low Risk EKG: Smooth green line with normal description
        low_risk_ekg = f"""
        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 350px; width: 100%; background: #000000; border-radius: 12px; border: 2px solid #064e3b; position: relative; overflow: hidden;">
            <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; background-image: linear-gradient(rgba(16, 185, 129, 0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(16, 185, 129, 0.1) 1px, transparent 1px); background-size: 20px 20px;"></div>
            
            <div style="position: absolute; top: 15px; right: 25px; color: #10b981; font-family: monospace; font-size: 32px; font-weight: bold; text-shadow: 0 0 10px #10b981;">
                ✅ {display_bpm} BPM
            </div>
            
            <!-- Clear Description Box -->
            <div style="position: absolute; top: 15px; left: 25px; background: rgba(0,0,0,0.8); padding: 10px 15px; border-left: 5px solid #10b981; color: #6ee7b7; font-family: sans-serif; font-size: 14px; max-width: 300px;">
                <b style="color: #10b981; font-size: 16px;">Sinus Rhythm Normal</b><br>
                Cardiovascular indicators fall within healthy parameters. No immediate signs of arterial blockage.
            </div>

            <!-- Smooth Green EKG Line -->
            <div style="width: 100%; height: 150px; background-image: url('data:image/svg+xml;utf8,<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"300\" height=\"150\"><path d=\"M0,75 L100,75 L110,50 L125,120 L140,75 L300,75\" fill=\"none\" stroke=\"%2310b981\" stroke-width=\"4\" stroke-linejoin=\"round\"/></svg>'); background-repeat: repeat-x; animation: monitorSlide 3s linear infinite; filter: drop-shadow(0 0 10px #10b981);"></div>
            
            <div style="color: #10b981; margin-top: 20px; font-weight: bold; letter-spacing: 4px; font-family: monospace; font-size: 18px;">VITALS STABLE</div>
            
            <style>
                @keyframes monitorSlide {{ from {{ background-position: 0 0; }} to {{ background-position: -300px 0; }} }}
            </style>
        </div>
        """
        with animation_placeholder:
            components.html(low_risk_ekg, height=360)
