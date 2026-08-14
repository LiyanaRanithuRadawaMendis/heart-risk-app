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
    h1 { color: #ef4444; text-align: center; margin-bottom: 0px; padding-bottom: 0px; letter-spacing: 2px; text-transform: uppercase; }
    p { text-align: center; color: #f87171; font-size: 1.1rem; }
    
    /* Input Form Styling */
    div[data-testid="stForm"] { border: 1px solid #450a0a; background-color: #0f0202; padding: 25px; border-radius: 12px; box-shadow: 0 0 20px rgba(239, 68, 68, 0.05); }
    
    /* Submit Button Styling */
    .stButton>button { width: 100%; border-radius: 8px; background-color: #b91c1c; color: white; border: 1px solid #f87171; padding: 15px; font-weight: 900; font-size: 18px; transition: all 0.3s; margin-top: 20px; text-transform: uppercase; letter-spacing: 3px; }
    .stButton>button:hover { background-color: #7f1d1d; transform: translateY(-2px); box-shadow: 0 0 20px rgba(239, 68, 68, 0.6); border-color: #ef4444; }
    
    /* Explanatory Captions */
    .st-emotion-cache-1n76uvr { color: #991b1b !important; font-size: 0.85rem !important; margin-top: -10px; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("CARDIAC TELEMETRY AI")
st.write("Enter patient vitals to initiate predictive scanning sequence.")

# 3. Central Animation Display (Live Javascript EKG & Random BPM)
animation_placeholder = st.empty()

# We use Chart.js to draw a live, scrolling redline and vanilla JS to randomize the BPM.
live_ekg_html = """
<div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 350px; width: 100%; background: #000000; border-radius: 12px; border: 2px solid #450a0a; position: relative; overflow: hidden;">
    
    <!-- EKG Grid Background -->
    <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; background-image: linear-gradient(rgba(239, 68, 68, 0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(239, 68, 68, 0.1) 1px, transparent 1px); background-size: 20px 20px;"></div>
    
    <!-- Top Right Random BPM Display -->
    <div style="position: absolute; top: 15px; right: 25px; color: #ef4444; font-family: monospace; font-size: 32px; font-weight: bold; text-shadow: 0 0 10px #ef4444;">
        <span id="bpm">72</span> BPM
    </div>
    
    <!-- Live Chart Canvas -->
    <div style="width: 100%; height: 200px; padding: 0 20px; position: relative; z-index: 10;">
        <canvas id="ekgChart"></canvas>
    </div>
    
    <div style="color: #ef4444; margin-top: 10px; font-weight: bold; letter-spacing: 4px; font-family: monospace; font-size: 18px; animation: blink 1.5s infinite;">SYSTEM STANDBY - AWAITING INPUT</div>
    
    <style>@keyframes blink { 0% { opacity: 0.3; } 50% { opacity: 1; } 100% { opacity: 0.3; } }</style>
    
    <!-- Chart.js Library -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        // Randomize BPM between 68 and 78 every 1.5 seconds
        setInterval(() => {
            const bpmElement = document.getElementById('bpm');
            const randomBpm = Math.floor(Math.random() * (78 - 68 + 1)) + 68;
            bpmElement.innerText = randomBpm;
        }, 1500);

        // Setup the live scrolling EKG line
        const ctx = document.getElementById('ekgChart').getContext('2d');
        const dataLength = 50;
        let dataPoints = new Array(dataLength).fill(0);
        
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: new Array(dataLength).fill(''),
                datasets: [{
                    data: dataPoints,
                    borderColor: '#ef4444',
                    borderWidth: 3,
                    pointRadius: 0,
                    tension: 0.1,
                    fill: false,
                    shadowColor: '#ef4444',
                    shadowBlur: 10
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 0 }, // Disable default animation for smooth scrolling
                scales: {
                    x: { display: false },
                    y: { display: false, min: -10, max: 10 }
                },
                plugins: { legend: { display: false } }
            }
        });

        // Function to simulate EKG spikes and scroll the data
        let tick = 0;
        setInterval(() => {
            tick++;
            dataPoints.shift(); // Remove oldest point
            
            // Create a heartbeat spike pattern every 10 ticks
            if (tick % 10 === 0) { dataPoints.push(-3); }
            else if (tick % 10 === 1) { dataPoints.push(8); }
            else if (tick % 10 === 2) { dataPoints.push(-5); }
            else if (tick % 10 === 3) { dataPoints.push(2); }
            else { dataPoints.push(0); } // Flatline in between beats
            
            chart.update();
        }, 100); // Update every 100ms
    </script>
</div>
"""
with animation_placeholder:
    components.html(live_ekg_html, height=360)

st.markdown("---")

# 4. Data Entry Section using a Form
with st.form("clinical_form"):
    st.subheader("Patient Clinical Attributes")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", 1, 120, 50)
        st.caption("Patient's chronological age in years.")
        
        sex_map = {"Female": 0, "Male": 1}
        sex_choice = st.selectbox("Biological Sex", list(sex_map.keys()))
        st.caption("Assigned biological sex at birth.")
        
        cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
        cp_choice = st.selectbox("Chest Pain Type", list(cp_map.keys()))
        st.caption("Clinical classification of reported chest pain.")
        
        trestbps = st.number_input("Resting Blood Pressure", 80.0, 200.0, 120.0)
        st.caption("Resting blood pressure (mm Hg) upon admission.")
        
        chol = st.number_input("Serum Cholesterol", 100.0, 600.0, 200.0)
        st.caption("Total serum cholesterol in mg/dl.")

    with col2:
        fbs_map = {"Under 120 mg/dl (Normal)": 0, "Over 120 mg/dl (Elevated)": 1}
        fbs_choice = st.selectbox("Fasting Blood Sugar", list(fbs_map.keys()))
        st.caption("Indicates if fasting blood sugar exceeds 120 mg/dl.")
        
        restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
        restecg_choice = st.selectbox("Resting ECG Results", list(restecg_map.keys()))
        st.caption("Primary resting electrocardiogram diagnosis.")
        
        thalch = st.number_input("Maximum Heart Rate", 60.0, 220.0, 150.0)
        st.caption("Maximum heart rate achieved during exercise testing.")
        
        exang_map = {"No": 0, "Yes": 1}
        exang_choice = st.selectbox("Exercise Induced Angina", list(exang_map.keys()))
        st.caption("Did exercise induce angina (chest pain)?")

    with col3:
        oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)
        st.caption("ST depression induced by exercise relative to rest.")
        
        slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
        slope_choice = st.selectbox("ST Segment Slope", list(slope_map.keys()))
        st.caption("Slope geometry of the peak exercise ST segment.")
        
        ca_map = {"0 Vessels": 0, "1 Vessel": 1, "2 Vessels": 2, "3 Vessels": 3}
        ca_choice = st.selectbox("Fluoroscopy Colored Vessels", list(ca_map.keys()))
        st.caption("Number of major vessels illuminated by fluoroscopy.")
        
        thal_map = {"Normal": 0, "Fixed Defect": 1, "Reversable Defect": 2}
        thal_choice = st.selectbox("Thalassemia", list(thal_map.keys()))
        st.caption("Diagnosis of thalassemia blood disorder.")

    submit_button = st.form_submit_button("INITIATE SCAN")

# 5. Prediction Logic and Output Results
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
    
    st.markdown("---")
    
    if prediction[0] == 1:
        # High Risk Output: Large text, red theme, and anatomical image
        st.markdown(f"""
        <div style="background-color: #270606; padding: 40px; border-radius: 12px; border: 3px solid #ef4444; box-shadow: 0 0 30px rgba(239, 68, 68, 0.4); text-align: center;">
            <h2 style="color: #ef4444; font-size: 3rem; font-weight: 900; margin-bottom: 10px; letter-spacing: 2px;">CRITICAL RISK DETECTED</h2>
            <h3 style="color: #fca5a5; font-size: 1.5rem; margin-bottom: 30px;">Model Confidence: {probability:.2%}</h3>
            
            <div style="display: flex; justify-content: center; align-items: center; gap: 40px; text-align: left;">
                <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Blausen_0259_CoronaryArteryDisease_02.png/640px-Blausen_0259_CoronaryArteryDisease_02.png" style="width: 350px; border-radius: 8px; border: 2px solid #ef4444;">
                <div style="max-width: 500px;">
                    <h4 style="color: #ef4444; font-size: 1.8rem; margin-bottom: 10px;">Vascular Anomaly Indicated</h4>
                    <p style="color: #f87171; font-size: 1.2rem; line-height: 1.6;">
                        The machine learning model indicates an elevated probability of cardiovascular disease based on the inputted clinical indicators. 
                        The data suggests potential restricted blood flow or ventricular strain similar to the atherosclerosis shown in the reference image. 
                        Immediate clinical intervention and further testing are recommended.
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        # Low Risk Output: Large text, green theme, and anatomical image
        st.markdown(f"""
        <div style="background-color: #022c22; padding: 40px; border-radius: 12px; border: 3px solid #10b981; box-shadow: 0 0 30px rgba(16, 185, 129, 0.3); text-align: center;">
            <h2 style="color: #10b981; font-size: 3rem; font-weight: 900; margin-bottom: 10px; letter-spacing: 2px;">NORMAL CARDIAC FUNCTION</h2>
            <h3 style="color: #6ee7b7; font-size: 1.5rem; margin-bottom: 30px;">Model Confidence: {1 - probability:.2%}</h3>
            
            <div style="display: flex; justify-content: center; align-items: center; gap: 40px; text-align: left;">
                <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Heart_normal.svg/640px-Heart_normal.svg.png" style="width: 300px; filter: invert(1) hue-rotate(180deg) brightness(1.5);">
                <div style="max-width: 500px;">
                    <h4 style="color: #10b981; font-size: 1.8rem; margin-bottom: 10px;">Sinus Rhythm Stable</h4>
                    <p style="color: #34d399; font-size: 1.2rem; line-height: 1.6;">
                        The cardiovascular indicators fall well within healthy parameters. The machine learning model detects no immediate signs of arterial blockage or critical ventricular strain. 
                        Routine monitoring is advised.
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
