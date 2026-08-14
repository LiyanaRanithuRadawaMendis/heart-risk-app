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
    .main { background-color: #050000; color: #fca5a5; font-family: 'Inter', sans-serif; }
    h1 { color: #ef4444; text-align: center; margin-bottom: 0px; padding-bottom: 0px; letter-spacing: 2px; text-transform: uppercase; }
    p { text-align: center; color: #f87171; font-size: 1.1rem; }
    div[data-testid="stForm"] { border: 1px solid #450a0a; background-color: #0f0202; padding: 25px; border-radius: 12px; box-shadow: 0 0 20px rgba(239, 68, 68, 0.05); }
    .stButton>button { width: 100%; border-radius: 8px; background-color: #b91c1c; color: white; border: 1px solid #f87171; padding: 15px; font-weight: 900; font-size: 18px; transition: all 0.3s; margin-top: 20px; text-transform: uppercase; letter-spacing: 3px; }
    .stButton>button:hover { background-color: #7f1d1d; transform: translateY(-2px); box-shadow: 0 0 20px rgba(239, 68, 68, 0.6); border-color: #ef4444; }
    </style>
""", unsafe_allow_html=True)

st.title("CARDIAC TELEMETRY AI")
st.write("Enter patient vitals to initiate predictive scanning sequence.")

# 3. Create the Main Monitor Display Area (Fixed Height)
animation_placeholder = st.empty()

# 4. Data Entry Section using a Form (No preselected defaults, Tooltips restored)
with st.form("clinical_form"):
    st.subheader("Patient Clinical Attributes")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=None, step=1, help="Patient's chronological age in years.")
        
        sex_map = {"Female": 0, "Male": 1}
        sex_choice = st.selectbox("Biological Sex", list(sex_map.keys()), index=None, placeholder="Select sex...", help="Assigned biological sex at birth.")
        
        cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
        cp_choice = st.selectbox("Chest Pain Type", list(cp_map.keys()), index=None, placeholder="Select chest pain type...", help="Clinical classification of reported chest pain.")
        
        trestbps = st.number_input("Resting Blood Pressure", min_value=50.0, max_value=250.0, value=None, help="Resting blood pressure (mm Hg) upon admission.")
        
        chol = st.number_input("Serum Cholesterol", min_value=50.0, max_value=600.0, value=None, help="Total serum cholesterol in mg/dl.")

    with col2:
        fbs_map = {"Under 120 mg/dl (Normal)": 0, "Over 120 mg/dl (Elevated)": 1}
        fbs_choice = st.selectbox("Fasting Blood Sugar", list(fbs_map.keys()), index=None, placeholder="Select blood sugar status...", help="Indicates if fasting blood sugar exceeds 120 mg/dl.")
        
        restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
        restecg_choice = st.selectbox("Resting ECG Results", list(restecg_map.keys()), index=None, placeholder="Select ECG results...", help="Primary resting electrocardiogram diagnosis.")
        
        thalch = st.number_input("Maximum Heart Rate", min_value=50.0, max_value=250.0, value=None, help="Maximum heart rate achieved during exercise testing.")
        
        exang_map = {"No": 0, "Yes": 1}
        exang_choice = st.selectbox("Exercise Induced Angina", list(exang_map.keys()), index=None, placeholder="Select status...", help="Did exercise induce angina (chest pain)?")

    with col3:
        oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=None, step=0.1, help="ST depression induced by exercise relative to rest.")
        
        slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
        slope_choice = st.selectbox("ST Segment Slope", list(slope_map.keys()), index=None, placeholder="Select slope...", help="Slope geometry of the peak exercise ST segment.")
        
        ca_map = {"0 Vessels": 0, "1 Vessel": 1, "2 Vessels": 2, "3 Vessels": 3}
        ca_choice = st.selectbox("Fluoroscopy Colored Vessels", list(ca_map.keys()), index=None, placeholder="Select vessels...", help="Number of major vessels illuminated by fluoroscopy.")
        
        thal_map = {"Normal": 0, "Fixed Defect": 1, "Reversable Defect": 2}
        thal_choice = st.selectbox("Thalassemia", list(thal_map.keys()), index=None, placeholder="Select diagnosis...", help="Diagnosis of thalassemia blood disorder.")

    submit_button = st.form_submit_button("INITIATE SCAN")

# Check if all inputs are filled out before predicting
inputs_list = [age, sex_choice, cp_choice, trestbps, chol, fbs_choice, restecg_choice, thalch, exang_choice, oldpeak, slope_choice, ca_choice, thal_choice]
is_form_complete = all(v is not None for v in inputs_list)

# 5. Dynamic Monitor Logic (Swaps between Standby, High Risk, and Low Risk)
if submit_button:
    if not is_form_complete:
        st.error("⚠️ **SYSTEM OVERRIDE:** Please fill in all clinical attributes before initiating the scan.")
        
        # Keep standby screen active if form fails
        show_standby = True
    else:
        show_standby = False
        
        # Process the model prediction
        input_data = pd.DataFrame([[
            age, sex_map[sex_choice], cp_map[cp_choice], trestbps, chol, 
            fbs_map[fbs_choice], restecg_map[restecg_choice], thalch, 
            exang_map[exang_choice], oldpeak, slope_map[slope_choice], 
            ca_map[ca_choice], thal_map[thal_choice]
        ]], columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
        
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0][1]
        
        display_bpm = int(thalch)

        if prediction[0] == 1:
            # High Risk Output (Replaces the top monitor entirely)
            high_risk_html = f"""
            <div style="display: flex; flex-direction: column; align-items: center; height: 580px; width: 100%; background: #1a0000; border-radius: 12px; border: 3px solid #ef4444; position: relative; overflow: hidden; font-family: sans-serif;">
                
                <!-- Top EKG Bar -->
                <div style="width: 100%; height: 200px; position: relative; border-bottom: 2px solid #ef4444; background: #000;">
                    <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; background-image: linear-gradient(rgba(239, 68, 68, 0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(239, 68, 68, 0.1) 1px, transparent 1px); background-size: 20px 20px;"></div>
                    <div style="position: absolute; top: 15px; right: 25px; color: #ef4444; font-family: monospace; font-size: 32px; font-weight: bold; text-shadow: 0 0 10px #ef4444;">⚠️ {display_bpm} BPM</div>
                    <div style="width: 100%; height: 200px; background-image: url('data:image/svg+xml;utf8,<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"200\" height=\"200\"><path d=\"M0,100 L40,100 L50,20 L70,180 L90,100 L200,100\" fill=\"none\" stroke=\"%23ef4444\" stroke-width=\"5\" stroke-linejoin=\"round\"/></svg>'); background-repeat: repeat-x; animation: monitorSlide 0.7s linear infinite; filter: drop-shadow(0 0 12px #ef4444);"></div>
                </div>
                
                <!-- Bottom Results Section -->
                <div style="display: flex; justify-content: center; align-items: center; gap: 40px; padding: 30px; width: 100%; height: 380px;">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Blausen_0259_CoronaryArteryDisease_02.png/640px-Blausen_0259_CoronaryArteryDisease_02.png" style="height: 280px; border-radius: 8px; border: 2px solid #ef4444; box-shadow: 0 0 20px rgba(239,68,68,0.4);">
                    <div style="max-width: 50%;">
                        <h2 style="color: #ef4444; margin: 0 0 10px 0; font-size: 32px; font-weight: 900; letter-spacing: 2px;">CRITICAL RISK DETECTED</h2>
                        <h3 style="color: #fca5a5; margin: 0 0 20px 0; font-size: 20px;">Model Confidence: {probability:.2%}</h3>
                        <h4 style="color: #ef4444; margin: 0 0 10px 0; font-size: 22px;">Vascular Anomaly Indicated</h4>
                        <p style="color: #f87171; font-size: 16px; line-height: 1.6; margin: 0;">The machine learning model indicates an elevated probability of cardiovascular disease based on the inputted clinical indicators. The data suggests potential restricted blood flow or ventricular strain similar to the atherosclerosis shown. Immediate clinical intervention and further testing are recommended.</p>
                    </div>
                </div>
                <style>@keyframes monitorSlide {{ from {{ background-position: 0 0; }} to {{ background-position: -200px 0; }} }}</style>
            </div>
            """
            with animation_placeholder:
                components.html(high_risk_html, height=600)

        else:
            # Low Risk Output (Replaces the top monitor entirely)
            low_risk_html = f"""
            <div style="display: flex; flex-direction: column; align-items: center; height: 580px; width: 100%; background: #022c22; border-radius: 12px; border: 3px solid #10b981; position: relative; overflow: hidden; font-family: sans-serif;">
                
                <!-- Top EKG Bar -->
                <div style="width: 100%; height: 200px; position: relative; border-bottom: 2px solid #10b981; background: #000;">
                    <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; background-image: linear-gradient(rgba(16, 185, 129, 0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(16, 185, 129, 0.1) 1px, transparent 1px); background-size: 20px 20px;"></div>
                    <div style="position: absolute; top: 15px; right: 25px; color: #10b981; font-family: monospace; font-size: 32px; font-weight: bold; text-shadow: 0 0 10px #10b981;">✅ {display_bpm} BPM</div>
                    <div style="width: 100%; height: 200px; background-image: url('data:image/svg+xml;utf8,<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"300\" height=\"200\"><path d=\"M0,100 L100,100 L110,60 L125,160 L140,100 L300,100\" fill=\"none\" stroke=\"%2310b981\" stroke-width=\"4\" stroke-linejoin=\"round\"/></svg>'); background-repeat: repeat-x; animation: monitorSlide 3s linear infinite; filter: drop-shadow(0 0 10px #10b981);"></div>
                </div>
                
                <!-- Bottom Results Section -->
                <div style="display: flex; justify-content: center; align-items: center; gap: 40px; padding: 30px; width: 100%; height: 380px;">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Heart_normal.svg/640px-Heart_normal.svg.png" style="height: 250px; filter: invert(1) hue-rotate(180deg) brightness(1.5);">
                    <div style="max-width: 50%;">
                        <h2 style="color: #10b981; margin: 0 0 10px 0; font-size: 32px; font-weight: 900; letter-spacing: 2px;">NORMAL CARDIAC FUNCTION</h2>
                        <h3 style="color: #6ee7b7; margin: 0 0 20px 0; font-size: 20px;">Model Confidence: {1 - probability:.2%}</h3>
                        <h4 style="color: #10b981; margin: 0 0 10px 0; font-size: 22px;">Sinus Rhythm Stable</h4>
                        <p style="color: #34d399; font-size: 16px; line-height: 1.6; margin: 0;">The cardiovascular indicators fall well within healthy parameters. The machine learning model detects no immediate signs of arterial blockage or critical ventricular strain. Routine monitoring is advised.</p>
                    </div>
                </div>
                <style>@keyframes monitorSlide {{ from {{ background-position: 0 0; }} to {{ background-position: -300px 0; }} }}</style>
            </div>
            """
            with animation_placeholder:
                components.html(low_risk_html, height=600)
else:
    show_standby = True

# 6. Render the Default Standby Screen (if form not submitted or validation failed)
if show_standby:
    standby_html = """
    <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 580px; width: 100%; background: #000000; border-radius: 12px; border: 2px solid #450a0a; position: relative; overflow: hidden;">
        <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; background-image: linear-gradient(rgba(239, 68, 68, 0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(239, 68, 68, 0.1) 1px, transparent 1px); background-size: 20px 20px;"></div>
        
        <div style="position: absolute; top: 20px; right: 30px; color: #ef4444; font-family: monospace; font-size: 36px; font-weight: bold; text-shadow: 0 0 10px #ef4444;">
            <span id="bpm">72</span> BPM
        </div>
        
        <div style="width: 100%; height: 300px; padding: 0 20px; position: relative; z-index: 10;">
            <canvas id="ekgChart"></canvas>
        </div>
        
        <div style="color: #ef4444; margin-top: 40px; font-weight: bold; letter-spacing: 5px; font-family: monospace; font-size: 22px; animation: blink 1.5s infinite;">SYSTEM STANDBY - AWAITING INPUT</div>
        
        <style>@keyframes blink { 0% { opacity: 0.3; } 50% { opacity: 1; } 100% { opacity: 0.3; } }</style>
        
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script>
            setInterval(() => {
                const bpmElement = document.getElementById('bpm');
                const randomBpm = Math.floor(Math.random() * (78 - 68 + 1)) + 68;
                bpmElement.innerText = randomBpm;
            }, 1500);

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
                        borderWidth: 4,
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
                    animation: { duration: 0 },
                    scales: {
                        x: { display: false },
                        y: { display: false, min: -10, max: 10 }
                    },
                    plugins: { legend: { display: false } }
                }
            });

            let tick = 0;
            setInterval(() => {
                tick++;
                dataPoints.shift();
                
                if (tick % 10 === 0) { dataPoints.push(-3); }
                else if (tick % 10 === 1) { dataPoints.push(8); }
                else if (tick % 10 === 2) { dataPoints.push(-5); }
                else if (tick % 10 === 3) { dataPoints.push(2); }
                else { dataPoints.push(0); }
                
                chart.update();
            }, 100);
        </script>
    </div>
    """
    with animation_placeholder:
        components.html(standby_html, height=600)
