import os
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib

# ----------------------------------------------------------------------------
# 1. Paths, page config, model loading
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="CardioScope | Cardiovascular Risk Assessment",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE_DIR, "heart_disease_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    return model, scaler


model, scaler = load_artifacts()

FEATURE_ORDER = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                  "thalch", "exang", "oldpeak", "slope", "ca", "thal"]

FEATURE_LABELS = {
    "age": "Age", "sex": "Biological sex", "cp": "Chest pain type",
    "trestbps": "Resting blood pressure", "chol": "Serum cholesterol",
    "fbs": "Fasting blood sugar > 120", "restecg": "Resting ECG",
    "thalch": "Max heart rate", "exang": "Exercise-induced angina",
    "oldpeak": "ST depression (oldpeak)", "slope": "ST segment slope",
    "ca": "Vessels colored by fluoroscopy", "thal": "Thalassemia",
}

# population reference stats pulled straight from the fitted scaler,
# so the "typical range" hints always match the data the model was trained on
SCALER_MEAN = dict(zip(FEATURE_ORDER, getattr(scaler, "mean_", [None] * len(FEATURE_ORDER))))
SCALER_STD = dict(zip(FEATURE_ORDER, getattr(scaler, "scale_", [None] * len(FEATURE_ORDER))))

# ----------------------------------------------------------------------------
# 2. Risk tiers — single source of truth for color / labels used everywhere
# ----------------------------------------------------------------------------
TIERS = {
    "standby": {"light": "#5EEAD4", "dark": "#0F766E", "glow": "45,212,191",
                "label": "Awaiting data", "sub": "Enter vitals to run an assessment"},
    "low": {"light": "#6EE7B7", "dark": "#059669", "glow": "16,185,129",
            "label": "Low risk", "sub": "No strong risk indicators detected"},
    "moderate": {"light": "#FCD34D", "dark": "#D97706", "glow": "245,158,11",
                 "label": "Moderate risk", "sub": "Some indicators warrant attention"},
    "high": {"light": "#FDA4AF", "dark": "#E11D48", "glow": "244,63,94",
             "label": "Elevated risk", "sub": "Multiple indicators are concerning"},
}


def tier_from_probability(p):
    if p is None:
        return "standby"
    if p < 0.34:
        return "low"
    if p < 0.67:
        return "moderate"
    return "high"


def beat_duration(bpm):
    """Seconds per animation cycle, clamped so nothing pulses fast enough to
    be a photosensitivity concern (never faster than ~2.2 beats/sec)."""
    if not bpm or bpm <= 0:
        return 1.1
    return max(60.0 / bpm, 0.45)


def hexs(hexcolor):
    return hexcolor.replace("#", "%23")


# ----------------------------------------------------------------------------
# 3. Global theme — injected once, styles native Streamlit widgets
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root{
  --bg:#080c14; --panel:#0e1420; --panel-2:#121a2a; --border:#1e2a3d;
  --text:#dfe7f2; --text-muted:#7d8ba3; --teal:#2dd4bf;
}

.stApp, [data-testid="stAppViewContainer"], .main{ background:var(--bg); }
html, body, [class*="css"]{ font-family:'Space Grotesk', sans-serif; color:var(--text); }
h1,h2,h3{ font-family:'Space Grotesk', sans-serif; letter-spacing:.2px; }

.app-eyebrow{ color:var(--teal); font-family:'IBM Plex Mono',monospace; font-size:12.5px;
  letter-spacing:3px; text-transform:uppercase; text-align:center; margin-bottom:2px; }
.app-title{ color:var(--text); text-align:center; font-size:2.1rem; font-weight:600; margin:0 0 6px 0; }
.app-sub{ color:var(--text-muted); text-align:center; font-size:.95rem; margin-bottom:1.6rem; }

div[data-testid="stForm"]{
  background:var(--panel); border:1px solid var(--border); border-radius:14px;
  padding:26px 26px 8px 26px;
}
div[data-testid="stForm"] label p{ color:var(--text-muted) !important; font-size:.82rem !important;
  text-transform:uppercase; letter-spacing:.6px; }
div[data-testid="stForm"] h3{ color:var(--text); font-size:1rem; font-weight:600;
  border-bottom:1px solid var(--border); padding-bottom:8px; margin-bottom:14px; }

input, select, textarea{ background:var(--panel-2) !important; color:var(--text) !important;
  border:1px solid var(--border) !important; border-radius:8px !important; }
div[data-baseweb="select"] > div{ background:var(--panel-2) !important; border-color:var(--border) !important; }

.stButton>button{
  width:100%; border-radius:9px; background:var(--teal); color:#04231d; border:none;
  padding:12px; font-weight:600; font-size:15px; letter-spacing:.4px; margin-top:14px;
  transition:transform .15s ease, box-shadow .15s ease;
}
.stButton>button:hover{ transform:translateY(-1px); box-shadow:0 6px 20px rgba(45,212,191,.25); }

.field-hint{ color:var(--text-muted); font-family:'IBM Plex Mono',monospace; font-size:11.5px;
  margin:-10px 0 10px 2px; }

/* panels used for the insight cards on the right column */
.panel-card{ background:var(--panel); border:1px solid var(--border); border-radius:14px;
  padding:18px 20px; margin-bottom:16px; }
.panel-title{ font-size:.78rem; text-transform:uppercase; letter-spacing:1.2px;
  color:var(--text-muted); margin-bottom:12px; }

.bar-row{ display:flex; align-items:center; gap:10px; margin-bottom:10px; }
.bar-label{ width:190px; flex-shrink:0; font-size:.83rem; color:var(--text); }
.bar-track{ flex:1; height:8px; background:var(--panel-2); border-radius:5px; overflow:hidden; }
.bar-fill{ height:100%; border-radius:5px; background:linear-gradient(90deg,#0f766e,#2dd4bf); }
.bar-value{ width:42px; text-align:right; font-family:'IBM Plex Mono',monospace;
  font-size:.78rem; color:var(--text-muted); }

.chip-row{ display:flex; flex-wrap:wrap; gap:8px; }
.chip{ background:var(--panel-2); border:1px solid var(--border); border-radius:8px;
  padding:8px 12px; font-size:.78rem; color:var(--text-muted); }
.chip b{ color:var(--text); font-family:'IBM Plex Mono',monospace; font-weight:600; }

.disclaimer{ color:var(--text-muted); font-size:.78rem; text-align:center;
  border-top:1px solid var(--border); padding-top:14px; margin-top:22px; line-height:1.6; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 4. Header
# ----------------------------------------------------------------------------
st.markdown("""
<div class="app-eyebrow">Machine-learning screening tool</div>
<div class="app-title">CardioScope</div>
<div class="app-sub">Enter clinical values to estimate cardiovascular disease risk from a trained model.</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# 5. Vitals monitor component (heart + ECG + gauge) — single HTML/SVG bundle
# ----------------------------------------------------------------------------
def render_monitor(bpm, probability):
    tier_key = tier_from_probability(probability)
    tier = TIERS[tier_key]
    dur = beat_duration(bpm)
    active = probability is not None
    beat_anim = "beat" if active else "breathe"
    ripple_display = "block" if active else "none"

    # gauge geometry
    half_circ = 251.2  # pi * r, r = 80
    prob_val = probability if probability is not None else 0.0
    offset = half_circ * (1 - prob_val)
    needle_deg = (prob_val * 180) - 90
    gauge_text = f"{prob_val * 100:.0f}%" if active else "—"

    ecg_color = hexs(tier["dark"])

    return f"""
    <div style="font-family:'Space Grotesk',sans-serif; background:transparent;">
    <style>
      .mon-wrap{{ display:flex; align-items:center; gap:22px; padding:10px 4px; }}
      .heart-box{{ position:relative; width:170px; height:160px; flex-shrink:0; }}
      @keyframes beat{{
        0%{{transform:scale(1);}} 14%{{transform:scale(1.09);}} 28%{{transform:scale(0.98);}}
        42%{{transform:scale(1.05);}} 70%{{transform:scale(1);}} 100%{{transform:scale(1);}}
      }}
      @keyframes breathe{{
        0%{{transform:scale(1);}} 50%{{transform:scale(1.035);}} 100%{{transform:scale(1);}}
      }}
      .heart-shape{{ transform-box:fill-box; transform-origin:center;
        animation:{beat_anim} {dur}s ease-in-out infinite; }}
      @keyframes ripple{{ 0%{{transform:scale(.8); opacity:.45;}} 100%{{transform:scale(2); opacity:0;}} }}
      .ripple{{ transform-box:fill-box; transform-origin:center; fill:none;
        stroke:{tier['dark']}; stroke-width:2; animation:ripple {dur}s ease-out infinite;
        display:{ripple_display}; }}
      .ripple-2{{ animation-delay:{dur/2}s; }}
      .bpm-tag{{ font-family:'IBM Plex Mono',monospace; font-size:22px; font-weight:600;
        color:{tier['light']}; text-align:center; margin-top:-6px; }}
      .bpm-unit{{ font-size:11px; color:var(--text-muted,#7d8ba3); letter-spacing:2px; }}
      .status-dot{{ width:8px; height:8px; border-radius:50%; background:{tier['light']};
        display:inline-block; margin-right:7px; box-shadow:0 0 8px rgba({tier['glow']},.8); }}
      .status-label{{ font-size:15px; font-weight:600; color:{tier['light']}; }}
      .status-sub{{ font-size:12.5px; color:var(--text-muted,#7d8ba3); margin:2px 0 10px 15px; }}
      .ecg-track{{ width:100%; height:56px; margin-top:6px;
        background-image:url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="220" height="90"><path d="M0,50 L40,50 Q56,30 72,50 L86,50 L90,58 L94,8 L98,52 L102,66 L106,50 L112,50 Q130,24 150,50 L220,50" fill="none" stroke="{ecg_color}" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/></svg>');
        background-repeat:repeat-x; background-size:220px 56px;
        animation:ecgslide {dur}s linear infinite; }}
      @keyframes ecgslide{{ from{{background-position:0 0;}} to{{background-position:-220px 0;}} }}
      .gauge-wrap{{ flex-shrink:0; text-align:center; }}
      @keyframes fillarc{{ from{{stroke-dashoffset:251.2;}} to{{stroke-dashoffset:{offset};}} }}
      .gauge-fg{{ stroke-dashoffset:{offset}; animation:fillarc 1.1s ease-out; }}
      .needle{{ transform:rotate({needle_deg}deg); transform-origin:100px 100px;
        transition:transform 1s ease-out; }}
      .gauge-num{{ font-family:'IBM Plex Mono',monospace; font-size:26px; font-weight:600; fill:{tier['light']}; }}
      .gauge-cap{{ font-size:10.5px; letter-spacing:1.5px; fill:var(--text-muted,#7d8ba3); text-transform:uppercase; }}
    </style>

    <div class="mon-wrap">
      <div class="heart-box">
        <svg viewBox="0 0 220 200" width="170" height="160" style="overflow:visible;">
          <defs>
            <radialGradient id="hg" cx="35%" cy="28%" r="75%">
              <stop offset="0%" stop-color="{tier['light']}"/>
              <stop offset="100%" stop-color="{tier['dark']}"/>
            </radialGradient>
            <filter id="glow" x="-60%" y="-60%" width="220%" height="220%">
              <feGaussianBlur stdDeviation="5" result="b"/>
              <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
          </defs>
          <circle class="ripple ripple-1" cx="110" cy="100" r="52"/>
          <circle class="ripple ripple-2" cx="110" cy="100" r="52"/>
          <g stroke="{tier['dark']}" stroke-width="9" fill="none" stroke-linecap="round" opacity="0.5">
            <path d="M95,45 C95,20 75,10 60,15"/>
            <path d="M125,45 C130,18 150,12 165,20"/>
          </g>
          <g class="heart-shape" filter="url(#glow)">
            <path d="M110,182 C110,182 32,132 32,80 C32,52 55,32 80,32 C93,32 103,40 110,55
                     C117,40 127,32 140,32 C165,32 188,52 188,80 C188,132 110,182 110,182 Z"
                  fill="url(#hg)" stroke="{tier['dark']}" stroke-width="2"/>
            <path d="M70,55 C60,65 55,80 58,95" fill="none" stroke="rgba(255,255,255,.35)"
                  stroke-width="6" stroke-linecap="round"/>
            <path d="M110,55 L110,165" stroke="rgba(0,0,0,.15)" stroke-width="2"/>
          </g>
        </svg>
        <div class="bpm-tag">{int(bpm) if bpm else '--'} <span class="bpm-unit">BPM</span></div>
      </div>

      <div style="flex:1; min-width:180px;">
        <div><span class="status-dot"></span><span class="status-label">{tier['label']}</span></div>
        <div class="status-sub">{tier['sub']}</div>
        <div class="ecg-track"></div>
      </div>

      <div class="gauge-wrap">
        <svg viewBox="0 0 200 115" width="150" height="90">
          <path d="M20,100 A80,80 0 0 1 180,100" fill="none" stroke="var(--panel-2,#121a2a)" stroke-width="14" stroke-linecap="round"/>
          <path class="gauge-fg" d="M20,100 A80,80 0 0 1 180,100" fill="none" stroke="{tier['light']}"
                stroke-width="14" stroke-linecap="round" stroke-dasharray="251.2"/>
          <line class="needle" x1="100" y1="100" x2="100" y2="30" stroke="{tier['light']}" stroke-width="3" stroke-linecap="round"/>
          <circle cx="100" cy="100" r="4" fill="{tier['light']}"/>
          <text x="100" y="85" text-anchor="middle" class="gauge-num">{gauge_text}</text>
          <text x="100" y="112" text-anchor="middle" class="gauge-cap">risk score</text>
        </svg>
      </div>
    </div>
    </div>
    """


# ----------------------------------------------------------------------------
# 6. Insight panels — model feature importance + population comparison
# ----------------------------------------------------------------------------
def render_feature_importance():
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return ""
    order = np.argsort(importances)[::-1][:6]
    max_imp = importances[order[0]]
    rows = ""
    for i in order:
        feat = FEATURE_ORDER[i]
        pct = importances[i] / max_imp * 100
        rows += f"""
        <div class="bar-row">
          <div class="bar-label">{FEATURE_LABELS.get(feat, feat)}</div>
          <div class="bar-track"><div class="bar-fill" style="width:{pct:.0f}%;"></div></div>
          <div class="bar-value">{importances[i]*100:.1f}%</div>
        </div>"""
    return f"""
    <div class="panel-card">
      <div class="panel-title">What this model weighs most</div>
      {rows}
    </div>"""


def render_population_chips(values):
    chips = ""
    for feat, label, fmt in [
        ("trestbps", "Resting BP", "{:.0f}"), ("chol", "Cholesterol", "{:.0f}"),
        ("thalch", "Max heart rate", "{:.0f}"), ("age", "Age", "{:.0f}"),
    ]:
        mean = SCALER_MEAN.get(feat)
        if mean is None:
            continue
        chips += f"""<div class="chip">{label}: <b>{fmt.format(values[feat])}</b>
        &nbsp;·&nbsp; population avg {fmt.format(mean)}</div>"""
    return f"""
    <div class="panel-card">
      <div class="panel-title">Your values vs. training population average</div>
      <div class="chip-row">{chips}</div>
    </div>"""


# ----------------------------------------------------------------------------
# 7. Layout: intake form (left) + live vitals monitor (right)
# ----------------------------------------------------------------------------
left, right = st.columns([1.15, 1], gap="large")

with right:
    monitor_slot = st.empty()
    insight_slot = st.container()
    with monitor_slot:
        components.html(render_monitor(bpm=72, probability=None), height=200)

with left:
    with st.form("clinical_form"):
        st.markdown("<h3>Demographics</h3>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", 1, 120, 50)
            sex_map = {"Female": 0, "Male": 1}
            sex_choice = st.selectbox("Biological sex", list(sex_map.keys()))
        with c2:
            cp_map = {"Typical angina": 0, "Atypical angina": 1, "Non-anginal pain": 2, "Asymptomatic": 3}
            cp_choice = st.selectbox("Chest pain type", list(cp_map.keys()))
            fbs_map = {"Under 120 mg/dl": 0, "Over 120 mg/dl": 1}
            fbs_choice = st.selectbox("Fasting blood sugar", list(fbs_map.keys()))

        st.markdown("<h3>Vitals & labs</h3>", unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            trestbps = st.number_input("Resting blood pressure (mm Hg)", 80.0, 200.0, 120.0)
            st.markdown(f"<div class='field-hint'>typical ~{SCALER_MEAN['trestbps']:.0f} mm Hg</div>", unsafe_allow_html=True)
            chol = st.number_input("Serum cholesterol (mg/dl)", 100.0, 600.0, 200.0)
            st.markdown(f"<div class='field-hint'>typical ~{SCALER_MEAN['chol']:.0f} mg/dl</div>", unsafe_allow_html=True)
        with c4:
            thalch = st.number_input("Max heart rate achieved", 60.0, 220.0, 150.0)
            st.markdown(f"<div class='field-hint'>typical ~{SCALER_MEAN['thalch']:.0f} bpm</div>", unsafe_allow_html=True)
            restecg_map = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
            restecg_choice = st.selectbox("Resting ECG", list(restecg_map.keys()))

        st.markdown("<h3>Exercise & cardiac testing</h3>", unsafe_allow_html=True)
        c5, c6 = st.columns(2)
        with c5:
            exang_map = {"No": 0, "Yes": 1}
            exang_choice = st.selectbox("Exercise-induced angina", list(exang_map.keys()))
            oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
            slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
            slope_choice = st.selectbox("ST segment slope", list(slope_map.keys()))
        with c6:
            ca_map = {"0 vessels": 0, "1 vessel": 1, "2 vessels": 2, "3 vessels": 3}
            ca_choice = st.selectbox("Vessels colored by fluoroscopy", list(ca_map.keys()))
            thal_map = {"Normal": 0, "Fixed defect": 1, "Reversable defect": 2}
            thal_choice = st.selectbox("Thalassemia", list(thal_map.keys()))

        submitted = st.form_submit_button("Run assessment")

# ----------------------------------------------------------------------------
# 8. Prediction + updating the monitor / insight panels
# ----------------------------------------------------------------------------
if submitted:
    row = {
        "age": age, "sex": sex_map[sex_choice], "cp": cp_map[cp_choice],
        "trestbps": trestbps, "chol": chol, "fbs": fbs_map[fbs_choice],
        "restecg": restecg_map[restecg_choice], "thalch": thalch,
        "exang": exang_map[exang_choice], "oldpeak": oldpeak,
        "slope": slope_map[slope_choice], "ca": ca_map[ca_choice],
        "thal": thal_map[thal_choice],
    }
    input_data = pd.DataFrame([[row[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)
    input_scaled = scaler.transform(input_data)
    probability = float(model.predict_proba(input_scaled)[0][1])

    with monitor_slot:
        components.html(render_monitor(bpm=thalch, probability=probability), height=200)

    with insight_slot:
        st.markdown(render_feature_importance(), unsafe_allow_html=True)
        st.markdown(render_population_chips(row), unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
CardioScope is a machine-learning screening aid trained on historical clinical data.
It does not diagnose disease and is not a substitute for professional medical evaluation —
share these results with a clinician rather than acting on them alone.
</div>
""", unsafe_allow_html=True)

