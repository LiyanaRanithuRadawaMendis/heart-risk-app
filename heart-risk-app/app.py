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

# which broad category each feature speaks to, used to pick which
# explanation / diagram zone to surface after a prediction
FEATURE_CATEGORY = {
    "age": "metabolic", "sex": "metabolic", "fbs": "metabolic",
    "cp": "coronary", "exang": "coronary", "oldpeak": "coronary",
    "slope": "coronary", "ca": "coronary",
    "restecg": "rhythm", "thalch": "rhythm",
    "trestbps": "vascular", "chol": "vascular", "thal": "vascular",
}

CATEGORY_INFO = {
    "coronary": {"label": "Coronary blood flow", "zone": (140, 48),
                 "text": "driven mainly by markers of blood flow to the heart muscle during exertion "
                         "(chest pain pattern, ST depression, blocked-vessel count). These are classic "
                         "signs of narrowed coronary arteries limiting oxygen supply under load."},
    "rhythm": {"label": "Heart rhythm & response", "zone": (78, 92),
               "text": "driven mainly by the heart's electrical activity and how its rate responds to "
                       "exertion. An abnormal resting ECG or a blunted heart-rate response can both point "
                       "toward reduced cardiac reserve."},
    "vascular": {"label": "Vascular system", "zone": (152, 112),
                 "text": "driven mainly by the broader vascular system. Blood pressure and cholesterol "
                         "reflect the condition of blood vessels throughout the body, which affects the "
                         "heart indirectly over time."},
    "metabolic": {"label": "Overall / metabolic profile", "zone": None,
                  "text": "driven mainly by general demographic and metabolic risk (age, sex, blood sugar) "
                          "rather than one specific mechanical or electrical finding."},
}

SCALER_MEAN = dict(zip(FEATURE_ORDER, getattr(scaler, "mean_", [None] * len(FEATURE_ORDER))))
SCALER_STD = dict(zip(FEATURE_ORDER, getattr(scaler, "scale_", [None] * len(FEATURE_ORDER))))

HELP = {
    "age": "Patient age in years.",
    "sex": "Biological sex recorded for this patient.",
    "cp": "Typical angina: classic exertional chest pain. Atypical angina: chest pain that doesn't fully "
          "fit the classic pattern. Non-anginal pain: chest pain unlikely to be heart-related. "
          "Asymptomatic: no chest pain reported.",
    "trestbps": "Blood pressure measured at rest, in mm Hg (the top number of a normal reading).",
    "chol": "Serum cholesterol level in mg/dl from a blood test.",
    "fbs": "Whether fasting blood sugar was measured above 120 mg/dl.",
    "restecg": "Resting electrocardiogram result: Normal, ST-T wave abnormality, or signs of left "
               "ventricular hypertrophy (enlarged heart muscle).",
    "thalch": "The highest heart rate reached during a stress/exercise test.",
    "exang": "Whether chest pain (angina) was triggered specifically by exercise.",
    "oldpeak": "ST depression on the ECG induced by exercise relative to rest — a marker doctors use to "
               "gauge reduced blood flow during exertion. Higher values indicate more depression.",
    "slope": "The slope of the ST segment during peak exercise: upsloping is generally lowest risk, "
             "downsloping generally highest.",
    "ca": "Number of major blood vessels (0-3) shown to be narrowed on a fluoroscopy scan with dye.",
    "thal": "Thalassemia stress-test result: Normal, Fixed defect (permanent reduced blood flow), or "
            "Reversable defect (reduced blood flow that improves at rest).",
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
# 2. Risk tiers. Brand chrome (buttons, headings, standby state) uses red as
#    requested. Low / moderate / high keep green -> amber -> red, since that
#    traffic-light convention is what makes the result readable at a glance;
#    making "low risk" red too would undercut that. Flag if you'd rather have
#    all four states on a single red scale instead.
# ----------------------------------------------------------------------------
ACCENT = "#DC2626"
ACCENT_DARK = "#7F1D1D"

TIERS = {
    "standby": {"light": "#F87171", "dark": "#7F1D1D", "glow": "220,38,38",
                "label": "Awaiting data", "sub": "Enter vitals to run an assessment"},
    "low": {"light": "#6EE7B7", "dark": "#059669", "glow": "16,185,129",
            "label": "Low risk", "sub": "No strong risk indicators detected"},
    "moderate": {"light": "#FCD34D", "dark": "#D97706", "glow": "245,158,11",
                 "label": "Moderate risk", "sub": "Some indicators warrant attention"},
    "high": {"light": "#FCA5A5", "dark": "#B91C1C", "glow": "220,38,38",
             "label": "Elevated risk", "sub": "Multiple indicators are concerning"},
}

# ----------------------------------------------------------------------------
# 3. Global theme
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root{
  --bg:#0c0808; --panel:#150e0e; --panel-2:#1a1212; --border:#2e1c1c;
  --text:#f2e7e7; --text-muted:#a68a8a; --accent:#DC2626;
}

.stApp, [data-testid="stAppViewContainer"], .main{ background:var(--bg); }
html, body, [class*="css"]{ font-family:'Space Grotesk', sans-serif; color:var(--text); }
h1,h2,h3{ font-family:'Space Grotesk', sans-serif; letter-spacing:.2px; }

.app-eyebrow{ color:var(--accent); font-family:'IBM Plex Mono',monospace; font-size:12.5px;
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
  width:100%; border-radius:9px; background:var(--accent); color:#fff; border:none;
  padding:12px; font-weight:600; font-size:15px; letter-spacing:.4px; margin-top:14px;
  transition:transform .15s ease, box-shadow .15s ease;
}
.stButton>button:hover{ transform:translateY(-1px); box-shadow:0 6px 20px rgba(220,38,38,.3); }

.field-hint{ color:var(--text-muted); font-family:'IBM Plex Mono',monospace; font-size:11.5px;
  margin:-10px 0 10px 2px; }

.panel-card{ background:var(--panel); border:1px solid var(--border); border-radius:14px;
  padding:18px 20px; margin-bottom:16px; }
.panel-title{ font-size:.78rem; text-transform:uppercase; letter-spacing:1.2px;
  color:var(--text-muted); margin-bottom:12px; }

.bar-row{ display:flex; align-items:center; gap:10px; margin-bottom:10px; }
.bar-label{ width:190px; flex-shrink:0; font-size:.83rem; color:var(--text); }
.bar-track{ flex:1; height:8px; background:var(--panel-2); border-radius:5px; overflow:hidden; }
.bar-fill{ height:100%; border-radius:5px; background:linear-gradient(90deg,#7f1d1d,#dc2626); }
.bar-value{ width:42px; text-align:right; font-family:'IBM Plex Mono',monospace;
  font-size:.78rem; color:var(--text-muted); }

.chip-row{ display:flex; flex-wrap:wrap; gap:8px; }
.chip{ background:var(--panel-2); border:1px solid var(--border); border-radius:8px;
  padding:8px 12px; font-size:.78rem; color:var(--text-muted); }
.chip b{ color:var(--text); font-family:'IBM Plex Mono',monospace; font-weight:600; }

.note-row{ font-size:.85rem; color:var(--text); margin-bottom:8px; padding-left:14px; position:relative; }
.note-row:before{ content:"-"; position:absolute; left:0; color:var(--accent); }

.disclaimer{ color:var(--text-muted); font-size:.78rem; text-align:center;
  border-top:1px solid var(--border); padding-top:14px; margin-top:22px; line-height:1.6; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div class='app-eyebrow'>Machine-learning screening tool</div>"
    "<div class='app-title'>CardioScope</div>"
    "<div class='app-sub'>Enter clinical values to estimate cardiovascular disease risk from a trained model.</div>",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# 4. Vitals monitor component (heart + ECG + gauge) — rendered inside its own
#    iframe via components.html, so raw HTML/CSS here is always safe.
# ----------------------------------------------------------------------------
def render_monitor(bpm, probability):
    tier_key = tier_from_probability(probability)
    tier = TIERS[tier_key]
    dur = beat_duration(bpm)
    active = probability is not None
    beat_anim = "beat" if active else "breathe"
    ripple_display = "block" if active else "none"

    half_circ = 251.2
    prob_val = probability if probability is not None else 0.0
    offset = half_circ * (1 - prob_val)
    needle_deg = (prob_val * 180) - 90
    gauge_text = f"{prob_val * 100:.0f}%" if active else "--"
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
      .bpm-unit{{ font-size:11px; color:#a68a8a; letter-spacing:2px; }}
      .status-dot{{ width:8px; height:8px; border-radius:50%; background:{tier['light']};
        display:inline-block; margin-right:7px; box-shadow:0 0 8px rgba({tier['glow']},.8); }}
      .status-label{{ font-size:15px; font-weight:600; color:{tier['light']}; }}
      .status-sub{{ font-size:12.5px; color:#a68a8a; margin:2px 0 10px 15px; }}
      .ecg-track{{ width:100%; height:56px; margin-top:6px;
        background-image:url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="220" height="90"><path d="M0,50 L40,50 Q56,30 72,50 L86,50 L90,58 L94,8 L98,52 L102,66 L106,50 L112,50 Q130,24 150,50 L220,50" fill="none" stroke="{ecg_color}" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/></svg>');
        background-repeat:repeat-x; background-size:220px 56px;
        animation:ecgslide {dur}s linear infinite; }}
      @keyframes ecgslide{{ from{{background-position:0 0;}} to{{background-position:-220px 0;}} }}
      .gauge-wrap{{ flex-shrink:0; text-align:center; }}
      .gauge-fg{{ stroke-dashoffset:{offset}; }}
      .needle{{ transform:rotate({needle_deg}deg); transform-origin:100px 100px;
        transition:transform 1s ease-out; }}
      .gauge-num{{ font-family:'IBM Plex Mono',monospace; font-size:26px; font-weight:600; fill:{tier['light']}; }}
      .gauge-cap{{ font-size:10.5px; letter-spacing:1.5px; fill:#a68a8a; text-transform:uppercase; }}
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
          <path d="M20,100 A80,80 0 0 1 180,100" fill="none" stroke="#1a1212" stroke-width="14" stroke-linecap="round"/>
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
# 5. Insight panels — built as SINGLE-LINE html strings. Streamlit's markdown
#    renderer treats any line indented 4+ spaces as a preformatted code block
#    (a CommonMark rule), which is what caused raw tags to print on screen
#    last time. Keeping each fragment on one line, with no leading
#    whitespace, avoids that entirely.
# ----------------------------------------------------------------------------
def render_feature_importance():
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return ""
    order = np.argsort(importances)[::-1][:6]
    max_imp = importances[order[0]]
    rows = "".join(
        f'<div class="bar-row"><div class="bar-label">{FEATURE_LABELS.get(FEATURE_ORDER[i], FEATURE_ORDER[i])}</div>'
        f'<div class="bar-track"><div class="bar-fill" style="width:{importances[i] / max_imp * 100:.0f}%;"></div></div>'
        f'<div class="bar-value">{importances[i] * 100:.1f}%</div></div>'
        for i in order
    )
    return f'<div class="panel-card"><div class="panel-title">What this model weighs most</div>{rows}</div>'


def render_population_chips(values):
    parts = []
    for feat, label, fmt in [("trestbps", "Resting BP", "{:.0f}"), ("chol", "Cholesterol", "{:.0f}"),
                              ("thalch", "Max heart rate", "{:.0f}"), ("age", "Age", "{:.0f}")]:
        mean = SCALER_MEAN.get(feat)
        if mean is None:
            continue
        parts.append(f'<div class="chip">{label}: <b>{fmt.format(values[feat])}</b> &nbsp;/&nbsp; population avg {fmt.format(mean)}</div>')
    return f'<div class="panel-card"><div class="panel-title">Your values vs. training population average</div><div class="chip-row">{"".join(parts)}</div></div>'


def top_category_for(row):
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return "metabolic", []
    scores = {}
    for i, feat in enumerate(FEATURE_ORDER):
        mean, std = SCALER_MEAN.get(feat), SCALER_STD.get(feat)
        if mean is not None and std:
            z = min(abs((row[feat] - mean) / std), 3.0)
            scores[feat] = importances[i] * max(z, 0.35)
        else:
            scores[feat] = importances[i]
    top_feat = max(scores, key=scores.get)
    category = FEATURE_CATEGORY.get(top_feat, "metabolic")

    notes = []
    for feat in ["trestbps", "chol", "thalch", "oldpeak", "age"]:
        mean, std = SCALER_MEAN.get(feat), SCALER_STD.get(feat)
        if mean is None or not std:
            continue
        z = (row[feat] - mean) / std
        if abs(z) < 0.5:
            continue
        direction = "above" if z > 0 else "below"
        notes.append((abs(z), f'{FEATURE_LABELS[feat]} ({row[feat]:g}) is {direction} the typical value in this dataset (~{mean:.0f}).'))
    notes.sort(key=lambda t: -t[0])
    return category, [n[1] for n in notes[:3]]


def render_zone_diagram(category_key, tier):
    info = CATEGORY_INFO[category_key]
    zone = info["zone"]
    marker = ""
    if zone:
        cx, cy = zone
        marker = (f'<circle cx="{cx}" cy="{cy}" r="16" fill="none" stroke="{tier["light"]}" stroke-width="2" opacity="0.9">'
                   f'<animate attributeName="r" values="10;20;10" dur="2.4s" repeatCount="indefinite"/>'
                   f'<animate attributeName="opacity" values="0.9;0.2;0.9" dur="2.4s" repeatCount="indefinite"/></circle>'
                   f'<circle cx="{cx}" cy="{cy}" r="4" fill="{tier["light"]}"/>')
        glow_filter = ""
    else:
        marker = ""
        glow_filter = 'filter="drop-shadow(0 0 10px rgba(220,38,38,0.55))"'
    return (f'<svg viewBox="0 0 220 200" width="150" height="140" style="overflow:visible;flex-shrink:0;">'
            f'<g {glow_filter} stroke="{tier["dark"]}" stroke-width="9" fill="none" stroke-linecap="round" opacity="0.4">'
            f'<path d="M95,45 C95,20 75,10 60,15"/><path d="M125,45 C130,18 150,12 165,20"/></g>'
            f'<path d="M110,182 C110,182 32,132 32,80 C32,52 55,32 80,32 C93,32 103,40 110,55 '
            f'C117,40 127,32 140,32 C165,32 188,52 188,80 C188,132 110,182 110,182 Z" '
            f'fill="#241616" stroke="{tier["dark"]}" stroke-width="2" {glow_filter}/>'
            f'{marker}</svg>')


def render_explanation(row, probability, tier_key):
    tier = TIERS[tier_key]
    category, notes = top_category_for(row)
    info = CATEGORY_INFO[category]
    notes_html = "".join(f'<div class="note-row">{n}</div>' for n in notes) or '<div class="note-row">No single value stands far outside the typical range; risk here reflects a combination of smaller factors.</div>'
    diagram = render_zone_diagram(category, tier)
    intro = f'This assessment is <b style="color:{tier["light"]};">{tier["label"].lower()}</b> ({probability * 100:.0f}% predicted probability), {info["text"]}'
    return (f'<div class="panel-card"><div class="panel-title">Result explained</div>'
            f'<div style="display:flex; gap:18px; align-items:flex-start;">{diagram}'
            f'<div style="flex:1;"><div style="font-size:.9rem; line-height:1.6; margin-bottom:12px;">{intro}</div>'
            f'<div style="font-size:.72rem; text-transform:uppercase; letter-spacing:1px; color:#a68a8a; margin-bottom:6px;">Values furthest from typical</div>'
            f'{notes_html}</div></div></div>')


# ----------------------------------------------------------------------------
# 6. Optional 3D heart (experimental). This is a stylized, schematic model
#    built from simple geometry with animated flow particles and a highlight
#    marker — not an anatomically precise medical rendering. Accurate 3D
#    anatomy normally comes from licensed medical asset libraries, which
#    isn't something that can be generated from scratch here. I wasn't able
#    to render-test WebGL in my sandbox either (no browser/GPU access there),
#    so check this view once after you deploy.
# ----------------------------------------------------------------------------
def render_3d_heart(tier_key, category_key):
    tier = TIERS[tier_key]
    info = CATEGORY_INFO[category_key]
    zone_label = info["label"]
    light = tier["light"]
    dark = tier["dark"]
    # Pre-compute the JS zone literal here (not inline in the f-string below) so this
    # stays compatible with Python < 3.12, which doesn't allow an f-string expression
    # to reuse the same quote character as the string's own delimiter.
    if info["zone"]:
        zx = (info["zone"][0] - 110) / 28.2
        zy = (107 - info["zone"][1]) / 28.2
        zone_js = f"[{zx},{zy},0.9]"
    else:
        zone_js = "null"
    return f"""
    <div id="heart3d-wrap" style="width:100%; text-align:center;">
      <div id="heart3d-canvas" style="width:100%; height:340px; border-radius:12px; overflow:hidden; background:radial-gradient(circle at 50% 30%, #201414, #0c0808);"></div>
      <div id="heart3d-caption" style="font-size:.78rem; color:#a68a8a; margin-top:10px;">Loading 3D view…</div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script>
    (function() {{
      function init() {{
        var mount = document.getElementById('heart3d-canvas');
        var caption = document.getElementById('heart3d-caption');
        if (!window.THREE || !mount) {{ if (caption) caption.textContent = '3D view failed to load.'; return; }}
        try {{
          var W = mount.clientWidth, H = mount.clientHeight;
          var scene = new THREE.Scene();
          var camera = new THREE.PerspectiveCamera(45, W / H, 0.1, 100);
          camera.position.set(0, 0.6, 9);
          var renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
          renderer.setPixelRatio(window.devicePixelRatio || 1);
          renderer.setSize(W, H);
          mount.appendChild(renderer.domElement);

          scene.add(new THREE.AmbientLight(0xffffff, 0.55));
          var dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
          dirLight.position.set(4, 6, 8);
          scene.add(dirLight);
          var glow = new THREE.PointLight(new THREE.Color('{light}'), 1.1, 12);
          glow.position.set(-2, 1, 4);
          scene.add(glow);

          var shape = new THREE.Shape();
          shape.moveTo(0, -75);
          shape.bezierCurveTo(0, -75, -78, -25, -78, 27);
          shape.bezierCurveTo(-78, 55, -55, 75, -30, 75);
          shape.bezierCurveTo(-17, 75, -7, 67, 0, 52);
          shape.bezierCurveTo(7, 67, 17, 75, 30, 75);
          shape.bezierCurveTo(55, 75, 78, 55, 78, 27);
          shape.bezierCurveTo(78, -25, 0, -75, 0, -75);
          var geo = new THREE.ExtrudeGeometry(shape, {{ depth: 40, bevelEnabled: true, bevelThickness: 8, bevelSize: 8, bevelSegments: 4 }});
          geo.center();
          geo.scale(0.025, 0.025, 0.025);
          var mat = new THREE.MeshStandardMaterial({{
            color: new THREE.Color('{light}'), emissive: new THREE.Color('{dark}'),
            emissiveIntensity: 0.35, roughness: 0.45, metalness: 0.08
          }});
          var heart = new THREE.Mesh(geo, mat);
          scene.add(heart);

          function makeVessel(points, color) {{
            var curve = new THREE.CatmullRomCurve3(points);
            var tubeGeo = new THREE.TubeGeometry(curve, 40, 0.13, 8, false);
            var tubeMat = new THREE.MeshStandardMaterial({{ color: color, roughness: 0.5, metalness: 0.1, transparent: true, opacity: 0.55 }});
            scene.add(new THREE.Mesh(tubeGeo, tubeMat));
            return curve;
          }}
          var aorta = makeVessel([
            new THREE.Vector3(0.1, 1.7, 0.2), new THREE.Vector3(-0.4, 2.6, 0.4), new THREE.Vector3(-1.6, 2.9, 0.3)
          ], new THREE.Color('#b91c1c'));
          var pulm = makeVessel([
            new THREE.Vector3(0.5, 1.7, 0.2), new THREE.Vector3(1.0, 2.6, 0.4), new THREE.Vector3(1.9, 2.7, 0.2)
          ], new THREE.Color('#2563eb'));

          var particles = [];
          [[aorta, '#fca5a5', 7], [pulm, '#93c5fd', 7]].forEach(function(cfg) {{
            var curve = cfg[0], color = cfg[1], count = cfg[2];
            for (var i = 0; i < count; i++) {{
              var g = new THREE.SphereGeometry(0.05, 8, 8);
              var m = new THREE.MeshBasicMaterial({{ color: color }});
              var sphere = new THREE.Mesh(g, m);
              scene.add(sphere);
              particles.push({{ mesh: sphere, curve: curve, offset: i / count }});
            }}
          }});

          var zone = {zone_js};
          var marker = null;
          if (zone) {{
            var ring = new THREE.Mesh(
              new THREE.TorusGeometry(0.22, 0.035, 8, 32),
              new THREE.MeshBasicMaterial({{ color: new THREE.Color('{light}') }})
            );
            ring.position.set(zone[0], zone[1], zone[2]);
            scene.add(ring);
            marker = ring;
          }}

          var controls = new THREE.OrbitControls(camera, renderer.domElement);
          controls.enableDamping = true;
          controls.autoRotate = true;
          controls.autoRotateSpeed = 1.1;
          controls.enableZoom = true;
          controls.minDistance = 5;
          controls.maxDistance = 14;

          var clock = new THREE.Clock();
          function animate() {{
            requestAnimationFrame(animate);
            var t = clock.getElapsedTime();
            particles.forEach(function(p) {{
              var pos = p.curve.getPointAt((t * 0.18 + p.offset) % 1);
              p.mesh.position.copy(pos);
            }});
            if (marker) {{
              var s = 1 + 0.25 * Math.sin(t * 3);
              marker.scale.set(s, s, s);
              marker.rotation.z += 0.01;
            }}
            controls.update();
            renderer.render(scene, camera);
          }}
          animate();
          caption.textContent = 'Highlighted: {zone_label} — drag to rotate, scroll to zoom.';

          window.addEventListener('resize', function() {{
            var w = mount.clientWidth, h = mount.clientHeight;
            camera.aspect = w / h; camera.updateProjectionMatrix();
            renderer.setSize(w, h);
          }});
        }} catch (e) {{
          caption.textContent = '3D view unavailable in this browser.';
        }}
      }}
      if (document.readyState === 'complete') {{ setTimeout(init, 200); }}
      else {{ window.addEventListener('load', function() {{ setTimeout(init, 200); }}); }}
    }})();
    </script>
    """


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
            age = st.number_input("Age", min_value=1, max_value=120, value=None, help=HELP["age"])
            sex_map = {"Female": 0, "Male": 1}
            sex_choice = st.selectbox("Biological sex", list(sex_map.keys()), index=None,
                                       placeholder="Select...", help=HELP["sex"])
        with c2:
            cp_map = {"Typical angina": 0, "Atypical angina": 1, "Non-anginal pain": 2, "Asymptomatic": 3}
            cp_choice = st.selectbox("Chest pain type", list(cp_map.keys()), index=None,
                                      placeholder="Select...", help=HELP["cp"])
            fbs_map = {"Under 120 mg/dl": 0, "Over 120 mg/dl": 1}
            fbs_choice = st.selectbox("Fasting blood sugar", list(fbs_map.keys()), index=None,
                                       placeholder="Select...", help=HELP["fbs"])

        st.markdown("<h3>Vitals & labs</h3>", unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            trestbps = st.number_input("Resting blood pressure (mm Hg)", min_value=80.0, max_value=200.0,
                                        value=None, help=HELP["trestbps"])
            st.markdown(f"<div class='field-hint'>typical ~{SCALER_MEAN['trestbps']:.0f} mm Hg</div>", unsafe_allow_html=True)
            chol = st.number_input("Serum cholesterol (mg/dl)", min_value=100.0, max_value=600.0,
                                    value=None, help=HELP["chol"])
            st.markdown(f"<div class='field-hint'>typical ~{SCALER_MEAN['chol']:.0f} mg/dl</div>", unsafe_allow_html=True)
        with c4:
            thalch = st.number_input("Max heart rate achieved", min_value=60.0, max_value=220.0,
                                      value=None, help=HELP["thalch"])
            st.markdown(f"<div class='field-hint'>typical ~{SCALER_MEAN['thalch']:.0f} bpm</div>", unsafe_allow_html=True)
            restecg_map = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
            restecg_choice = st.selectbox("Resting ECG", list(restecg_map.keys()), index=None,
                                           placeholder="Select...", help=HELP["restecg"])

        st.markdown("<h3>Exercise & cardiac testing</h3>", unsafe_allow_html=True)
        c5, c6 = st.columns(2)
        with c5:
            exang_map = {"No": 0, "Yes": 1}
            exang_choice = st.selectbox("Exercise-induced angina", list(exang_map.keys()), index=None,
                                         placeholder="Select...", help=HELP["exang"])
            oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=6.0,
                                       value=None, step=0.1, help=HELP["oldpeak"])
            slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
            slope_choice = st.selectbox("ST segment slope", list(slope_map.keys()), index=None,
                                         placeholder="Select...", help=HELP["slope"])
        with c6:
            ca_map = {"0 vessels": 0, "1 vessel": 1, "2 vessels": 2, "3 vessels": 3}
            ca_choice = st.selectbox("Vessels colored by fluoroscopy", list(ca_map.keys()), index=None,
                                      placeholder="Select...", help=HELP["ca"])
            thal_map = {"Normal": 0, "Fixed defect": 1, "Reversable defect": 2}
            thal_choice = st.selectbox("Thalassemia", list(thal_map.keys()), index=None,
                                        placeholder="Select...", help=HELP["thal"])

        submitted = st.form_submit_button("Run assessment")

# ----------------------------------------------------------------------------
# 8. Prediction + updating the monitor / insight panels
# ----------------------------------------------------------------------------
if submitted:
    raw = {
        "age": age, "sex": sex_map.get(sex_choice), "cp": cp_map.get(cp_choice),
        "trestbps": trestbps, "chol": chol, "fbs": fbs_map.get(fbs_choice),
        "restecg": restecg_map.get(restecg_choice), "thalch": thalch,
        "exang": exang_map.get(exang_choice), "oldpeak": oldpeak,
        "slope": slope_map.get(slope_choice), "ca": ca_map.get(ca_choice),
        "thal": thal_map.get(thal_choice),
    }
    missing = [FEATURE_LABELS[f] for f in FEATURE_ORDER if raw[f] is None]

    if missing:
        st.error("Please fill in every field before running an assessment. Missing: " + ", ".join(missing))
    else:
        input_data = pd.DataFrame([[raw[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)
        input_scaled = scaler.transform(input_data)
        probability = float(model.predict_proba(input_scaled)[0][1])
        tier_key = tier_from_probability(probability)
        category, _ = top_category_for(raw)

        with monitor_slot:
            components.html(render_monitor(bpm=raw["thalch"], probability=probability), height=200)

        with insight_slot:
            st.markdown(render_explanation(raw, probability, tier_key), unsafe_allow_html=True)
            st.markdown(render_feature_importance(), unsafe_allow_html=True)
            st.markdown(render_population_chips(raw), unsafe_allow_html=True)

        with st.expander("View interactive 3D heart model"):
            st.caption("Stylized schematic model, not medical-grade anatomy")
            components.html(render_3d_heart(tier_key, category), height=390)

st.markdown("""
<div class="disclaimer">
CardioScope is a machine-learning screening aid trained on historical clinical data.
It does not diagnose disease and is not a substitute for professional medical evaluation.
</div>
""", unsafe_allow_html=True)
