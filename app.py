""" Improved Streamlit app for Vitamin C degradation prediction Features added:

Better validation and units handling

Option to load kinetic parameters from JSON/CSV

Optional user-supplied C0

Plot of Ct vs time with uncertainty band

Half-life calculation

Presets for common processes

Safer pickle loading fallback

Session history of predictions

Clear UI layout with columns


Author: Umar Faruk Zakariyyya | BnZackx Date: 2025 """

import streamlit as st import numpy as np import pandas as pd import json import os import math import matplotlib.pyplot as plt from typing import Dict, Any, Optional import pickle

-------------------- Constants & Defaults --------------------

R_GAS = 8.314  # J/(mol*K) DEFAULT_KINETIC_FILE_JSON = "kinetics.json" MODEL_FILENAME = "vitamin_c_predictor.pkl"  # optional pickle-based model

Default kinetic parameters (units: C0 in mg/100g, Ea in J/mol, A in s^-1)

DEFAULT_KINETIC_PARAMETERS: Dict[str, Dict[str, Any]] = { 'Orange (Citrus sinensis)': {'C0': 52.3, 'Ea': 68.4 * 1000, 'A': 2.34e10}, 'Baobab (Adansonia digitata)': {'C0': 225.8, 'Ea': 72.9 * 1000, 'A': 1.07e11}, 'Fluted pumpkin (Telfairia occidentalis)': {'C0': 85.2, 'Ea': 66.8 * 1000, 'A': 8.45e9}, 'Spinach (Amaranthus hybridus)': {'C0': 62.4, 'Ea': 75.2 * 1000, 'A': 3.89e11} }

-------------------- Helper Classes --------------------

class VitaminCPredictor: """Predict Vitamin C Ct using first-order kinetics and Arrhenius law. Assumes A is in s^-1; returns k in min^-1 for time in minutes. """

def __init__(self, parameters: Dict[str, Dict[str, Any]]):
    self.parameters = parameters

def _validate_crop(self, crop: str) -> bool:
    return crop in self.parameters

def get_rate_constant(self, crop_type: str, temp_celsius: float, A_unit: str = 's^-1') -> float:
    """Return k in min^-1. A_unit may be 's^-1' or 'min^-1'."""
    if not self._validate_crop(crop_type):
        raise KeyError(f"Unknown crop type: {crop_type}")

    params = self.parameters[crop_type]
    Ea = float(params['Ea'])
    A = float(params['A'])

    if temp_celsius <= -273.15:
        raise ValueError("Temperature must be greater than absolute zero.")

    T_K = temp_celsius + 273.15
    k_s = A * math.exp(-Ea / (R_GAS * T_K))

    if A_unit == 's^-1':
        k_min = k_s * 60.0
    elif A_unit == 'min^-1':
        k_min = k_s
    else:
        raise ValueError("A_unit must be 's^-1' or 'min^-1'")

    return float(k_min)

def predict(self, crop_type: str, temp_celsius: float, time_min: float, C0_override: Optional[float] = None,
            uncertainty_frac: float = 0.0) -> Dict[str, float]:
    """Return a dict with Ct, Ct_low, Ct_high, k_min, half_life_min"""
    if not self._validate_crop(crop_type):
        raise KeyError(f"Unknown crop type: {crop_type}")

    params = self.parameters[crop_type]
    C0 = float(params.get('C0', 0.0)) if C0_override is None else float(C0_override)

    k = self.get_rate_constant(crop_type, temp_celsius)
    Ct = C0 * math.exp(-k * time_min)

    if uncertainty_frac and uncertainty_frac > 0:
        Ct_low = Ct * (1 - uncertainty_frac)
        Ct_high = Ct * (1 + uncertainty_frac)
    else:
        Ct_low = Ct_high = Ct

    half_life_min = math.log(2) / k if k > 0 else float('inf')

    return {
        'C0': C0,
        'Ct': max(0.0, Ct),
        'Ct_low': max(0.0, Ct_low),
        'Ct_high': max(0.0, Ct_high),
        'k_min': k,
        'half_life_min': half_life_min
    }

def simulate_curve(self, crop_type: str, temp_celsius: float, times_min: np.ndarray,
                   C0_override: Optional[float] = None, uncertainty_frac: float = 0.0) -> pd.DataFrame:
    rows = []
    for t in times_min:
        d = self.predict(crop_type, temp_celsius, float(t), C0_override=C0_override,
                         uncertainty_frac=uncertainty_frac)
        rows.append({'time_min': t, 'Ct': d['Ct'], 'Ct_low': d['Ct_low'], 'Ct_high': d['Ct_high']})
    return pd.DataFrame(rows)

-------------------- Utilities --------------------

@st.cache_resource def load_kinetics_from_json(path: str) -> Dict[str, Dict[str, Any]]: if not os.path.exists(path): return DEFAULT_KINETIC_PARAMETERS try: with open(path, 'r') as f: data = json.load(f) # Basic validation for k, v in data.items(): if 'C0' not in v or 'Ea' not in v or 'A' not in v: raise ValueError(f"Kinetic entry {k} missing required keys") return data except Exception: st.warning("Failed to load kinetics.json; falling back to embedded defaults.") return DEFAULT_KINETIC_PARAMETERS

def safe_load_pickle(path: str): if not path.endswith('.pkl') or not os.path.exists(path): return None try: with open(path, 'rb') as f: obj = pickle.load(f) return obj except Exception: st.warning("Could not load pickle safely; ignoring .pkl file.") return None

-------------------- Streamlit App --------------------

st.set_page_config(page_title="Vitamin C Predictor", layout='wide')

Header

st.markdown("""

<div style='text-align:center'>
  <h2 style='color:green; margin:0;'>BnZackx — Vitamin C Degradation Predictor</h2>
  <div style='font-size:0.95em; color:gray;'>Department of Food Science and Technology, ADUSTECH</div>
</div>
""", unsafe_allow_html=True)
st.write("---")Load kinetics (option to upload)

col_top = st.columns([2, 1]) with col_top[0]: st.header("Model configuration") st.write("You can either use the embedded kinetic parameters, upload a kinetics JSON/CSV file, or load a pickle model.")

with col_top[1]: uploaded_file = st.file_uploader("Upload kinetics.json (optional)", type=['json', 'csv']) use_pickle = st.checkbox("Attempt to load predictor from pickle (vitamin_c_predictor.pkl)", value=False)

kinetics = DEFAULT_KINETIC_PARAMETERS

If user uploaded file

if uploaded_file is not None: try: if uploaded_file.type == 'application/json' or uploaded_file.name.endswith('.json'): kinetics = json.load(uploaded_file) else: df = pd.read_csv(uploaded_file) # Expect columns: crop, C0, Ea_kJ_per_mol, A_s^-1 (or similar) kinetics = {} for _, r in df.iterrows(): crop = r.iloc[0] C0 = float(r[1]) Ea_raw = float(r[2]) A_raw = float(r[3]) # convert Ea to J/mol if input in kJ/mol if Ea_raw < 10000:  # heuristic: assume kJ/mol Ea = Ea_raw * 1000 else: Ea = Ea_raw kinetics[crop] = {'C0': C0, 'Ea': Ea, 'A': A_raw} st.success("Successfully loaded kinetics from uploaded file.") except Exception as e: st.error(f"Failed to parse uploaded kinetics file: {e}") kinetics = DEFAULT_KINETIC_PARAMETERS else: # try load kinetics.json from disk kinetics = load_kinetics_from_json(DEFAULT_KINETIC_FILE_JSON)

Try load from pickle if requested

predictor_from_pickle = None if use_pickle: predictor_from_pickle = safe_load_pickle(MODEL_FILENAME)

if predictor_from_pickle is not None: predictor = predictor_from_pickle st.info("Predictor loaded from pickle.") else: predictor = VitaminCPredictor(kinetics)

Main interactive panel

st.header("Prediction panel") left, right = st.columns([1, 2])

with left: crop_type = st.selectbox("Select crop:", options=list(kinetics.keys()))

# Presets
st.write("**Presets**")
preset = st.selectbox("Choose a preset process:", options=['Custom', 'Blanching (90°C, 2 min)', 'Boiling (100°C, 5 min)', 'Pasteurization (72°C, 15 s)'])
if preset == 'Blanching (90°C, 2 min)':
    default_temp = 90.0
    default_time = 2.0
elif preset == 'Boiling (100°C, 5 min)':
    default_temp = 100.0
    default_time = 5.0
elif preset == 'Pasteurization (72°C, 15 s)':
    default_temp = 72.0
    default_time = 0.25  # 15 seconds -> 0.25 min
else:
    default_temp = 85.0
    default_time = 15.0

temperature = st.slider("Processing Temperature (°C)", min_value=0.0, max_value=200.0, value=float(default_temp), step=0.5)
time_duration = st.number_input("Processing Time (minutes)", min_value=0.0, max_value=10000.0, value=float(default_time), step=0.1)

# Optional override for C0
st.write("Optional: supply your own initial Vitamin C (C₀) if you measured it in lab")
C0_override = st.number_input("Custom C₀ (mg/100g) — leave 0 to use default", min_value=0.0, value=0.0, step=0.1)
if C0_override == 0.0:
    C0_override = None

uncertainty_pct = st.slider("Assumed uncertainty for Ct (± %)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
uncertainty_frac = float(uncertainty_pct) / 100.0

st.write("---")
if st.button("Calculate and plot"):
    try:
        result = predictor.predict(crop_type, temperature, time_duration, C0_override=C0_override,
                                   uncertainty_frac=uncertainty_frac)

        # Retention
        C0_used = result['C0']
        Ct = result['Ct']
        retention_pct = (Ct / C0_used * 100.0) if C0_used > 0 else 0.0

        # Save to session history
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        st.session_state['history'].append({
            'crop': crop_type,
            'temp_C': temperature,
            'time_min': time_duration,
            'C0': C0_used,
            'Ct': Ct,
            'retention_pct': retention_pct
        })

        # Display textual results
        st.success(f"Final Vitamin C (Cₜ): {Ct:.2f} mg/100g")
        st.info(f"Initial (C₀): {C0_used:.2f} mg/100g")
        st.warning(f"Retention: {retention_pct:.2f}%")
        st.write(f"Apparent rate constant k = {result['k_min']:.4e} min⁻¹")
        st.write(f"Half-life (t₁/₂) = {result['half_life_min']:.2f} minutes")

        # Simulate curve
        times = np.linspace(0, max(time_duration, 1.0) * 1.2, 200)
        df_curve = predictor.simulate_curve(crop_type, temperature, times, C0_override=C0_override,
                                           uncertainty_frac=uncertainty_frac)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df_curve['time_min'], df_curve['Ct'], label='Ct (predicted)')
        if uncertainty_frac > 0:
            ax.fill_between(df_curve['time_min'], df_curve['Ct_low'], df_curve['Ct_high'], alpha=0.25,
                            label=f'±{uncertainty_pct:.1f}% uncertainty')
        ax.axvline(time_duration, color='gray', linestyle='--', label='Selected time')
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Vitamin C (mg/100g)')
        ax.set_title(f"Vitamin C degradation — {crop_type} at {temperature:.1f}°C")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Prediction error: {e}")

with right: st.subheader("Kinetic parameters (preview)") kp_df = pd.DataFrame.from_dict(kinetics, orient='index') kp_df_display = kp_df.copy() # Convert Ea to kJ/mol for display if large kp_df_display['Ea_kJ_per_mol'] = kp_df_display['Ea'].apply(lambda x: float(x) / 1000.0) kp_df_display = kp_df_display[['C0', 'Ea_kJ_per_mol', 'A']] kp_df_display.columns = ['C0 (mg/100g)', 'Ea (kJ/mol)', 'A (s^-1)'] st.dataframe(kp_df_display)

st.write("\n\nNotes:\n- A is assumed in s⁻¹. If your A is in min⁻¹ set it in the uploaded file accordingly.\n- Ea should be in J/mol (or kJ/mol and converted automatically for CSV uploads).")

st.write("---")
st.subheader("Prediction history (this session)")
if 'history' in st.session_state and st.session_state['history']:
    hist_df = pd.DataFrame(st.session_state['history'])
    st.dataframe(hist_df.style.format({
        'temp_C': '{:.1f}', 'time_min': '{:.2f}', 'C0': '{:.2f}', 'Ct': '{:.2f}', 'retention_pct': '{:.2f}'
    }))
    if st.button("Export history to CSV"):
        csv = hist_df.to_csv(index=False)
        st.download_button("Download CSV", data=csv, file_name="prediction_history.csv", mime='text/csv')
else:
    st.write("No predictions made yet in this session.")

Footer

st.write("---") st.markdown("<div style='text-align:center;color:gray;'>© Umar Faruk Zakariyyya | BnZackx® 2025</div>", unsafe_allow_html=True)

End of app
