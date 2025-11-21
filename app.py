"""
Improved Streamlit app for Vitamin C degradation prediction.

Features:
- Better validation and units handling
- Load kinetic parameters from JSON or CSV
- Optional user-supplied C0
- Plot of Ct vs time with uncertainty band
- Half-life calculation
- Presets for common thermal processes
- Safe pickle loading fallback
- Session history of predictions
- Clean UI layout with column design

Author: Umar Faruk Zakariyyya | BnZackx
Date: 2025
"""

import streamlit as st
import numpy as np
import pandas as pd
import json
import os
import math
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
import pickle

# ==============================================================
# CONSTANTS & DEFAULTS
# ==============================================================

R_GAS = 8.314   # J/(mol*K)
DEFAULT_KINETIC_FILE_JSON = "kinetics.json"
MODEL_FILENAME = "vitamin_c_predictor.pkl"

DEFAULT_KINETIC_PARAMETERS: Dict[str, Dict[str, Any]] = {
    'Orange (Citrus sinensis)': {'C0': 52.3, 'Ea': 68.4 * 1000, 'A': 2.34e10},
    'Baobab (Adansonia digitata)': {'C0': 225.8, 'Ea': 72.9 * 1000, 'A': 1.07e11},
    'Fluted pumpkin (Telfairia occidentalis)': {'C0': 85.2, 'Ea': 66.8 * 1000, 'A': 8.45e9},
    'Spinach (Amaranthus hybridus)': {'C0': 62.4, 'Ea': 75.2 * 1000, 'A': 3.89e11},
}

# ==============================================================
# MODEL CLASS
# ==============================================================

class VitaminCPredictor:
    """Predict Vitamin C using first-order kinetics and the Arrhenius equation."""

    def __init__(self, parameters: Dict[str, Dict[str, Any]]):
        self.parameters = parameters

    def _validate_crop(self, crop: str) -> bool:
        return crop in self.parameters

    def get_rate_constant(self, crop_type: str, temp_celsius: float, A_unit: str = "s^-1") -> float:
        """Return k in min^-1."""
        if not self._validate_crop(crop_type):
            raise KeyError(f"Unknown crop type: {crop_type}")

        params = self.parameters[crop_type]
        Ea = float(params["Ea"])
        A = float(params["A"])

        if temp_celsius <= -273.15:
            raise ValueError("Temperature must be above absolute zero.")

        T_K = temp_celsius + 273.15
        k_s = A * math.exp(-Ea / (R_GAS * T_K))

        if A_unit == "s^-1":
            return k_s * 60.0
        elif A_unit == "min^-1":
            return k_s
        else:
            raise ValueError("A_unit must be 's^-1' or 'min^-1'")

    def predict(self, crop_type: str, temp_celsius: float, time_min: float,
                C0_override: Optional[float] = None, uncertainty_frac: float = 0.0) -> Dict[str, float]:

        if not self._validate_crop(crop_type):
            raise KeyError(f"Unknown crop type: {crop_type}")

        params = self.parameters[crop_type]
        C0 = float(params["C0"]) if C0_override is None else float(C0_override)

        k = self.get_rate_constant(crop_type, temp_celsius)
        Ct = C0 * math.exp(-k * time_min)

        if uncertainty_frac > 0:
            Ct_low = Ct * (1 - uncertainty_frac)
            Ct_high = Ct * (1 + uncertainty_frac)
        else:
            Ct_low = Ct_high = Ct

        half_life = math.log(2) / k if k > 0 else float("inf")

        return {
            "C0": C0,
            "Ct": max(0.0, Ct),
            "Ct_low": max(0.0, Ct_low),
            "Ct_high": max(0.0, Ct_high),
            "k_min": k,
            "half_life_min": half_life,
        }

    def simulate_curve(self, crop_type: str, temp_celsius: float, times_min: np.ndarray,
                       C0_override: Optional[float] = None, uncertainty_frac: float = 0.0):

        rows = []
        for t in times_min:
            d = self.predict(
                crop_type, temp_celsius, float(t),
                C0_override=C0_override, uncertainty_frac=uncertainty_frac
            )
            rows.append({
                "time_min": t,
                "Ct": d["Ct"],
                "Ct_low": d["Ct_low"],
                "Ct_high": d["Ct_high"],
            })

        return pd.DataFrame(rows)

# ==============================================================
# UTILITIES
# ==============================================================

@st.cache_resource
def load_kinetics_from_json(path: str):
    if not os.path.exists(path):
        return DEFAULT_KINETIC_PARAMETERS

    try:
        with open(path, "r") as f:
            data = json.load(f)

        for k, v in data.items():
            if not all(x in v for x in ("C0", "Ea", "A")):
                raise ValueError(f"Invalid kinetic entry: {k}")

        return data
    except Exception:
        st.warning("Error loading kinetics.json → using default parameters.")
        return DEFAULT_KINETIC_PARAMETERS


def safe_load_pickle(path: str):
    if not path.endswith(".pkl") or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        st.warning("Could not load pickle safely → ignoring.")
        return None

# ==============================================================
# STREAMLIT UI
# ==============================================================

st.set_page_config(page_title="Vitamin C Predictor", layout="wide")

st.markdown("""
<div style="text-align:center">
  <h2 style="color:green;margin:0;">BnZackx — Vitamin C Degradation Predictor</h2>
  <p style="color:gray;font-size:0.95em;">Department of Food Science and Technology, ADUSTECH</p>
</div>
""", unsafe_allow_html=True)

st.write("---")

# ==============================================================
# LOAD PARAMETERS
# ==============================================================

col_top = st.columns([2, 1])

with col_top[0]:
    st.header("Model Configuration")
    st.write("Use default parameters, upload JSON/CSV, or load a pickle model.")

with col_top[1]:
    uploaded = st.file_uploader("Upload kinetics.json / .csv", type=["json", "csv"])
    use_pickle = st.checkbox("Load predictor from pickle (.pkl)", value=False)

kinetics = DEFAULT_KINETIC_PARAMETERS

if uploaded:
    try:
        if uploaded.name.endswith(".json"):
            kinetics = json.load(uploaded)
        else:
            df = pd.read_csv(uploaded)
            kinetics = {}
            for _, r in df.iterrows():
                Ea_raw = float(r[2])
                Ea = Ea_raw * 1000 if Ea_raw < 10000 else Ea_raw
                kinetics[r.iloc[0]] = {"C0": float(r[1]), "Ea": Ea, "A": float(r[3])}

        st.success("Kinetic parameters loaded successfully!")
    except Exception as e:
        st.error(f"Error parsing uploaded file: {e}")
else:
    kinetics = load_kinetics_from_json(DEFAULT_KINETIC_FILE_JSON)

predictor_from_pickle = safe_load_pickle(MODEL_FILENAME) if use_pickle else None
predictor = predictor_from_pickle if predictor_from_pickle else VitaminCPredictor(kinetics)

# ==============================================================
# MAIN PANEL
# ==============================================================

st.header("Prediction Panel")

left, right = st.columns([1, 2])

with left:

    crop_type = st.selectbox("Select crop:", list(kinetics.keys()))

    st.write("### Presets")
    preset = st.selectbox(
        "Choose a preset:",
        ["Custom", "Blanching (90°C, 2 min)", "Boiling (100°C, 5 min)", "Pasteurization (72°C, 15 s)"]
    )

    if preset == "Blanching (90°C, 2 min)":
        default_temp, default_time = 90.0, 2.0
    elif preset == "Boiling (100°C, 5 min)":
        default_temp, default_time = 100.0, 5.0
    elif preset == "Pasteurization (72°C, 15 s)":
        default_temp, default_time = 72.0, 0.25
    else:
        default_temp, default_time = 85.0, 15.0

    temperature = st.slider("Processing Temperature (°C)", 0.0, 200.0, default_temp, 0.5)
    time_min = st.number_input("Processing Time (min)", 0.0, 10000.0, default_time, 0.1)

    st.write("#### Optional: custom initial Vitamin C (C₀)")
    C0_override = st.number_input("C₀ (mg/100g) → leave 0 to use default", 0.0, 10000.0, 0.0)
    if C0_override == 0.0:
        C0_override = None

    uncertainty_pct = st.slider("Uncertainty (±%)", 0.0, 50.0, 5.0, 0.5)
    uncertainty_frac = uncertainty_pct / 100.0

    st.write("---")

    if st.button("Calculate & Plot"):

        try:
            result = predictor.predict(
                crop_type, temperature, time_min,
                C0_override=C0_override, uncertainty_frac=uncertainty_frac
            )

            C0_used = result["C0"]
            Ct = result["Ct"]
            retention = (Ct / C0_used) * 100 if C0_used > 0 else 0

            st.success(f"Final Vitamin C (Cₜ): {Ct:.2f} mg/100g")
            st.info(f"Initial (C₀): {C0_used:.2f} mg/100g")
            st.warning(f"Retention: {retention:.2f}%")
            st.write(f"k = {result['k_min']:.4e} min⁻¹")
            st.write(f"Half-life (t₁/₂): {result['half_life_min']:.2f} min")

            times = np.linspace(0, max(time_min, 1) * 1.2, 200)
            df_curve = predictor.simulate_curve(
                crop_type, temperature, times,
                C0_override=C0_override, uncertainty_frac=uncertainty_frac
            )

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(df_curve["time_min"], df_curve["Ct"], label="Ct")
            if uncertainty_frac > 0:
                ax.fill_between(df_curve["time_min"], df_curve["Ct_low"], df_curve["Ct_high"],
                                alpha=0.25, label=f"±{uncertainty_pct}%")
            ax.axvline(time_min, color="gray", linestyle="--", label="Selected time")
            ax.set_xlabel("Time (min)")
            ax.set_ylabel("Vitamin C (mg/100g)")
            ax.set_title(f"Vitamin C degradation — {crop_type} at {temperature:.1f}°C")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)

            # save history
            if "history" not in st.session_state:
                st.session_state["history"] = []

            st.session_state["history"].append({
                "crop": crop_type,
                "temp_C": temperature,
                "time_min": time_min,
                "C0": C0_used,
                "Ct": Ct,
                "retention_pct": retention,
            })

        except Exception as e:
            st.error(f"Error: {e}")

with right:
    st.subheader("Kinetic Parameters")
    kp_df = pd.DataFrame.from_dict(kinetics, orient="index")
    kp_df["Ea_kJ/mol"] = kp_df["Ea"] / 1000
    st.dataframe(kp_df[["C0", "Ea_kJ/mol", "A"]])

    st.write("Notes:")
    st.write("- A is assumed in s⁻¹ unless your file specifies otherwise.")
    st.write("- Ea may be in kJ/mol or J/mol (auto-detected for CSV).")

    st.write("---")
    st.subheader("History (this session)")
    if "history" in st.session_state and st.session_state["history"]:
        hist = pd.DataFrame(st.session_state["history"])
        st.dataframe(hist)
        if st.button("Export history"):
            st.download_button(
                "Download CSV",
                hist.to_csv(index=False),
                "prediction_history.csv",
                "text/csv"
            )
    else:
        st.write("No predictions made yet.")

# ==============================================================
# FOOTER
# ==============================================================

st.write("---")
st.markdown("<div style='text-align:center;color:gray;'>© Umar Faruk Zakariyyya | BnZackx® 2025</div>", unsafe_allow_html=True)
