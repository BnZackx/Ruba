import streamlit as st
import pickle
import numpy as np
import os

# =================================================================
# 🛑 CRITICAL CODE: CLASS AND CONSTANTS
# =================================================================

# Gas constant in J/(mol*K)
R_GAS = 8.314

# Kinetic parameters extracted from the document
KINETIC_PARAMETERS = {
    'Orange (Citrus sinensis)': {
        'C0': 52.3,
        'Ea': 68.4 * 1000, # Convert kJ/mol to J/mol
        'A': 2.34e10
    },
    'Baobab (Adansonia digitata)': {
        'C0': 225.8,
        'Ea': 72.9 * 1000,
        'A': 1.07e11
    },
    'Fluted pumpkin (Telfairia occidentalis)': {
        'C0': 85.2,
        'Ea': 66.8 * 1000,
        'A': 8.45e9
    },
    'Spinach (Amaranthus hybridus)': {
        'C0': 62.4,
        'Ea': 75.2 * 1000,
        'A': 3.89e11
    }
}

class VitaminCPredictor:
    """
    Predicts Vitamin C content (Ct) after thermal processing 
    based on first-order kinetics and the Arrhenius equation.
    """

    def __init__(self, parameters):
        """Initializes the predictor with kinetic parameters."""
        self.parameters = parameters

    def get_rate_constant(self, crop_type, temp_celsius):
        """
        Calculates the degradation rate constant (k) and converts it to min⁻¹.
        """
        if crop_type not in self.parameters:
            return 0 

        params = self.parameters[crop_type]
        Ea = params['Ea']
        A = params['A']
        
        # Convert Celsius to Kelvin
        T_K = temp_celsius + 273.15
        
        # Arrhenius Equation: k_s is the rate constant in s⁻¹
        k_s = A * np.exp(-Ea / (R_GAS * T_K))
        
        # 🟢 CORRECTION: Convert k from s⁻¹ to min⁻¹ (multiply by 60) 
        # because the time input (t) is in minutes.
        k_min = k_s * 60
        return k_min

    def predict(self, crop_type, temp_celsius, time_min):
        """
        Predicts the final Vitamin C content (Ct) in mg/100g.
        """
        if crop_type not in self.parameters:
            return None 

        params = self.parameters[crop_type]
        C0 = params['C0']
        
        # Calculate the rate constant 'k' (now correctly in min⁻¹)
        k = self.get_rate_constant(crop_type, temp_celsius)
        
        # First-order kinetic model: Ct = C0 * exp(-k * t)
        # k (min⁻¹) and t (min) have consistent units.
        Ct = C0 * np.exp(-k * time_min)
        
        return max(0.0, Ct)

# =================================================================
# 🛑 END OF CRITICAL CODE
# =================================================================

# --- STREAMLIT CONFIGURATION AND MODEL LOADING ---

MODEL_FILENAME = 'vitamin_c_predictor.pkl'
CROP_OPTIONS = list(KINETIC_PARAMETERS.keys())

st.set_page_config(page_title="Vitamin C Predictor", layout="centered")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILENAME):
        st.warning("Could not find the .pkl file. Using embedded KINETIC_PARAMETERS for prediction.")
        return VitaminCPredictor(KINETIC_PARAMETERS)
        
    try:
        with open(MODEL_FILENAME, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"FATAL ERROR during model loading, check pickle compatibility: {e}")
        return None

predictor = load_model()

# =================================================================
# 🎓 HEADER IMPLEMENTATION (NEW GREEN HEADER)
# =================================================================

st.markdown("""
<div style="text-align: center; padding: 10px;">
    <h3 style="margin: 0; color: green;">BnZackx</h3>
    <p style="margin: 0; font-size: 1.1em; color: green;">Department of Food Science and Technology, ADUSTECH
