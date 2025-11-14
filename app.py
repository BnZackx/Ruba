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
# 🎓 HEADER IMPLEMENTATION (Background removed, University name is red)
# =================================================================

st.markdown("""
<div style="text-align: center; padding: 10px;">
    <h3 style="margin: 0; color: red;">Aliko Ɗangote University of Science and Technology, Wudil</h3>
    <p style="margin: 0; font-size: 1.1em; color: #555;">Faculty of Agriculture and Agricultural Technology</p>
    <p style="margin: 0; font-size: 1.1em; color: #555;">Department of Food Science and Technology</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---") 

st.title('🌡️ Vitamin C Degradation Predictor (Kinetic Model)')
st.markdown('### Estimate Vitamin C content remaining after thermal processing.')

if predictor:
    # --- User Input Widgets ---
    st.header('1. Select Food and Processing Parameters')
    
    # Crop Selection
    crop_type = st.selectbox(
        'Select Crop Type:',
        options=CROP_OPTIONS,
        key='crop_selector'
    )
    
    # Temperature Slider
    temperature = st.slider(
        'Processing Temperature (°C)',
        min_value=50.0, 
        max_value=120.0, 
        value=85.0, 
        step=0.5,
        key='temperature_input_key'
    )
    
    # Time Slider
    time_duration = st.slider(
        'Processing Time (minutes)',
        min_value=1.0, 
        max_value=120.0, 
        value=15.0, 
        step=1.0,
        key='time_duration_key'
    )

    # --- Prediction Logic ---
    st.markdown("---")
    if st.button('Calculate Remaining Vitamin C', type="primary", key='predict_button_key'): 
        
        # Run prediction
        predicted_ct = predictor.predict(crop_type, temperature, time_duration)
        
        if predicted_ct is not None:
            
            # Get initial concentration for retention calculation
            C0 = KINETIC_PARAMETERS[crop_type]['C0']
            
            # Calculate retention and loss
            retention_percent = (predicted_ct / C0) * 100
            
            # --- Display Results ---
            st.header('Prediction Results')
            st.success(f"Final Vitamin C Content (Cₜ): **{predicted_ct:.2f} mg/100g**")
            st.info(f"Initial Content (C₀): **{C0:.2f} mg/100g**")
            st.warning(f"Retention Percentage: **{retention_percent:.2f}%**")

            # =================================================================
            # 💡 ADAPTIVE FEEDBACK IMPLEMENTATION
            # =================================================================
            st.markdown("---")
            st.subheader("Process Recommendation")
            
            if retention_percent >= 80:
                st.balloons()
                st.success(f"**Excellent Retention!** You are predicted to retain over {retention_percent:.0f}% of Vitamin C. Your current combination of **{temperature}°C** and **{time_duration} min** is highly effective for preservation.")
                
            elif retention_percent >= 50:
                st.info(f"**Moderate Retention.** The retention is acceptable, but only {retention_percent:.0f}% remains. To **increase Vitamin C retention**, consider reducing the **Processing Time** or the **Temperature** by a small margin.")
                
            elif retention_percent >= 25:
                st.warning(f"**Significant Degradation.** Only {retention_percent:.0f}% remains. Vitamin C loss is high at this setting. You should **immediately reduce the temperature** to below **{temperature - 10}°C** or cut the processing time significantly to preserve quality.")

            else: # retention_percent < 25
                st.error(f"**Severe Degradation!** Less than {retention_percent:.0f}% remains. Vitamin C is almost completely degraded. This thermal process is too aggressive and should be **optimized for lower temperature and much shorter duration**.")
            # =================================================================

        else:
            st.error("Prediction failed. Please ensure the selected crop type is valid.")

# =================================================================
# 🦶 FOOTER IMPLEMENTATION (Fixed position)
# =================================================================

st.markdown("""
<style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        color: #888;
        text-align: center;
        padding: 5px;
        font-size: 0.8em;
        border-top: 1px solid #e6e6e6;
    }
</style>
<div class="footer">
    <p>© Umar Faruk Zakariyya | BnZackx®, MMXXV</p>
</div>
""", unsafe_allow_html=True)
