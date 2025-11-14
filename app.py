import streamlit as st
import pickle
import numpy as np
import os

# =================================================================
# 🛑 START: CRITICAL CODE FROM YOUR TRAINING SCRIPT (THE RECIPE CARD)
# This code MUST be present to successfully load the pickled object.
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
        """Calculates the degradation rate constant (k) in min⁻¹."""
        if crop_type not in self.parameters:
            # We don't raise an exception here; let the main predict handle it
            return 0 

        params = self.parameters[crop_type]
        Ea = params['Ea']
        A = params['A']
        
        # Convert Celsius to Kelvin
        T_K = temp_celsius + 273.15
        
        # Arrhenius Equation: k = A * exp(-Ea / (R * T_K))
        k = A * np.exp(-Ea / (R_GAS * T_K))
        return k

    def predict(self, crop_type, temp_celsius, time_min):
        """
        Predicts the final Vitamin C content (Ct) in mg/100g.
        """
        if crop_type not in self.parameters:
            return None # Return None if crop is not found

        params = self.parameters[crop_type]
        C0 = params['C0']
        
        # Calculate the rate constant 'k'
        k = self.get_rate_constant(crop_type, temp_celsius)
        
        # First-order kinetic model: Ct = C0 * exp(-k * t)
        Ct = C0 * np.exp(-k * time_min)
        
        return max(0.0, Ct) # Ensure content is not negative

# =================================================================
# 🛑 END OF CRITICAL CODE FROM YOUR TRAINING SCRIPT
# =================================================================

# --- STREAMLIT APPLICATION CODE ---

MODEL_FILENAME = 'vitamin_c_predictor.pkl'
CROP_OPTIONS = list(KINETIC_PARAMETERS.keys())

@st.cache_resource
def load_model():
    # Since the class definition is now available, loading should work!
    if not os.path.exists(MODEL_FILENAME):
        # Fallback: If the .pkl is missing, instantiate the class directly 
        # using the embedded KINETIC_PARAMETERS dictionary.
        st.warning("Could not find the .pkl file. Using embedded KINETIC_PARAMETERS for prediction.")
        return VitaminCPredictor(KINETIC_PARAMETERS)
        
    try:
        with open(MODEL_FILENAME, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        # This should catch any remaining pickle errors
        st.error(f"FATAL ERROR during model loading, check pickle compatibility: {e}")
        return None

predictor = load_model()

# --- Application Title and Layout ---
st.set_page_config(page_title="Vitamin C Predictor", layout="centered")
st.title('🌡️ Vitamin C Degradation Predictor (Kinetic Model)')
st.markdown('### Estimate Vitamin C content remaining after thermal processing.')

if predictor:
    # --- User Input Widgets ---
    st.header('1. Select Food and Processing Parameters')
    
    # Crop Selection
    crop_type = st.selectbox(
        'Select Crop Type:',
        options=CROP_OPTIONS,
        key='crop'
    )
    
    # Temperature Slider
    temperature = st.slider(
        'Processing Temperature (°C)',
        min_value=50.0, 
        max_value=120.0, 
        value=85.0, 
        step=0.5,
        key='temp'
    )
    
    # Time Slider
    time_duration = st.slider(
        'Processing Time (minutes)',
        min_value=1.0, 
        max_value=120.0, 
        value=15.0, 
        step=1.0,
        key='time'
    )

    # --- Prediction Logic ---
    st.markdown("---")
    if st.button('Calculate Remaining Vitamin C', type="primary"):
        
        # Run prediction using the loaded or instantiated predictor
        predicted_ct = predictor.predict(crop_type, temperature, time_duration)
        
        if predicted_ct is not None:
            
            # Get initial concentration for retention calculation
            C0 = KINETIC_PARAMETERS[crop_type]['C0']
            retention_percent = (predicted_ct / C0) * 100
            
            # Display Results
            st.header('Prediction Results')
            st.success(f"Final Vitamin C Content (Cₜ): **{predicted_ct:.2f} mg/100g**")
            st.info(f"Initial Content (C₀): **{C0:.2f} mg/100g**")
            st.warning(f"Retention Percentage: **{retention_percent:.2f}%**")
        else:
            st.error("Prediction failed. Please ensure the selected crop type is valid.")    # 1. Temperature Slider (Assuming typical food processing ranges)
    temperature = st.slider(
        'Processing Temperature (°C)',
        min_value=50, 
        max_value=120, 
        value=85, 
        step=1,
        key='temp'
    )
    
    # 2. Time Slider
    time_duration = st.slider(
        'Processing Time (minutes)',
        min_value=1, 
        max_value=60, 
        value=15, 
        step=1,
        key='time'
    )

    st.header('2. Set Initial Sample Properties')

    # 3. Initial Vitamin C Content
    initial_vc = st.number_input(
        'Initial Vitamin C Content (mg/100g)',
        min_value=50.0,
        max_value=500.0,
        value=150.0,
        step=5.0,
        key='vc_init'
    )

    # 4. Initial Moisture Content
    initial_moisture = st.slider(
        'Initial Moisture Content (%)',
        min_value=50.0,
        max_value=95.0,
        value=80.0,
        step=0.5,
        key='moisture_init'
    )
    
    # 5. pH Level
    ph_level = st.slider(
        'pH Level',
        min_value=2.0,
        max_value=8.0,
        value=4.0,
        step=0.1,
        key='ph_level'
    )
    
    # --- Prediction Button and Logic ---
    st.markdown("---")
    if st.button('Calculate Predicted Retention', type="primary"):
        
        # Collect all inputs in the exact order of FEATURE_COLUMNS
        input_values = [temperature, time_duration, initial_vc, initial_moisture, ph_level]

        # Create a DataFrame for prediction
        input_data = pd.DataFrame(
            [input_values],
            columns=FEATURE_COLUMNS 
        )
        
        # Run prediction
        try:
            predicted_retention = predictor.predict(input_data)[0]
            
            # Ensure the prediction is within a realistic range (0% to 100%)
            predicted_retention = max(0, min(100, predicted_retention))
            
            # Display Result
            st.header('Prediction Result')
            st.success(f'The predicted Vitamin C {TARGET_NAME} is **{predicted_retention:.2f}%**')
            st.balloons() 

        except Exception as e:
            st.error(f"An error occurred during prediction. Please check your model format. Error: {e}")
