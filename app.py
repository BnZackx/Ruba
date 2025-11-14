import streamlit as st
import pickle
import pandas as pd
import os

# --- Configuration: Adjust these to match your training data! ---
MODEL_FILENAME = 'vitamin_c_predictor.pkl'
FEATURE_COLUMNS = ['Temperature', 'Time_Duration', 'Initial_Vitamin_C', 'Initial_Moisture_Content', 'pH_Level']
TARGET_NAME = 'Retention Percentage'

# --- Model Loading (Caches the model to load only once) ---
@st.cache_resource
def load_model():
    # Check if the model file is available in the deployment environment
    if not os.path.exists(MODEL_FILENAME):
        st.error(f"Model file '{MODEL_FILENAME}' not found. Check GitHub commit.")
        return None
    try:
        with open(MODEL_FILENAME, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

predictor = load_model()

# --- Application Title and Layout ---
st.set_page_config(page_title="Vitamin C Predictor", layout="centered")
st.title('🌡️ Vitamin C Degradation Predictor')
st.markdown('### Estimate Vitamin C retention based on processing and initial properties.')

if predictor:
    # --- User Input Sliders ---
    st.header('1. Set Processing Parameters')
    
    # 1. Temperature Slider (Assuming typical food processing ranges)
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
