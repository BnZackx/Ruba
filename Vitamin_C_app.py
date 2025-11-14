import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import joblib

# Configure page
st.set_page_config(
    page_title="Vitamin C Degradation AI", 
    page_icon="🔬",
    layout="wide"
)

# AI Model for Vitamin C Degradation Prediction
class VitaminCDegradationPredictor:
    def __init__(self):
        # Model coefficients based on our polynomial regression
        self.coefficients = {
            'intercept': 52.34,
            'temperature': -0.283,
            'time': -0.421,
            'temp_squared': -0.0018,
            'time_squared': 0.0032,
            'temp_time': -0.0021,
            'baobab': 125.67,
            'pumpkin': 34.28,
            'spinach': 9.45
        }
    
    def predict_vitamin_c(self, temperature, time, crop_type):
        """Predict Vitamin C content after thermal processing"""
        base = (self.coefficients['intercept'] + 
                self.coefficients['temperature'] * temperature +
                self.coefficients['time'] * time +
                self.coefficients['temp_squared'] * temperature**2 +
                self.coefficients['time_squared'] * time**2 +
                self.coefficients['temp_time'] * temperature * time)
        
        crop_effect = {
            'Orange': 0,
            'Baobab': self.coefficients['baobab'],
            'Fluted Pumpkin': self.coefficients['pumpkin'],
            'Spinach': self.coefficients['spinach']
        }
        
        final_vitamin_c = base + crop_effect[crop_type]
        return max(0, final_vitamin_c)  # Ensure non-negative
    
    def predict_retention(self, temperature, time, crop_type, initial_vitamin_c):
        """Calculate Vitamin C retention percentage"""
        final_vitamin_c = self.predict_vitamin_c(temperature, time, crop_type)
        retention = (final_vitamin_c / initial_vitamin_c) * 100
        return min(100, retention)  # Cap at 100%

# Initialize AI
ai = VitaminCDegradationPredictor()

# App title and description
st.title("🔬 Vitamin C Degradation AI Predictor")
st.markdown("""
**Predict Vitamin C retention during thermal processing of Nigerian crops**
*Powered by BnZackx LIMITED Research*
""")

# Sidebar for inputs
with st.sidebar:
    st.header("⚙️ Processing Parameters")
    
    crop_type = st.selectbox(
        "Select Crop",
        ["Orange", "Baobab", "Fluted Pumpkin", "Spinach"],
        help="Choose the crop for analysis"
    )
    
    temperature = st.slider(
        "Processing Temperature (°C)",
        min_value=40,
        max_value=100,
        value=80,
        step=5,
        help="Thermal processing temperature"
    )
    
    time = st.slider(
        "Processing Time (minutes)",
        min_value=0,
        max_value=120,
        value=30,
        step=5,
        help="Duration of thermal processing"
    )
    
    # Initial Vitamin C values based on crop
    initial_values = {
        'Orange': 52.3,
        'Baobab': 225.8,
        'Fluted Pumpkin': 85.2,
        'Spinach': 62.4
    }
    
    initial_vitamin_c = st.number_input(
        f"Initial Vitamin C (mg/100g)",
        min_value=0.0,
        max_value=300.0,
        value=initial_values[crop_type],
        step=0.1,
        help="Initial Vitamin C content before processing"
    )

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Prediction Results")
    
    if st.button("Predict Vitamin C Degradation", type="primary"):
        # Calculate predictions
        final_vitamin_c = ai.predict_vitamin_c(temperature, time, crop_type)
        retention = ai.predict_retention(temperature, time, crop_type, initial_vitamin_c)
        degradation = 100 - retention
        
        # Display results in metrics
        st.subheader("🔍 Results")
        
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            st.metric(
                "Final Vitamin C",
                f"{final_vitamin_c:.1f} mg/100g",
                delta=f"-{initial_vitamin_c - final_vitamin_c:.1f} mg/100g"
            )
        
        with result_col2:
            st.metric(
                "Retention",
                f"{retention:.1f}%",
                delta=f"-{degradation:.1f}%",
                delta_color="inverse"
            )
        
        with result_col3:
            if retention > 70:
                status = "Excellent"
                color = "green"
            elif retention > 50:
                status = "Good"
                color = "blue"
            elif retention > 30:
                status = "Moderate"
                color = "orange"
            else:
                status = "Poor"
                color = "red"
            
            st.metric("Quality", status)
        
        # Detailed analysis
        st.subheader("📈 Detailed Analysis")
        
        if retention > 70:
            st.success("""
            **✅ EXCELLENT RETENTION**: This processing condition maintains high Vitamin C levels. 
            Consider this for optimal nutrient preservation.
            """)
        elif retention > 50:
            st.info("""
            **🔶 GOOD RETENTION**: Acceptable Vitamin C retention. Suitable for most processing applications 
            while maintaining reasonable nutrient quality.
            """)
        elif retention > 30:
            st.warning("""
            **⚠️ MODERATE RETENTION**: Significant Vitamin C loss. Consider reducing temperature or time 
            if higher nutrient retention is required.
            """)
        else:
            st.error("""
            **❌ POOR RETENTION**: High Vitamin C degradation. This condition is not recommended for 
            nutrient-sensitive applications. Consider alternative processing methods.
            """)
        
        # Time series prediction
        st.subheader("🕒 Time Series Prediction")
        times = list(range(0, min(121, time + 30), 10))
        vitamin_c_values = [ai.predict_vitamin_c(temperature, t, crop_type) for t in times]
        retention_values = [(vc / initial_vitamin_c) * 100 for vc in vitamin_c_values]
        
        # Create time series chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Vitamin C content over time
        ax1.plot(times, vitamin_c_values, 'b-o', linewidth=2, markersize=4)
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Vitamin C (mg/100g)')
        ax1.set_title('Vitamin C Degradation Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Retention percentage over time
        ax2.plot(times, retention_values, 'r-o', linewidth=2, markersize=4)
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Retention (%)')
        ax2.set_title('Vitamin C Retention Over Time')
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig)

with col2:
    st.header("🌡️ Quick Insights")
    
    # Comparative analysis
    st.subheader("Crop Comparison")
    crops = ["Orange", "Baobab", "Fluted Pumpkin", "Spinach"]
    retentions = []
    
    for crop in crops:
        final_vc = ai.predict_vitamin_c(temperature, time, crop)
        initial_vc = initial_values[crop]
        retention = (final_vc / initial_vc) * 100
        retentions.append(retention)
    
    comparison_df = pd.DataFrame({
        'Crop': crops,
        'Retention (%)': retentions,
        'Stability': ['Low', 'High', 'Medium', 'Low']  # Based on activation energy
    })
    
    st.dataframe(
        comparison_df.style.format({'Retention (%)': '{:.1f}%'}).background_gradient(
            subset=['Retention (%)'], cmap='RdYlGn_r'
        ),
        use_container_width=True
    )
    
    # Optimal conditions guide
    st.subheader("💡 Processing Tips")
    
    if crop_type == "Baobab":
        st.info("""
        **Baobab Tip**: Excellent thermal stability. Can withstand higher temperatures (80-90°C) 
        while maintaining good Vitamin C retention.
        """)
    elif crop_type == "Orange":
        st.info("""
        **Orange Tip**: Moderate stability. Consider shorter processing times (< 30 minutes) 
        at lower temperatures (60-70°C).
        """)
    elif crop_type == "Fluted Pumpkin":
        st.info("""
        **Pumpkin Tip**: Add towards the end of cooking in traditional preparations 
        to minimize Vitamin C loss.
        """)
    else:  # Spinach
        st.info("""
        **Spinach Tip**: High sensitivity to heat. Use minimal cooking time and 
        consider steaming instead of boiling.
        """)
    
    # Kinetic parameters
    st.subheader("🔬 Kinetic Data")
    
    # Rate constants based on temperature (simplified)
    if temperature <= 60:
        k_value = 0.005
        half_life = 138.6
    elif temperature <= 80:
        k_value = 0.015
        half_life = 46.2
    else:
        k_value = 0.030
        half_life = 23.1
    
    st.metric("Degradation Rate Constant", f"{k_value:.3f} min⁻¹")
    st.metric("Half-life", f"{half_life:.1f} minutes")

# Footer
st.markdown("---")
st.markdown("""
**🔍 About This AI**: 
This predictive model is based on experimental data from BnZackx LIMITED research on Vitamin C 
degradation kinetics in Nigerian crops. The model uses polynomial regression trained on 
thermal processing data across multiple temperature-time conditions.

*© 2024 Umar Faruk Zakariyya | BnZackx LIMITED*
""")
