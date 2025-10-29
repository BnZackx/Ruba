import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Configure page
st.set_page_config(page_title="Vitamin C AI", page_icon="🍊")

# AI Model
class VitaminCPredictor:
    def __init__(self):
        self.weights = np.array([0.3, 0.4, 0.2, 0.1])
        self.bias = 0.1
    
    def predict(self, age, diet, stress, sleep):
        user_data = [age, diet, stress, sleep]
        linear_output = np.dot(user_data, self.weights) + self.bias
        return 1 / (1 + np.exp(-linear_output))

# Initialize AI
ai = VitaminCPredictor()

# App title
st.title("🍊 Vitamin C AI Advisor")

# User inputs
st.header("Your Health Profile")
age = st.slider("Age", 18, 80, 35)
diet = st.slider("Diet Quality (1-10)", 1, 10, 5)
stress = st.slider("Stress Level (1-10)", 1, 10, 5)
sleep = st.slider("Sleep Quality (1-10)", 1, 10, 6)

# Calculate results
if st.button("Analyze Vitamin C Needs"):
    score = ai.predict(age, diet, stress, sleep)
    
    # Display results
    st.header("Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Effectiveness Score", f"{score:.1%}")
    with col2:
        if score > 0.7: 
            st.metric("Priority", "HIGH", delta="Strong benefit")
        elif score > 0.4: 
            st.metric("Priority", "MODERATE", delta="May benefit")
        else: 
            st.metric("Priority", "LOW", delta="Lifestyle first")
    with col3:
        if score > 0.7: 
            st.metric("Dose", "1000mg")
        elif score > 0.4: 
            st.metric("Dose", "500mg")
        else: 
            st.metric("Dose", "250mg")
    
    if score > 0.7:
        st.success("🚀 **HIGH PRIORITY**: Strong benefits from Vitamin C supplementation. Consider 1000mg daily.")
    elif score > 0.4:
        st.info("✅ **MODERATE PRIORITY**: Vitamin C may provide noticeable benefits. A 500mg daily dose could be helpful.")
    else:
        st.warning("💡 **LOW PRIORITY**: Focus on improving diet and lifestyle first. Consider low-dose (250mg) or dietary sources.")
    
    st.subheader("🔍 Factor Impact Analysis")
    factors = ['Age', 'Diet', 'Stress', 'Sleep']
    contributions = np.array([age, diet, stress, sleep]) * ai.weights
    
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = ax.bar(factors, contributions, color=colors)
    ax.set_ylabel('Impact Score')
    
    for bar, value in zip(bars, contributions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.2f}', ha='center', va='bottom')
    
    st.pyplot(fig)

with st.expander("ℹ️ About This AI"):
    st.markdown("""
    This AI analyzes 4 key factors that influence Vitamin C effectiveness:
    - **Age**: Older adults often have higher requirements
    - **Diet Quality**: Poor diets lack natural Vitamin C sources  
    - **Stress Level**: Stress increases Vitamin C utilization
    - **Sleep Quality**: Poor sleep affects nutrient absorption
    """)
