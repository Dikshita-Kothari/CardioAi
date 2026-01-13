import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CardioAI | Advanced Risk Assessment",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- LOAD CSS ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# --- UTILITY FUNCTIONS ---
@st.cache_resource
def load_model():
    """Load the trained model from disk."""
    try:
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_input(age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active):
    """
    Preprocess input features to match model training requirements.
    Feature order must match the model's expected input.
    Expected features usually: age_days, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active
    """
    # Convert age to days (approximate) as the model likely trained on days
    age_days = int(age * 365.25)
    
    # Calculate BMI (Body Mass Index) - commonly used derived feature, 
    # BUT we must validte if the model expects BMI or raw features. 
    # Based on standard datasets (like Kaggle Cardio), BMI might be a separate feature or just raw.
    # We will stick to the 11 raw features provided in the datasets commonly:
    # id (drop), age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, cardio
    
    # Create a DataFrame with the exact column names expected by the model
    # Note: Column names might need to match exactly if the model is a pipeline or sensitive to names.
    # Standard columns: ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    
    input_data = pd.DataFrame({
        'age': [age_days],
        'gender': [gender], # 1: Female, 2: Male
        'height': [height],
        'weight': [weight],
        'ap_hi': [ap_hi],
        'ap_lo': [ap_lo],
        'cholesterol': [cholesterol],
        'gluc': [gluc],
        'smoke': [smoke],
        'alco': [alco],
        'active': [active]
    })
    
    # --- Feature Engineering ---
    # Calculate BMI
    input_data['bmi'] = input_data['weight'] / ((input_data['height'] / 100) ** 2)
    
    # Calculate Pulse Pressure (ap_hi - ap_lo)
    input_data['bp_diff'] = input_data['ap_hi'] - input_data['ap_lo']
    
    # Calculate Age * Systolic BP Interaction
    # Note: Using the 'age' column which is in days
    input_data['age_ap_hi'] = input_data['age'] * input_data['ap_hi']
    
    return input_data

# --- APP LOGIC ---

# Inject CSS
try:
    local_css("style.css")
except:
    st.warning("style.css not found. Creating default styles.")

# Header Section
st.markdown("""
<div class="main-header">
    <div class="logo-container">
        <div class="logo-icon">🫀</div>
        <div class="logo-text">Cardio<span class="highlight">AI</span></div>
    </div>
    <div class="header-subtitle">Advanced Cardiovascular Risk Prediction System</div>
</div>
""", unsafe_allow_html=True)

# Main Container
with st.container():
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3, gap="medium")
    
    with col1:
        st.markdown("### 👤 Patient Profile")
        st.markdown("Enter physiological data.")
        
        # Demographics
        age = st.slider("Age (Years)", 20, 100, 50)
        gender_display = st.radio("Gender", ["Female", "Male"], horizontal=True)
        gender = 1 if gender_display == "Female" else 2
        
        col_h, col_w = st.columns(2)
        with col_h:
            height = st.number_input("Height (cm)", 100, 250, 165)
        with col_w:
            weight = st.number_input("Weight (kg)", 30, 200, 70)

    with col2:
        st.markdown("### 🏥 Vitals & Labs")
        st.markdown("Clinical measurements.")
        
        col_bp1, col_bp2 = st.columns(2)
        with col_bp1:
            ap_hi = st.number_input("Systolic BP", 80, 220, 120, help="Upper number")
        with col_bp2:
            ap_lo = st.number_input("Diastolic BP", 40, 140, 80, help="Lower number")
            
        cholesterol = st.select_slider(
            "Cholesterol Level",
            options=[1, 2, 3],
            format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "High"}[x]
        )
        
        gluc = st.select_slider(
            "Glucose Level",
            options=[1, 2, 3],
            format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "High"}[x]
        )

    with col3:
        st.markdown("### 🏃 Lifestyle Factors")
        st.markdown("Habits & Activity.")
        
        smoke = st.toggle("Smoker", value=False)
        alco = st.toggle("Alcohol Intake", value=False)
        active = st.toggle("Physically Active", value=True)
            
        # Convert to int
        smoke = 1 if smoke else 0
        alco = 1 if alco else 0
        active = 1 if active else 0

        st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
        
        # Prediction Section
        st.markdown('<div class="prediction-section">', unsafe_allow_html=True)
        
        predict_btn = st.button("ANALYZE RISK PROFILE", type="primary", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Result Section (Full Width below columns)
    if predict_btn:
        with st.spinner("Processing bio-metric data..."):
            model = load_model()
            
            if model:
                # Preprocess
                input_df = preprocess_input(
                    age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active
                )
                
                try:
                    # Prediction
                    prediction = model.predict(input_df)[0]
                    probability = model.predict_proba(input_df)[0][1]
                    
                    st.markdown("---")
                    
                    if prediction == 1:
                        risk_class = "high-risk"
                        risk_label = "HIGH RISK DETECTED"
                        risk_icon = "⚠️"
                        risk_msg = "Model indicates a high probability of cardiovascular disease."
                    else:
                        risk_class = "low-risk"
                        risk_label = "LOW RISK PROFILE"
                        risk_icon = "✅"
                        risk_msg = "Model indicates a low probability of cardiovascular disease."
                        
                    # Result Card
                    st.markdown(f"""
                    <div class="result-card {risk_class}">
                        <div class="risk-header">
                            <span class="risk-icon">{risk_icon}</span>
                            {risk_label}
                        </div>
                        <div class="probability-container">
                            <div class="prob-label">Risk Probability</div>
                            <div class="prob-value">{probability:.1%}</div>
                            <div class="progress-bar-bg">
                                <div class="progress-bar-fill" style="width: {probability*100}%;"></div>
                            </div>
                        </div>
                        <div class="risk-message">{risk_msg}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Additional Insights
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                         if cholesterol > 1 or gluc > 1:
                            st.warning("ELEVATED LABS: Consider monitoring cholesterol/glucose levels.")
                    with col_res2:
                        if ap_hi > 140 or ap_lo > 90:
                            st.warning("HYPERTENSION: Blood pressure readings suggest pre-hypertension.")
                        
                except Exception as e:
                    st.error(f"Prediction logic error: {e}")
                    st.info("Ensure the model expects the inputs in the provided format.")
            else:
                st.error("Model unavailable.")
        
    # Visual Decoration
    st.markdown("""
    <div class="decoration-circle"></div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True) # End glass-container
