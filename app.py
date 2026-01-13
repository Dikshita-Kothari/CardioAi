import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="CardioAI | Advanced Risk Assessment",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# UTILITY FUNCTIONS
# --------------------------------------------------

def local_css(file_name):
    css_path = os.path.join(os.path.dirname(__file__), file_name)
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Safely load trained model for Streamlit Cloud"""
    try:
        model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_input(
    age, gender, height, weight,
    ap_hi, ap_lo, cholesterol, gluc,
    smoke, alco, active
):
    """Prepare input exactly as model expects"""

    age_days = int(age * 365.25)

    input_data = pd.DataFrame({
        "age": [age_days],
        "gender": [gender],
        "height": [height],
        "weight": [weight],
        "ap_hi": [ap_hi],
        "ap_lo": [ap_lo],
        "cholesterol": [cholesterol],
        "gluc": [gluc],
        "smoke": [smoke],
        "alco": [alco],
        "active": [active]
    })

    # ---- Feature Engineering (MUST MATCH TRAINING) ----
    input_data["bmi"] = input_data["weight"] / ((input_data["height"] / 100) ** 2)
    input_data["bp_diff"] = input_data["ap_hi"] - input_data["ap_lo"]
    input_data["age_ap_hi"] = input_data["age"] * input_data["ap_hi"]

    return input_data

# --------------------------------------------------
# LOAD STYLES
# --------------------------------------------------
local_css("style.css")

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div class="main-header">
    <div class="logo-container">
        <div class="logo-icon">🫀</div>
        <div class="logo-text">Cardio<span class="highlight">AI</span></div>
    </div>
    <div class="header-subtitle">
        Advanced Cardiovascular Risk Prediction System
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# MAIN APP
# --------------------------------------------------
with st.container():
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="medium")

    # ------------------- COLUMN 1 -------------------
    with col1:
        st.markdown("### 👤 Patient Profile")
        age = st.slider("Age (Years)", 20, 100, 50)

        gender_display = st.radio("Gender", ["Female", "Male"], horizontal=True)
        gender = 1 if gender_display == "Female" else 2

        h_col, w_col = st.columns(2)
        height = h_col.number_input("Height (cm)", 100, 250, 165)
        weight = w_col.number_input("Weight (kg)", 30, 200, 70)

    # ------------------- COLUMN 2 -------------------
    with col2:
        st.markdown("### 🏥 Vitals & Labs")

        bp1, bp2 = st.columns(2)
        ap_hi = bp1.number_input("Systolic BP", 80, 220, 120)
        ap_lo = bp2.number_input("Diastolic BP", 40, 140, 80)

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

    # ------------------- COLUMN 3 -------------------
    with col3:
        st.markdown("### 🏃 Lifestyle Factors")

        smoke = 1 if st.toggle("Smoker") else 0
        alco = 1 if st.toggle("Alcohol Intake") else 0
        active = 1 if st.toggle("Physically Active", value=True) else 0

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button(
            "ANALYZE RISK PROFILE",
            type="primary",
            use_container_width=True
        )

    # ------------------- RESULTS -------------------
    if predict_btn:
        with st.spinner("Analyzing cardiovascular risk..."):
            model = load_model()

            if model is not None:
                input_df = preprocess_input(
                    age, gender, height, weight,
                    ap_hi, ap_lo, cholesterol, gluc,
                    smoke, alco, active
                )

                try:
                    prediction = model.predict(input_df)[0]
                    probability = model.predict_proba(input_df)[0][1]

                    st.markdown("---")

                    if prediction == 1:
                        risk_class = "high-risk"
                        icon, label = "⚠️", "HIGH RISK DETECTED"
                        msg = "Model indicates elevated cardiovascular risk."
                    else:
                        risk_class = "low-risk"
                        icon, label = "✅", "LOW RISK PROFILE"
                        msg = "Model indicates low cardiovascular risk."

                    st.markdown(f"""
                    <div class="result-card {risk_class}">
                        <div class="risk-header">{icon} {label}</div>
                        <div class="prob-value">{probability:.1%}</div>
                        <div class="risk-message">{msg}</div>
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Prediction error: {e}")

            else:
                st.error("Model could not be loaded.")

    st.markdown('<div class="decoration-circle"></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
