import streamlit as st
from streamlit.components.v1 import html
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import time

# Load the models and scalers
heart_disease_model = joblib.load('logistic_regression_model.pkl')
stroke_model = joblib.load('stroke_prediction_model.pkl')
stroke_scaler = joblib.load('scaler.pkl')
stroke_model_columns = joblib.load('model_columns.pkl')

# Dummy data generators
def generate_ecg_waveform():
    t = np.linspace(0, 1, 500)
    ecg = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(500)
    return t, ecg

def generate_respiration_waveform():
    t = np.linspace(0, 10, 500)
    respiration = np.sin(2 * np.pi * 0.2 * t) + 0.1 * np.random.randn(500)
    return t, respiration

def generate_spo2_waveform():
    t = np.linspace(0, 10, 500)
    spo2 = 0.9 * np.sin(2 * np.pi * 1 * t) + 1
    return t, spo2

def generate_bp_waveform():
    t = np.linspace(0, 10, 500)
    bp = 120 + 10 * np.sin(2 * np.pi * 0.5 * t) + 5 * np.random.randn(500)
    return t, bp

def generate_icp_waveform():
    t = np.linspace(0, 10, 500)
    icp = 10 + 5 * np.sin(2 * np.pi * 0.1 * t) + 2 * np.random.randn(500)
    return t, icp

def generate_cbf_waveform():
    t = np.linspace(0, 10, 500)
    cbf = 50 + 10 * np.sin(2 * np.pi * 0.2 * t) + 3 * np.random.randn(500)
    return t, cbf

# Function to gather inputs for models
def get_combined_input():
    age = st.number_input("Enter age", min_value=0, step=1)
    sex = st.selectbox("Select sex", ["Select","Female", "Male"])
    sex = 1 if sex == "Male" else 0
    cp = st.selectbox("Select chest pain type", [0, 1, 2, 3])
    trestbps = st.number_input("Enter resting blood pressure", min_value=0, step=1)
    chol = st.number_input("Enter serum cholesterol in mg/dl", min_value=0, step=1)
    fbs = st.selectbox("Select fasting blood sugar", [0, 1])
    restecg = st.selectbox("Select resting electrocardiographic results", [0, 1, 2])
    thalach = st.number_input("Enter maximum heart rate achieved", min_value=0, step=1)
    exang = st.selectbox("Select exercise induced angina", [0, 1])
    oldpeak = st.number_input("Enter ST depression induced by exercise relative to rest", min_value=0.0, step=0.1)
    slope = st.selectbox("Select the slope of the peak exercise ST segment", [0, 1, 2])
    ca = st.selectbox("Enter number of major vessels colored by fluoroscopy", [0, 1, 2, 3])
    thal = st.selectbox("Enter thalassemia", ["",1, 2, 3])
    hypertension = st.selectbox("Enter hypertension", [0, 1])
    heart_disease = st.selectbox("Enter heart disease", [0, 1])
    ever_married = st.selectbox("Enter marital status", ["Select","No", "Yes"])
    work_type = st.selectbox("Enter work type", ["Select","Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
    residence_type = st.selectbox("Enter residence type", ["Select","Urban", "Rural"])
    avg_glucose_level = st.number_input("Enter average glucose level", min_value=0.0, step=1.0)
    bmi = st.number_input("Enter BMI", min_value=0.0, step=0.1)
    smoking_status = st.selectbox("Enter smoking status", ["Select","Formerly smoked", "Never smoked", "Smokes", "Unknown"])

    heart_disease_input = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    stroke_input = {
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'gender_Male': 1 if sex == 1 else 0,
        'ever_married_Yes': 1 if ever_married == 'Yes' else 0,
        'work_type_Private': 1 if work_type == 'Private' else 0,
        'work_type_Self-employed': 1 if work_type == 'Self-employed' else 0,
        'work_type_children': 1 if work_type == 'Children' else 0,
        'work_type_Govt_job': 1 if work_type == 'Govt_job' else 0,
        'Residence_type_Urban': 1 if residence_type == 'Urban' else 0,
        'smoking_status_formerly smoked': 1 if smoking_status == 'Formerly smoked' else 0,
        'smoking_status_never smoked': 1 if smoking_status == 'Never smoked' else 0,
        'smoking_status_smokes': 1 if smoking_status == 'Smokes' else 0,
    }

    return np.asarray(heart_disease_input).reshape(1, -1), pd.DataFrame([stroke_input]).reindex(columns=stroke_model_columns, fill_value=0)

# Set up Streamlit configuration
st.set_page_config(page_title="Real-Time Monitoring Dashboard", layout="wide")
st.title('')

# Create tabs with chatbot included
with st.sidebar:
    selected = option_menu(
        "Monitoring Tabs",
        ["ECG", "Respiratory Rate", "Oxygen Saturation", "Blood Pressure", "Intracranial Pressure", "Cerebral Blood Flow", "Health Prediction", "Doctor Consultation", "Chatbot"],
        icons=["activity", "lungs", "droplet", "heart", "app", "eye", "play", "telephone", "chat"],
        menu_icon="cast",
        default_index=0,
    )

# Embed the chatbot using an iframe within Streamlit
def embed_chatbot(url, height=600):
    iframe = f'<iframe src="{url}" width="100%" height="{height}px" frameborder="0" allowfullscreen></iframe>'
    html(iframe, height=height)

# Plotting function using Plotly
def plot_waveform(title, t, data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=data, mode='lines', name=title))
    fig.update_layout(title=title, xaxis_title='Time', yaxis_title='Value')
    return fig

# Control flow for app functionality based on selected tab
if selected == "Chatbot":
    st.header("HealthCare Chatbot")
    chatbot_url = "http://localhost:3000/"
    embed_chatbot(chatbot_url)

elif selected == "Doctor Consultation":
    st.header("Doctor Consultation")
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION)

elif selected == "Health Prediction":
    st.header("Health Prediction Dashboard")
    st.subheader("Enter patient details to predict heart disease and stroke")
    heart_disease_input, stroke_input = get_combined_input()
    if st.button("Predict"):
        with st.spinner('Predicting...'):
            heart_disease_prediction = heart_disease_model.predict(heart_disease_input)
            stroke_input_scaled = stroke_scaler.transform(stroke_input)
            stroke_prediction = stroke_model.predict(stroke_input_scaled)
            if heart_disease_prediction[0] == 0:
                st.success('The Patient does not have Heart Disease')
            else:
                st.error('The Patient may have Heart Disease')
            if stroke_prediction[0] == 1:
                st.error('The Patient may have Stroke')
            else:
                st.success('The Patient does not have Stroke')

else:
    # Real-time data updating for physiological parameters
    t_placeholder = st.empty()
    fig_placeholder = st.empty()
    start_time = time.time()
    while True:
        elapsed_time = time.time() - start_time
        if selected == "ECG":
            t, ecg = generate_ecg_waveform()
            fig = plot_waveform('ECG Waveform', t, ecg)
        elif selected == "Respiratory Rate":
            t, respiration = generate_respiration_waveform()
            fig = plot_waveform('Respiratory Rate Waveform', t, respiration)
        elif selected == "Oxygen Saturation":
            t, spo2 = generate_spo2_waveform()
            fig = plot_waveform('Oxygen Saturation Waveform', t, spo2)
        elif selected == "Blood Pressure":
            t, bp = generate_bp_waveform()
            fig = plot_waveform('Blood Pressure Waveform', t, bp)
        elif selected == "Intracranial Pressure":
            t, icp = generate_icp_waveform()
            fig = plot_waveform('Intracranial Pressure Waveform', t, icp)
        elif selected == "Cerebral Blood Flow":
            t, cbf = generate_cbf_waveform()
            fig = plot_waveform('Cerebral Blood Flow Waveform', t, cbf)
        t_placeholder.text(f"Elapsed Time: {elapsed_time:.2f} seconds")
        fig_placeholder.plotly_chart(fig)
        time.sleep(1)
