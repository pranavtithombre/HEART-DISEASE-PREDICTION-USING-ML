import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction System")

st.write("Enter patient details below:")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=52)

sex = st.selectbox(
    "Sex",
    options=[1, 0],
    format_func=lambda x: "Male" if x == 1 else "Female"
)

chest_pain = st.selectbox(
    "Chest Pain Type",
    options=[0, 1, 2, 3],
    format_func=lambda x: {
        0: "Atypical",
        1: "Typical",
        2: "Non-anginal",
        3: "Asymptomatic"
    }[x]
)

bp = st.number_input(
    "Resting Blood Pressure",
    min_value=50,
    max_value=250,
    value=125
)

cholesterol = st.number_input(
    "Cholesterol",
    min_value=50,
    max_value=700,
    value=212
)

fbs = st.selectbox(
    "Fasting Blood Sugar",
    options=[0, 1],
    format_func=lambda x:
    "< 120 mg/ml" if x == 0 else "> 120 mg/ml"
)

rest_ecg = st.selectbox(
    "Resting ECG",
    options=[0, 1, 2],
    format_func=lambda x: {
        0: "Left ventricular hypertrophy",
        1: "Normal",
        2: "ST-T wave abnormality"
    }[x]
)

max_hr = st.number_input(
    "Maximum Heart Rate Achieved",
    min_value=50,
    max_value=250,
    value=122
)

# Prediction button
if st.button("Predict"):

    input_data = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "Chest_Pain_Type": [chest_pain],
        "BP": [bp],
        "Cholesterol": [cholesterol],
        "Fasting_Blood_Sugar": [fbs],
        "Resting_ECG": [rest_ecg],
        "Max_heart_rate": [max_hr]
    })

    prediction = model.predict(input_data)

    if prediction[0] == 0:
        st.success("✅ No Heart Disease Detected")
    else:
        st.error("⚠️ Heart Disease Detected")

    # Probability (if supported)
    try:
        probability = model.predict_proba(input_data)

        st.subheader("Prediction Confidence")

        st.write(
            f"No Heart Disease: {probability[0][0]*100:.2f}%"
        )
        st.write(
            f"Heart Disease: {probability[0][1]*100:.2f}%"
        )
    except:
        pass