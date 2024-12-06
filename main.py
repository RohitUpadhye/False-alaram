import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

# Train the model
def train_model():
    try:
        # Load dataset
        data = pd.read_excel(r"C:\Users\HP\Downloads\Historical Alarm Cases.xlsx")
        x = data.iloc[:, 1:7]
        y = data['Spuriosity Index(0/1)']

        # Train the logistic regression model
        logm = LogisticRegression()
        logm.fit(x, y)

        # Save the trained model
        joblib.dump(logm, 'train.pkl')
        st.success("Model trained successfully and saved as 'train.pkl'.")
    except FileNotFoundError:
        st.error("'Historical Alarm Cases.xlsx' not found.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Validate input fields
def validate_input(value, field_name):
    if value.strip() == "":
        raise ValueError(f"{field_name} cannot be empty.")
    try:
        return float(value) if '.' in value else int(value)
    except ValueError:
        raise ValueError(f"{field_name} must be a valid number.")

# Predict based on user input
def test_model(input_data):
    try:
        # Load the trained model
        pkl_file = joblib.load('train.pkl')
    except FileNotFoundError:
        st.error("'train.pkl' not found. Train the model first.")
        return

    try:
        # Predict the result
        test_array = np.array(input_data).reshape(1, -1)
        df_test = pd.DataFrame(
            test_array,
            columns=[
                'Ambient Temperature( deg C)',
                'Calibration(days)',
                'Unwanted substance deposition(0/1)',
                'Humidity(%)',
                'H2S Content(ppm)',
                'detected by(% of sensors)',
            ]
        )
        y_pred = pkl_file.predict(df_test)

        result = "False Alarm, No Danger" if y_pred == 1 else "True Alarm, Danger"
        st.success(f"Prediction Result: {result}")
    except ValueError as ve:
        st.error(str(ve))
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Streamlit App
st.title("Alarm Prediction System")

# Sidebar for model training
st.sidebar.header("Model Training")
if st.sidebar.button("Train Model"):
    train_model()

# Input fields for prediction
st.header("Enter Input Data for Prediction")

ambient_temp = st.text_input("Ambient Temperature (°C):")
calibration = st.text_input("Calibration (days):")
unwanted_deposition = st.text_input("Unwanted Substance Deposition (0 or 1):")
humidity = st.text_input("Humidity (%):")
h2s_content = st.text_input("H2S Content (ppm):")
detected_by = st.text_input("Detected by (% of sensors):")

# Button for prediction
if st.button("Predict"):
    try:
        # Validate and collect input data
        input_data = [
            validate_input(ambient_temp, "Ambient Temperature (°C)"),
            validate_input(calibration, "Calibration (days)"),
            validate_input(unwanted_deposition, "Unwanted Substance Deposition (0 or 1)"),
            validate_input(humidity, "Humidity (%)"),
            validate_input(h2s_content, "H2S Content (ppm)"),
            validate_input(detected_by, "Detected by (% of sensors)")
        ]
        test_model(input_data)
    except ValueError as ve:
        st.error(str(ve))
