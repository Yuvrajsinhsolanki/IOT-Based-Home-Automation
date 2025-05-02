# Importing necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle

MODEL_FILE = "Random_Forest.pkl"

# Function to load the trained model
def load_model(model_file):
    try:
        with open(model_file, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error(f"Model file '{model_file}' not found. Please ensure the model file is in the correct location.")
        return None

# Function to make predictions
def predict_crop(model, inputs):
    if model:
        prediction = model.predict(np.array(inputs).reshape(1, -1))
        return prediction[0]
    else:
        st.error("Model is not loaded. Unable to make predictions.")
        return None
    
# Main function to run the Streamlit app
def main():
    # App Header
    st.title("CROP RECOMMENDATION SYSTEM")
    st.write("Provide the required details below to get a recommended crop based on the environmental factors.")

    # Form for user inputs
    with st.form("crop_input_form"):
        st.write("### Input Crop Details:")
        nitrogen = st.number_input("Nitrogen (N)", min_value=0.0, max_value=140.0, value=0.0, step=0.1)
        phosphorus = st.number_input("Phosphorus (P)", min_value=0.0, max_value=145.0, value=0.0, step=0.1)
        potassium = st.number_input("Potassium (K)", min_value=0.0, max_value=205.0, value=0.0, step=0.1)
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=51.0, value=0.0, step=0.1)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
        ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=0.0, step=0.1)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=0.0, step=0.1)

        # Submit button
        submitted = st.form_submit_button("Predict")
    
    # Load model or train if not available
    model = load_model(MODEL_FILE)

    # If the form is submitted, make predictions
    if submitted:
        inputs = [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]
        if all(x > 0 for x in inputs):  # Ensure all inputs are valid
            crop_prediction = predict_crop(model, inputs)
            st.success(f"Recommended Crop: {crop_prediction}")
        else:
            st.error("Please fill in all input fields with valid values before predicting.")

# Run the Streamlit app
if __name__ == "__main__":
    main()