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
def predict_fertilizer(model, inputs):
    if model:
        prediction = model.predict(np.array(inputs).reshape(1, -1))
        return prediction[0]
    else:
        st.error("Model is not loaded. Unable to make predictions.")
        return None

# Main function to run the Streamlit app
def main():
    # App Header
    st.title("FERTILIZER RECOMMENDATION SYSTEM")
    st.write("Provide the required details below to get a recommended fertilizer based on the environmental and soil factors.")

    # Form for user inputs
    with st.form("fertilizer_input_form"):
        st.write("### Input Fertilizer Details:")
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
        soil_type = st.selectbox("Soil Type", ["Black", "Clayey", "Loamy", "Red", "Sandy"])
        crop_type = st.selectbox("Crop Type", ["Barley", "Cotton", "Groundnuts", "Maize", "Millets", "Oil seeds", "Paddy", "Pulses", "Sugarcane", "Wheat"])
        nitrogen = st.number_input("Nitrogen (N)", min_value=0.0, max_value=140.0, value=50.0, step=0.1)
        phosphorus = st.number_input("Phosphorus (P)", min_value=0.0, max_value=145.0, value=30.0, step=0.1)
        potassium = st.number_input("Potassium (K)", min_value=0.0, max_value=205.0, value=20.0, step=0.1)

        # Submit button
        submitted = st.form_submit_button("Predict")

    # Load model
    model = load_model(MODEL_FILE)

    # If the form is submitted, make predictions
    if submitted:
        # Encode categorical inputs
        soil_type_mapping = {"Black": 0, "Clayey": 1, "Loamy": 2, "Red": 3, "Sandy": 4}
        crop_type_mapping = {"Barley": 0, "Cotton": 1, "Groundnuts": 2, "Maize": 3, "Millets": 4, 
                             "Oil seeds": 5, "Paddy": 6, "Pulses": 7, "Sugarcane": 8, "Wheat": 9}
        
        if soil_type in soil_type_mapping and crop_type in crop_type_mapping:
            encoded_soil_type = soil_type_mapping[soil_type]
            encoded_crop_type = crop_type_mapping[crop_type]
            inputs = [temperature, humidity, encoded_soil_type, encoded_crop_type, nitrogen, phosphorus, potassium]
            
            # Make prediction
            fertilizer_prediction = predict_fertilizer(model, inputs)
            st.success(f"Recommended Fertilizer: {fertilizer_prediction}")
        else:
            st.error("Invalid input values for Soil Type or Crop Type.")

# Run the Streamlit app
if __name__ == "__main__":
    main()