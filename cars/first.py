# app.py
import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model

# Load the saved model
model = load_model('car_model')  # No need to add .pkl extension

# Title of the app
st.title("Car Price Prediction App")

# Subtitle
st.write("Enter car details to predict its price using a Random Forest model.")

# Input fields for car details
brand = st.text_input("Brand", value="Jeep")
model_name = st.text_input("Model", value="Compass 2.0")
manufacture = st.slider("Manufacture Year", min_value=1990, max_value=2025, value=2017, step=1)
kms_driven = st.slider("KMs Driven", min_value=0, max_value=500000, value=86226, step=100)
fuel_type = st.selectbox("Fuel Type", options=["Petrol", "Diesel", "LPG"], index=1)
transmission = st.selectbox("Transmission", options=["Manual", "Automatic"], index=0)
ownership = st.slider("Ownership (1st, 2nd, etc.)", min_value=1, max_value=5, value=1, step=1)
engine = st.slider("Engine (cc)", min_value=1000, max_value=5000, value=1956, step=1)
seats = st.slider("Seats", min_value=2, max_value=10, value=5, step=1)

# Calculate derived features
car_age = 2025 - manufacture
kms_per_year = kms_driven / car_age if car_age > 0 else 0

# Create a DataFrame for prediction
input_data = pd.DataFrame({
    'brand': [brand],
    'model': [model_name],
    'manufacture': [manufacture],
    'kms_driven': [kms_driven],
    'fuel_type': [fuel_type],
    'transmission': [transmission],
    'ownership': [ownership],
    'engine': [engine],
    'Seats': [seats],
    'car_age': [car_age],
    'kms_per_year': [kms_per_year]
})

# Predict button
if st.button("Predict Price"):
    # Make prediction
    prediction = predict_model(model, data=input_data)
    
    # Since log_transform is not used, prediction is already in original scale
    predicted_price = prediction['prediction_label'].iloc[0]
    
    # Display the predicted price
    st.success(f"Predicted Price: ₹{predicted_price:,.2f}")

# Add some instructions
st.write("### Instructions")
st.write("1. Enter the car details in the fields above.")
st.write("2. Click the 'Predict Price' button to get the predicted price.")
st.write("3. The predicted price will be displayed in rupees.")