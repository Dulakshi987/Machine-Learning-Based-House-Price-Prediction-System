# app.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

# -----------------------
# Load the trained pipeline
# -----------------------
pipeline = joblib.load('house_price_model.pkl')  # Pipeline includes scaler + model

# -----------------------
# Streamlit App
# -----------------------
st.title("House Price Prediction")

# User inputs
OverallQual = st.slider("Overall Quality (1-10)", 1, 10, 5)
GrLivArea = st.number_input("Living Area (sqft)", 500, 5000, 1500)
GarageCars = st.slider("Garage Cars", 0, 4, 1)

# Create a DataFrame for the input
# Make sure column names match training data
input_data = pd.DataFrame([{
    'OverallQual': OverallQual,
    'GrLivArea': GrLivArea,
    'GarageCars': GarageCars
}])

# Predict the price
predicted_price = pipeline.predict(input_data)

st.subheader("Predicted House Price")
st.write(f"Rs:{predicted_price[0]:,.2f}")
