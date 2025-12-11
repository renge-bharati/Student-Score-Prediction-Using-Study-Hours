import streamlit as st
import joblib
import numpy as np

st.title("📘 Student Score Prediction App")
st.write("Enter study hours to predict expected exam score.")

# Load model
model = load("model.pkl")

# User input
hours = st.number_input("Enter Hours Studied:", min_value=0.0, max_value=24.0, step=0.5)

if st.button("Predict Score"):
    prediction = model.predict([[hours]])[0]
    st.success(f"📊 Predicted Score: {prediction:.2f}")
