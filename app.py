# app.py
import streamlit as st
import joblib
import numpy as np
import os

st.title("📘 Student Score Prediction App")
st.write("Enter study hours to predict expected exam score.")

# Name of the model file you uploaded to the repo
MODEL_FILENAME = "model.pkl"   # <-- make sure this file is present in the same folder as app.py

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        # Show a helpful message in Streamlit rather than crashing
        st.error(f"Model file not found: {path}. Make sure you uploaded it to the repo and the filename matches.")
        return None
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model(MODEL_FILENAME)
if model is None:
    st.stop()  # stop the app here so the rest doesn't run without a model

# User input
hours = st.number_input("Enter Hours Studied:", min_value=0.0, max_value=24.0, step=0.5, value=1.0)

if st.button("Predict Score"):
    try:
        prediction = model.predict([[hours]])[0]
        st.success(f"📊 Predicted Score: {prediction:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
