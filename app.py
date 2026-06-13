import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
with open('locations.json', 'r') as f:
    data = json.load(f)
locations = data['locations']
columns = data['columns']

st.set_page_config(page_title="Bangalore House Price Predictor", page_icon="🏠")
st.title("🏠 Bangalore House Price Predictor")
st.markdown("Predict house prices in Bangalore instantly!")

col1, col2 = st.columns(2)
with col1:
    location = st.selectbox("📍 Location", sorted(locations))
    total_sqft = st.number_input("📐 Total Sqft", min_value=300, max_value=10000, value=1000)
with col2:
    bath = st.number_input("🚿 Bathrooms", min_value=1, max_value=10, value=2)
    bhk = st.number_input("🛏️ BHK", min_value=1, max_value=10, value=2)

if st.button("🔍 Predict Price"):
    input_data = np.zeros(len(columns))
    if location in columns:
        input_data[columns.index(location)] = 1
    input_data[columns.index('total_sqft')] = total_sqft
    input_data[columns.index('bath')] = bath
    input_data[columns.index('bhk')] = bhk
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)[0]
    st.success(f"💰 Estimated Price: ₹ {round(prediction, 2)} Lakhs")