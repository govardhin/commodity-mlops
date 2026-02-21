import streamlit as st
import requests
import json

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Commodity Price Forecasting Platform",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000/predict"

st.title("Commodity Price Forecasting Platform")
st.caption("Connected to Production FastAPI Model Server")

st.divider()

# ===============================
# INPUT SECTION
# ===============================

st.subheader("Enter Market Indicators")

col1, col2 = st.columns(2)

with col1:
    open_price = st.number_input("Open Price", value=0.0)
    high_price = st.number_input("High Price", value=0.0)
    low_price = st.number_input("Low Price", value=0.0)

with col2:
    volume = st.number_input("Volume", value=0.0)
    prev_close = st.number_input("Previous Close", value=0.0)

st.divider()

# ===============================
# PREDICTION BUTTON
# ===============================

if st.button("Generate Forecast"):

    payload = {
        "Open": open_price,
        "High": high_price,
        "Low": low_price,
        "Vol": volume,
        "Prev_Close": prev_close
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            predicted_price = result["predicted_price"]

            st.success(f"Predicted Price: {predicted_price}")

        else:
            st.error("API returned an error")

    except Exception as e:
        st.error("Cannot connect to FastAPI server. Make sure it is running.")
