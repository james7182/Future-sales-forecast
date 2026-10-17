import streamlit as st
import pandas as pd
import requests


# Base URL of the Flask backend
BACKEND_URL = "http://backend:7860"

# Streamlit UI for Boston Housing Price Prediction
st.title("Boston Housing Price Prediction App")
st.write("This app predicts the median value of owner-occupied homes (`MEDV`) in $1000s based on Boston housing dataset features.")
st.write("Move the sliders below to adjust values and get a prediction.")

# Collect user input using sliders
CRIM = st.slider("Per capita crime rate by town (CRIM)", 0.0, 100.0, 0.2, 0.1)
ZN = st.slider("Proportion of residential land zoned for lots over 25,000 sq.ft. (ZN)", 0.0, 100.0, 12.0, 1.0)
INDUS = st.slider("Proportion of non-retail business acres per town (INDUS)", 0.0, 30.0, 11.0, 0.5)
NX = st.slider("Nitric oxides concentration (NX)", 0.0, 1.0, 0.55, 0.01)
RM = st.slider("Average number of rooms per dwelling (RM)", 3.0, 9.0, 6.3, 0.1)
AGE = st.slider("Proportion of owner-occupied units built prior to 1940 (AGE)", 0.0, 100.0, 65.0, 1.0)
DIS = st.slider("Weighted distances to employment centers (DIS)", 1.0, 12.0, 4.0, 0.1)
RAD = st.slider("Index of accessibility to radial highways (RAD)", 1, 24, 4, 1)
TAX = st.slider("Full-value property tax rate per $10,000 (TAX)", 100, 700, 300, 1)
PTRATIO = st.slider("Pupil-teacher ratio by town (PTRATIO)", 10.0, 25.0, 19.0, 0.1)
LSTAT = st.slider("% lower status of the population (LSTAT)", 0.0, 40.0, 12.0, 0.1)

# Categorical feature
CHAS = st.selectbox("Charles River dummy variable (CHAS)", ["0 (No)", "1 (Yes)"])
CHAS_value = 1 if CHAS.startswith("1") else 0

# Create input DataFrame
input_data = {
    'CRIM': CRIM,
    'ZN': ZN,
    'INDUS': INDUS,
    'NX': NX,
    'RM': RM,
    'AGE': AGE,
    'DIS': DIS,
    'RAD': RAD,
    'TAX': TAX,
    'PTRATIO': PTRATIO,
    'LSTAT': LSTAT,
    'CHAS': CHAS_value
}

if st.button("Predict", type='primary'):
    # Replace with your deployed backend URL
    response = requests.post(
        # "https://{BACKEND_URL}/v1/house",
        f"{BACKEND_URL}/v1/house",
        json=input_data
    )

    if response.status_code == 200:
        result = response.json()
        predicted_price = result["Predicted_MEDV"]
        st.success(f"🏡 Predicted Median House Value: **${predicted_price * 1000:.2f}**")
    else:
        st.error("Error in API request")


# Batch Prediction
st.subheader("Batch Prediction")

file = st.file_uploader("Upload CSV file", type=["csv"])

if file is not None:
    if st.button("Predict for Batch", type='primary'):
        response = requests.post(
            # "https://{BACKEND_URL}/v1/housebatch",
            f"{BACKEND_URL}/v1/housebatch",
            files={"file": file}
        )

        if response.status_code == 200:
            result = response.json()
            st.header("Batch Prediction Results")
            st.write(result)
        else:
            st.error("Error in API request")
