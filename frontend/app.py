import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend
BACKEND_URL = "http://backend:7860"

# Streamlit UI for Future sales Prediction
st.title("Future sales Prediction App")
st.write("This app predicts the Future sales based on SuperKart retail chain supermarkets dataset features.")
st.write("Move the sliders below to adjust values and get a prediction.")

# Collect user input using sliders
CRIM = st.slider("Enter Product Weight", 0.0, 100.0, 0.2, 0.1)
ZN = st.slider("Enter Product Allocated Area", 0.0, 100.0, 12.0, 1.0)
INDUS = st.slider("Enter Product MRP", 0.0, 30.0, 11.0, 0.5)
NX = st.slider("Enter Store Age in Years", 0.0, 1.0, 0.55, 0.01)
RM = st.slider("Choose Product Sugar Content (No Sugar/Low Sugar/Regular/reg)", 3.0, 9.0, 6.3, 0.1)
AGE = st.slider("Choose Store Size Medium (Small/Medium/High)", 0.0, 100.0, 65.0, 1.0)
DIS = st.slider("Choose Store Location (Tier 1/Tier 2/Tier 3)", 1.0, 12.0, 4.0, 0.1)
RAD = st.slider("Choose Product Type (Perishables/Non Perishables)", 1, 24, 4, 1)
TAX = st.slider("Choose Store Type (Departmental Store/Food Mart/Supermarket Type1/Supermarket Type2/Supermarket Type3)", 100, 700, 300, 1)
PTRATIO = st.slider("Choose Item Class (DR/FD/NC)", 10.0, 25.0, 19.0, 0.1)
# Categorical feature

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
    'PTRATIO': PTRATIO
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
