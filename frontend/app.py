import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend running locally in the Colab container
BACKEND_URL = "http://127.0.0.1:7860"

st.title("Future Sales Prediction App")
st.write("This app predicts future sales based on SuperKart retail chain features.")

# --- SECTION 1: SINGLE PREDICTION SLIDERS ---
st.subheader("Single Record Prediction")
CRIM = st.slider("Enter Product Weight", 0.0, 100.0, 0.2, 0.1)
ZN = st.slider("Enter Product Allocated Area", 0.0, 100.0, 12.0, 1.0)
INDUS = st.slider("Enter Product MRP", 0.0, 30.0, 11.0, 0.5)
NX = st.slider("Enter Store Age in Years", 0.0, 1.0, 0.55, 0.01)
RM = st.selectbox(
    "Choose Product Sugar Content", 
    options=["Low Sugar", "No Sugar", "Regular", "reg"]
)
AGE = st.selectbox(
    "Choose Store Size",
    options=["High", "Medium", "Small"]
)
DIS = st.selectbox(
    "Choose Store Location",
    options=["Tier 1", "Tier 2", "Tier 3"]
)
RAD = st.selectbox(
    "Choose Product Type",
    options=["Non Perishables", "Perishables"]
)
TAX = st.selectbox(
    "Choose Store Type",
    options=["Departmental Store", "Food Mart", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"]
)
PTRATIO = st.selectbox(
    "Choose Product Id char",
    options=["DR", "FD", "NC"]
)

input_data = {
    'CRIM': CRIM, 'ZN': ZN, 'INDUS': INDUS, 'NX': NX, 'RM': RM,
    'AGE': AGE, 'DIS': DIS, 'RAD': RAD, 'TAX': TAX, 'PTRATIO': PTRATIO
}

if st.button("Predict Single", type='primary'):
    response = requests.post(f"{BACKEND_URL}/v1/predict", json=input_data)
    if response.status_code == 200:
        result = response.json()
        st.success(f"📈 Predicted Future Sales: **\${result.get('prediction', 0):,.2f}**")
    else:
        st.error("Error in API single prediction request")


# --- SECTION 2: BATCH PREDICTION (MULTIPLE FILES) ---
st.markdown("---")
st.subheader("Batch Prediction Matrix")
st.write("Upload both your raw identity file (`file1.csv`) and engineered features file (`file2.csv`) below.")

file1 = st.file_uploader("Upload Raw Metadata (file1.csv)", type=["csv"])
file2 = st.file_uploader("Upload Engineered Features (file2.csv)", type=["csv"])

if file1 is not None and file2 is not None:
    if st.button("Predict for Batch", type='primary'):
        # Map matching file streams to multi-part key dictionary
        csv_files = {
            "file1": (file1.name, file1.getvalue(), "text/csv"),
            "file2": (file2.name, file2.getvalue(), "text/csv")
        }
        
        with st.spinner("Processing batch records through server pipeline..."):
            response = requests.post(f"{BACKEND_URL}/v1/predictbatch", files=csv_files)

        if response.status_code == 200:
            result = response.json()
            st.header("🏆 Batch Prediction Results Map")
            
            # Convert dictionary mapping results to clean display table
            if "results" in result:
                df_results = pd.DataFrame(result["results"].items(), columns=["Unique_Tracking_Key", "Predicted_Sales"])
                st.dataframe(df_results)
            else:
                st.write(result)
        else:
            st.error(f"Error in API request: {response.text}")
