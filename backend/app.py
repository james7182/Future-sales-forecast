from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
import io

# Initialize Flask App
app = Flask(__name__)

# Load the saved model directly using joblib
# Fallback path checking logic
model = joblib.load('sales_model_v1_0.joblib')

# Single prediction
@app.post('/v1/predict')
def predict():
    # Get JSON data from the request
    house_data_raw = request.get_json() or {}

    # Use request data or default to sample data if empty
    house_data = {
        "Product_Weight": house_data_raw.get("Product_Weight"),
        "Product_Allocated_Area": house_data_raw.get("Product_Allocated_Area"),
        "Product_MRP": house_data_raw.get("Product_MRP"),
        "Store_Age_Years": house_data_raw.get("Store_Age_Years"),
        "Product_Sugar_Content": house_data_raw.get("Product_Sugar_Content"),
        "Store_Size": house_data_raw.get("Store_Size"),
        "Store_Location_City_Type": house_data_raw.get("Store_Location_City_Type"),
        "Product_Type": house_data_raw.get("Product_Type"),
        "Store_Type": house_data_raw.get("Store_Type"),
        "Item_Class": house_data_raw.get("Item_Class")
    }

    Product_Sugar_Content_No_Sugar = False
    Product_Sugar_Content_Reg = False
    Product_Sugar_Content_Regular = False

    if house_data["Product_Sugar_Content"] == "No Sugar":
        Product_Sugar_Content_No_Sugar = True
    if house_data["Product_Sugar_Content"] == "Reg":
        Product_Sugar_Content_Reg = True
    if house_data["Product_Sugar_Content"] == "Regular":
        Product_Sugar_Content_Regular = True

    Store_Size_Medium = False
    Store_Size_Small = False
    if house_data["Store_Size"] == "Medium":
        Store_Size_Medium = True
    if house_data["Store_Size"] == "Small":
        Store_Size_Small = True

    Store_Location_City_Type_Tier_2 = False
    Store_Location_City_Type_Tier_3 = False
    # Fixed matching values to align with training string labels (with space)
    if house_data["Store_Location_City_Type"] in ["Tier2", "Tier 2"]:
        Store_Location_City_Type_Tier_2 = True
    if house_data["Store_Location_City_Type"] in ["Tier3", "Tier 3"]:
        Store_Location_City_Type_Tier_3 = True

    # Map general incoming Product_Type categories using your exact training list
    perishable_categories = ['Dairy', 'Meat', 'Fruits and Vegetables', 'Breakfast', 'Seafood', 'Perishables']
    Product_Type_Perishables = False
    if house_data["Product_Type"] in perishable_categories:
        Product_Type_Perishables = True

    Store_Type_Food_Mart = False
    Store_Type_Supermarket_Type1 = False
    Store_Type_Supermarket_Type2 = False
    Store_Type_Supermarket_Type3 = False

    if house_data["Store_Type"] == "Food Mart":
        Store_Type_Food_Mart = True
    if house_data["Store_Type"] == "Supermarket Type1":
        Store_Type_Supermarket_Type1 = True
    if house_data["Store_Type"] == "Supermarket Type2":
        Store_Type_Supermarket_Type2 = True
    if house_data["Store_Type"] == "Supermarket Type3":
        Store_Type_Supermarket_Type3 = True

    Item_Class_FD = False
    Item_Class_NC = False
    if house_data["Item_Class"] == "FD":
        Item_Class_FD = True
    if house_data["Item_Class"] == "NC":
        Item_Class_NC = True

    # Raw dictionary data mapping variables to their exact clean column equivalents
    data = {
        "Product_Weight": [house_data["Product_Weight"]],
        "Product_Allocated_Area": [house_data["Product_Allocated_Area"]],
        "Product_MRP": [house_data["Product_MRP"]],
        "Store_Age_Years": [house_data["Store_Age_Years"]],
        "Product_Type_Perishables": [Product_Type_Perishables],
        "Product_Sugar_Content_No Sugar": [Product_Sugar_Content_No_Sugar],
        "Product_Sugar_Content_Reg": [Product_Sugar_Content_Reg],
        "Product_Sugar_Content_Regular": [Product_Sugar_Content_Regular],
        "Store_Size_Medium": [Store_Size_Medium],
        "Store_Size_Small": [Store_Size_Small],
        "Store_Location_City_Type_Tier 2": [Store_Location_City_Type_Tier_2],
        "Store_Location_City_Type_Tier 3": [Store_Location_City_Type_Tier_3],
        "Store_Type_Food Mart": [Store_Type_Food_Mart],
        "Store_Type_Supermarket Type1": [Store_Type_Supermarket_Type1],
        "Store_Type_Supermarket Type2": [Store_Type_Supermarket_Type2],
        "Store_Type_Supermarket Type3": [Store_Type_Supermarket_Type3],
        "Item_Class_FD": [Item_Class_FD],
        "Item_Class_NC": [Item_Class_NC]
    }

    # Initialize the DataFrame
    df = pd.DataFrame(data)

    # -------------------------------------------------------------------------
    # CRITICAL SCHEMATIC ALIGNMENT
    # -------------------------------------------------------------------------
    # Enforce the exact structural column sequence expected by model.fit()
    expected_training_features = [
        "Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Age_Years",
        "Product_Type_Perishables", "Product_Sugar_Content_No Sugar",
        "Product_Sugar_Content_Reg", "Product_Sugar_Content_Regular",
        "Store_Size_Medium", "Store_Size_Small",
        "Store_Location_City_Type_Tier 2", "Store_Location_City_Type_Tier 3",
        "Store_Type_Food Mart", "Store_Type_Supermarket Type1",
        "Store_Type_Supermarket Type2", "Store_Type_Supermarket Type3",
        "Item_Class_FD", "Item_Class_NC"
    ]

    # Reindex forces column ordering to match perfectly
    df = df.reindex(columns=expected_training_features, fill_value=False)

    # Enforce precise scalar object types
    df = df.astype({
        "Product_Weight": "float64",
        "Product_Allocated_Area": "float64",
        "Product_MRP": "float64",
        "Store_Age_Years": "int64",
        "Product_Type_Perishables": "bool",
        "Product_Sugar_Content_No Sugar": "bool",
        "Product_Sugar_Content_Reg": "bool",
        "Product_Sugar_Content_Regular": "bool",
        "Store_Size_Medium": "bool",
        "Store_Size_Small": "bool",
        "Store_Location_City_Type_Tier 2": "bool",
        "Store_Location_City_Type_Tier 3": "bool",
        "Store_Type_Food Mart": "bool",
        "Store_Type_Supermarket Type1": "bool",
        "Store_Type_Supermarket Type2": "bool",
        "Store_Type_Supermarket Type3": "bool",
        "Item_Class_FD": "bool",
        "Item_Class_NC": "bool"
    })

    # Make predictions
    predictions = model.predict(df)
    prediction_output = predictions.tolist()
    print("predictions")
    print(predictions)

    return jsonify({
        "status": "success",
        "prediction": prediction_output[0]
    })

@app.route('/v1/predictbatch', methods=['POST'])  # Use app.route for Flask endpoints
def predictbatch():
    # 2. Extract the file handles from the incoming request payload (Matching Colab Client Keys)
    batch_file = request.files.get('file_1') or request.files.get('file1')

    print(f"Batch File extracted: {batch_file}", flush=True)

    if not batch_file:
        return jsonify({"error": "Missing file_1 multi-part form upload"}), 400

    # CORRECTED: Exact sequential column order matching your 18 training features
    EXPECTED_COLUMNS = [
        "Product_Weight",
        "Product_Allocated_Area",
        "Product_MRP",
        "Store_Age_Years",
        "Product_Type_Perishables",  # Position 4
        "Product_Sugar_Content_No Sugar",
        "Product_Sugar_Content_Reg",
        "Product_Sugar_Content_Regular",
        "Store_Size_Medium",
        "Store_Size_Small",
        "Store_Location_City_Type_Tier 2",
        "Store_Location_City_Type_Tier 3",
        "Store_Type_Food Mart",
        "Store_Type_Supermarket Type1",
        "Store_Type_Supermarket Type2",
        "Store_Type_Supermarket Type3",
        "Item_Class_FD",
        "Item_Class_NC"
    ]

    try:
        # 3. Read directly into memory space using Pandas
        temp_batch_df = pd.read_csv(io.BytesIO(batch_file.read()))

        # --- STEP 1: RESTORE MISSING FEATURE ENGINEERING (Mirroring Training Phase) ---

        # Extract Item Class tracking identifier prefix string
        if 'Product_Id_char' in temp_batch_df.columns:
            temp_batch_df['Item_Class'] = temp_batch_df['Product_Id_char']

        # NEW FEATURE ENGINEERING: Map Product_Type_Category to Perishables Boolean
        perishable_categories = ['Dairy', 'Meat', 'Fruits and Vegetables', 'Breakfast', 'Seafood']
        if 'Product_Type_Category' in temp_batch_df.columns:
            temp_batch_df['Product_Type_Perishables'] = temp_batch_df['Product_Type_Category'].isin(perishable_categories)
        else:
            # Fallback if the column is named Product_Type instead
            col_name = 'Product_Type' if 'Product_Type' in temp_batch_df.columns else temp_batch_df.columns[0]
            temp_batch_df['Product_Type_Perishables'] = temp_batch_df[col_name].isin(perishable_categories)

        # Save tracking index before dropping unique non-feature string handles
        metadata_keys = temp_batch_df['Product_Id_char'].astype(str).tolist() if 'Product_Id_char' in temp_batch_df.columns else [f"Row_{i}" for i in range(len(temp_batch_df))]

        # Standardizing baseline categoricals text string structure to match training clean up
        if 'Product_Sugar_Content' in temp_batch_df.columns:
            temp_batch_df['Product_Sugar_Content'] = temp_batch_df['Product_Sugar_Content'].astype(str).str.title().str.strip()

        # --- STEP 2: CLEAN EXTRA TRACKING IDENTIFIERS UNUSED BY NEW SCHEMA ---
        # Crucial: Drop Product_Type_Category so it doesn't get dummy encoded into 17 columns
        drop_cols = ['Product_Id_char', 'Product_Type_Category', 'Product_Type']
        features_only = temp_batch_df.drop(columns=[col for col in drop_cols if col in temp_batch_df.columns])

        # --- STEP 3: ONE-HOT ENCODE RECONSTRUCTED MATRIX (Product_Type_Category is removed) ---
        df_consolidated = pd.get_dummies(
            features_only,
            columns=['Product_Sugar_Content', 'Store_Size', 'Store_Location_City_Type', 'Store_Type', 'Item_Class'],
            drop_first=True,
            dtype=bool
        )

        # --- STEP 4: ALIGN DATAFRAME SCHEMA WITH NEW MATRIX SHAPE ---
        X_test = df_consolidated.reindex(columns=EXPECTED_COLUMNS, fill_value=False)

        # Enforce exact training types to satisfy validation restrictions
        X_test = X_test.astype({
            "Product_Weight": "float64",
            "Product_Allocated_Area": "float64",
            "Product_MRP": "float64",
            "Store_Age_Years": "int64",
            "Product_Type_Perishables": "bool",
            "Product_Sugar_Content_No Sugar": "bool",
            "Product_Sugar_Content_Reg": "bool",
            "Product_Sugar_Content_Regular": "bool",
            "Store_Size_Medium": "bool",
            "Store_Size_Small": "bool",
            "Store_Location_City_Type_Tier 2": "bool",
            "Store_Location_City_Type_Tier 3": "bool",
            "Store_Type_Food Mart": "bool",
            "Store_Type_Supermarket Type1": "bool",
            "Store_Type_Supermarket Type2": "bool",
            "Store_Type_Supermarket Type3": "bool",
            "Item_Class_FD": "bool",
            "Item_Class_NC": "bool"
        })

        # Run your ML evaluation
        y_pred_test = model.predict(X_test)
        print(f"Prediction size: {len(y_pred_test)}")

        # Map predictions back to tracking index keys
        mapped_results = dict(zip(metadata_keys, y_pred_test.tolist()))

        return jsonify({
            "status": "success",
            "results": mapped_results
        }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to process batch payload: {str(e)}"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)
