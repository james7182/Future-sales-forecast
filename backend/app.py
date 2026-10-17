import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask("Boston House Predictor ")

# Load the trained Boston housing model
model = joblib.load("boston_housing_model_v1_0.joblib")

# Define a route for the home page
@app.get('/')
def home():
    return "Welcome to the Boston House Price Prediction API!"

# Define an endpoint to predict price for a single house
@app.post('/v1/house')
def predict_house_price():
    # Get JSON data from the request
    house_data = request.get_json()

    # Extract relevant house features from the input data
    sample = {
        'CRIM': house_data['CRIM'],
        'ZN': house_data['ZN'],
        'INDUS': house_data['INDUS'],
        'CHAS': house_data['CHAS'],
        'NX': house_data['NX'],     # should be NOX in your dataset, check consistency
        'RM': house_data['RM'],
        'AGE': house_data['AGE'],
        'DIS': house_data['DIS'],
        'RAD': house_data['RAD'],
        'TAX': house_data['TAX'],
        'PTRATIO': house_data['PTRATIO'],
        'LSTAT': house_data['LSTAT']
    }

    # Convert the extracted data into a DataFrame
    input_data = pd.DataFrame([sample])

    # Make a prediction using the trained model
    prediction = model.predict(input_data).tolist()[0]

    # Return the prediction as a JSON response
    return jsonify({'Predicted_MEDV': prediction})

# Define an endpoint to predict price for a batch of houses
@app.post('/v1/housebatch')
def predict_house_batch():
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the file into a DataFrame
    input_data = pd.read_csv(file)

    # Make predictions for the batch data
    predictions = model.predict(input_data).tolist()

    # Add predictions to the DataFrame
    input_data['Predicted_MEDV'] = predictions

    # Convert results to dictionary
    result = input_data.to_dict(orient="records")

    return jsonify(result)

# Run the Flask app in debug mode
if __name__ == '__main__':
    app.run(debug=True)
