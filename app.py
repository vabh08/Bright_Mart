import joblib
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('linear_regression_model.sav')

@app.route('/')
def home():
    return "Welcome to the Sales Prediction API! Use /predict to make predictions."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        data = request.get_json(force=True)

        # Expecting data in the format:
        # {'TV': 230.1, 'Radio': 37.8, 'Newspaper': 69.2}
        # Convert dictionary to a DataFrame for prediction
        input_df = pd.DataFrame([data])

        # Ensure the columns are in the correct order as trained
        # Assuming the model was trained with 'TV', 'Radio', 'Newspaper'
        expected_columns = ['TV', 'Radio', 'Newspaper']
        if not all(col in input_df.columns for col in expected_columns):
            return jsonify({'error': 'Missing one or more input features. Expected: TV, Radio, Newspaper'}), 400
        
        input_df = input_df[expected_columns]

        # Make prediction
        prediction = model.predict(input_df)

        # Return the prediction as JSON
        return jsonify({'sales_prediction': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # In a production environment, you might use a more robust WSGI server like Gunicorn
    # For local testing, you can run:
    # flask run --host=0.0.0.0 --port=5000
    # or using app.run:
    app.run(debug=True, host='0.0.0.0', port=5000)
