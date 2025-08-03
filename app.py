from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load model artifacts
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)


def create_input_data(form_data):
    """Create input data with all expected features"""
    # Initialize with default values for all features
    input_dict = {feature: 0 for feature in feature_names}

    # Set values for direct input features
    input_dict['Price (in rupees)'] = float(form_data.get('price', 0))
    input_dict['Carpet Area'] = float(form_data.get('carpet_area', 0))
    input_dict['Status'] = 1 if form_data.get('status') == 'Ready to Move' else 0
    input_dict['Bathroom'] = int(form_data.get('bathroom', 1))
    input_dict['Balcony'] = int(form_data.get('balcony', 0))
    input_dict['Car Parking'] = int(form_data.get('car_parking', 0))
    input_dict['Super Area'] = float(form_data.get('super_area', 0))
    input_dict['CurrentFloor'] = int(form_data.get('current_floor', 0))
    input_dict['TotalFloors'] = int(form_data.get('total_floors', 1))

    # Calculate derived features
    if input_dict['TotalFloors'] > 0:
        input_dict['Floor ratio'] = input_dict['CurrentFloor'] / input_dict['TotalFloors']

    # Handle categorical features
    location = form_data.get('location', 'bangalore').lower().replace('-', '_')
    input_dict[f'location_{location}'] = 1

    furnishing = form_data.get('furnishing', 'Semi-Furnished')
    if furnishing == 'Unfurnished':
        input_dict['Furnishing_-1'] = 0
        input_dict['Furnishing_0'] = 1
    elif furnishing == 'Semi-Furnished':
        input_dict['Furnishing_-1'] = 0
        input_dict['Furnishing_1'] = 1
    elif furnishing == 'Furnished':
        input_dict['Furnishing_-1'] = 0
        input_dict['Furnishing_2'] = 1
    else:
        input_dict['Furnishing_-1'] = 1

    # Handle floor level
    if input_dict['TotalFloors'] == 0:
        input_dict['Floor level_Unknown'] = 1
    elif input_dict['CurrentFloor'] == 0:
        input_dict['Floor level_Ground'] = 1
    else:
        floor_ratio = input_dict.get('Floor ratio', 0)
        if floor_ratio <= 0.33:
            input_dict['Floor level_Low'] = 1
        elif floor_ratio <= 0.66:
            input_dict['Floor level_Mid'] = 1
        else:
            input_dict['Floor level_High'] = 1

    return input_dict


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Create input data with all features
        input_dict = create_input_data(request.form)

        # Create DataFrame with features in correct order
        df = pd.DataFrame([input_dict])
        df = df[feature_names]  # Ensure correct feature order

        # Scale features
        scaled_features = scaler.transform(df)

        # Predict
        prediction = model.predict(scaled_features)

        return render_template('index.html', prediction=round(prediction[0], 2))

    return render_template('index.html', prediction=None)


if __name__ == '__main__':
    app.run(debug=True)
