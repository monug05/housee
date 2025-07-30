from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the expected feature columns (from training)
# You can also hardcode these based on df1_ohe.columns[:-1] if needed
model_features = [
    'Carpet Area',
    'Bathroom',
    'Balcony',
    'Car Parking',
    'Super Area',
    'Current Floor',
    'Total Floors',
    'Status_Ready',
    'Status_Under'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None

    if request.method == 'POST':
        try:
            # Get form data
            carpet_area = float(request.form['carpet_area'])
            status = request.form['status']
            bathroom = int(request.form['bathroom'])
            balcony = int(request.form['balcony'])
            car_parking = int(request.form['car_parking'])
            super_area = float(request.form['super_area'])
            current_floor = int(request.form['current_floor'])
            total_floors = int(request.form['total_floors'])

            # Manual one-hot encoding
            status_ready = 1 if status == 'Ready to Move' else 0
            status_under = 1 if status == 'Under Construction' else 0

            # Build feature vector in correct order
            input_data = pd.DataFrame([[carpet_area, bathroom, balcony, car_parking, super_area,
                                        current_floor, total_floors, status_ready, status_under]],
                                      columns=model_features)

            # Make prediction
            pred = model.predict(input_data)[0]
            prediction = round(pred, 2)

        except Exception as e:
            error = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction, error=error)

if __name__ == '__main__':
    app.run(debug=True)
