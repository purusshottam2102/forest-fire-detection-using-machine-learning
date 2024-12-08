from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from tensorflow.keras.losses import MeanSquaredError
app = Flask(__name__)

# Load the retrained model and scaler
model = load_model('forest_fire_nn_fewer_features.h5', custom_objects={'mse':MeanSquaredError()})
scaler = joblib.load('scaler_fewer_features.pkl')

# Default values for prediction
DEFAULT_VALUES = {
    'temp': 15.0,
    'RH': 40.0,
    'wind': 4.0,
    'rain': 0.0
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract or use default inputs
    temp = float(request.form.get('temp', DEFAULT_VALUES['temp']))
    RH = float(request.form.get('RH', DEFAULT_VALUES['RH']))
    wind = float(request.form.get('wind', DEFAULT_VALUES['wind']))
    rain = float(request.form.get('rain', DEFAULT_VALUES['rain']))

    # Prepare features for prediction
    features = np.array([[temp, RH, wind, rain]])
    scaled_features = scaler.transform(features)

    # Predict burned area
    prediction = model.predict(scaled_features)
    predicted_area = np.expm1(prediction[0][0])  # Reverse log1p transformation

    return render_template('result.html', prediction=round(predicted_area, 2))

if __name__ == '__main__':
    app.run(debug=True)
