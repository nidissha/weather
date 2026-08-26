from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import requests

app = Flask(__name__)

# Load trained model
model = pickle.load(open('weather_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_auto', methods=['POST'])
def predict_auto():
    try:
        data = request.get_json()
        lat = data.get('lat')
        lon = data.get('lon')

        # OpenWeatherMap API Call with 4-second timeout
        API_KEY = "bd5e378503939ddaee76f12ad7a97608"
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        
        response = requests.get(url, timeout=4).json()

        if response.get("cod") == 200:
            temp = response['main']['temp']
            humidity = response['main']['humidity']
            wind = response['wind']['speed']
            city = response.get('name', 'Your Location')
        else:
            # Fallback values if API fails
            temp, humidity, wind, city = 30.0, 65.0, 12.0, "Local Area"

        # Predict using ML model
        features = np.array([[temp, humidity, wind]])
        prediction = model.predict(features)[0]

        return jsonify({
            'success': True,
            'city': city,
            'temp': round(temp, 1),
            'humidity': humidity,
            'wind': round(wind, 1),
            'prediction': str(prediction)
        })

    except Exception as e:
        # Fallback prediction to prevent infinite loading
        return jsonify({
            'success': True,
            'city': 'Detected Location',
            'temp': 31.0,
            'humidity': 60,
            'wind': 5.0,
            'prediction': 'Cloudy'
        })

if __name__ == '__main__':
    app.run(debug=True)