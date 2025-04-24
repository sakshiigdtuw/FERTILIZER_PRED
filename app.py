from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and encoders
model = pickle.load(open('fertilizer_model.pkl', 'rb'))
soil_encoder = pickle.load(open('soil_encoder.pkl', 'rb'))
crop_encoder = pickle.load(open('crop_encoder.pkl', 'rb'))
fert_encoder = pickle.load(open('fert_encoder.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    features = [
        int(data['Temperature']),
        int(data['Humidity']),
        int(data['Moisture']),
        soil_encoder.transform([data['Soil']])[0],
        crop_encoder.transform([data['Crop']])[0],
        int(data['Nitrogen']),
        int(data['Potassium']),
        int(data['Phosphorous']),
    ]

    prediction = model.predict([features])[0]
    fertilizer = fert_encoder.inverse_transform([prediction])[0]

    return render_template('index.html', prediction_text=f"Recommended Fertilizer: {fertilizer}")

if __name__ == '__main__':
    app.run(debug=True)
