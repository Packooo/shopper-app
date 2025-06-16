from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model dan scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
meta = joblib.load('meta.pkl')
threshold = meta['threshold']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil input dari form HTML
        features = [
            float(request.form['BounceRates']),
            float(request.form['Administrative_Duration']),
            float(request.form['ProductRelated']),
            float(request.form['ProductRelated_Duration']),
            float(request.form['Administrative']),
            float(request.form['ExitRates']),
            float(request.form['PageValues'])
        ]
        arr = np.array(features).reshape(1, -1)

        # Scaling
        arr_scaled = scaler.transform(arr)

        # Predict prob dan hasil
        prob = model.predict_proba(arr_scaled)[:, 1][0]
        prediction = int(prob >= threshold)

        result = "Akan Membeli" if prediction == 1 else "Tidak Membeli"
        return render_template('index.html', result=result, prob=round(prob, 4))

    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
