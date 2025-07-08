from flask import Flask, render_template, request, make_response, redirect, url_for
import joblib
import numpy as np
import pandas as pd
import io

app = Flask(__name__)

# --- Memuat semua model dan file yang dibutuhkan ---
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    meta = joblib.load('meta.pkl')
    threshold = meta['threshold']
except FileNotFoundError as e:
    print(f"FATAL ERROR: Tidak dapat memuat file model/scaler/meta: {e}")
    print("Aplikasi tidak dapat berjalan tanpa file-file ini.")
    # Set ke None agar lebih mudah diperiksa nanti
    model = scaler = threshold = None

# --- Mendefinisikan urutan kolom fitur yang benar (sudah dibersihkan) ---
FEATURE_COLS = [
    'BounceRates',
    'Administrative_Duration',
    'ProductRelated',
    'ProductRelated_Duration',
    'Administrative',
    'ExitRates',
    'PageValues'
]

@app.route('/')
def home():
    """Merender halaman utama."""
    return render_template('index.html', form_data={})

@app.route('/predict', methods=['POST'])
def predict():
    """Menangani prediksi dari input form manual."""
    if not all([model, scaler, threshold is not None]):
        return render_template('index.html', result="Error: Model tidak siap. Periksa log server.")

    try:
        # Ambil input dari form HTML menggunakan daftar FEATURE_COLS
        features = [float(request.form[col]) for col in FEATURE_COLS]
        arr = np.array(features).reshape(1, -1)

        # Scaling
        arr_scaled = scaler.transform(arr)

        # Predict prob dan hasil
        prob = model.predict_proba(arr_scaled)[:, 1][0]
        prediction = 1 if prob >= threshold else 0

        result = "Akan Membeli" if prediction == 1 else "Tidak Membeli"
        
        # Mengembalikan data form agar input tidak hilang setelah submit
        return render_template('index.html', result=result, prob=round(prob, 4), form_data=request.form)

    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")
    

# --- FUNGSI UNTUK UPLOAD CSV ---
@app.route('/upload', methods=['POST'])
def predict_csv():
    """Menangani prediksi dari upload file CSV."""
    if not all([model, scaler, threshold is not None]):
        return "Error: Model tidak siap. Periksa log server.", 500

    if 'file' not in request.files:
        return redirect(url_for('home'))
        
    file = request.files['file']

    if file.filename == '':
        return redirect(url_for('home'))

    if file and file.filename.endswith('.csv'):
        try:
            df = pd.read_csv(file)
            
            # Pastikan semua kolom yang diperlukan ada di CSV
            missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
            if missing_cols:
                return f"Error: File CSV Anda tidak memiliki kolom yang dibutuhkan: {', '.join(missing_cols)}", 400

            # Ambil data fitur dari DataFrame dengan urutan yang benar
            features_df = df[FEATURE_COLS]
            
            # Lakukan scaling pada semua data
            features_scaled = scaler.transform(features_df)
            
            # Lakukan prediksi probabilitas untuk semua baris
            probabilities = model.predict_proba(features_scaled)[:, 1]
            
            # Terapkan threshold untuk mendapatkan hasil prediksi
            predictions = [1 if p >= threshold else 0 for p in probabilities]

            # Tambahkan hasil ke DataFrame
            df['Hasil_Prediksi'] = ["Akan Membeli" if p == 1 else "Tidak Membeli" for p in predictions]
            df['Probabilitas_Pembelian'] = np.round(probabilities, 4)
            
            # Buat file CSV di memori untuk di-download
            output_stream = io.StringIO()
            df.to_csv(output_stream, index=False, encoding='utf-8')
            output_stream.seek(0)
            
            # Buat response untuk mengunduh file
            response = make_response(output_stream.getvalue())
            response.headers["Content-Disposition"] = "attachment; filename=hasil_prediksi.csv"
            response.headers["Content-type"] = "text/csv; charset=utf-8"
            
            return response

        except Exception as e:
            return f"Terjadi kesalahan saat memproses file: {e}", 500
    
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
