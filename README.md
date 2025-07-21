# Aplikasi Prediksi Niat Pembelian Online

Aplikasi web berbasis Flask yang menggunakan machine learning untuk memprediksi niat pembelian pelanggan berdasarkan data interaksi pengguna di halaman web.

## 📋 Deskripsi

Aplikasi ini menganalisis perilaku pengunjung website untuk memprediksi kemungkinan mereka melakukan pembelian. Sistem menggunakan model machine learning yang telah dilatih dengan data historis interaksi pengguna untuk memberikan prediksi yang akurat.

## ✨ Fitur Utama

- **Prediksi Manual**: Input data sesi pengunjung secara manual melalui form web
- **Prediksi Massal**: Upload file CSV untuk prediksi multiple data sekaligus
- **Interface Responsif**: Antarmuka web yang user-friendly dan responsif
- **Download Hasil**: Unduh hasil prediksi dalam format CSV
- **Real-time Prediction**: Prediksi langsung dengan probabilitas pembelian

## 🔧 Teknologi yang Digunakan

- **Backend**: Flask (Python)
- **Machine Learning**: scikit-learn, joblib
- **Data Processing**: pandas, numpy
- **Frontend**: HTML5, CSS3, JavaScript
- **Styling**: Font Awesome icons

## 📊 Parameter Input

Aplikasi membutuhkan 7 parameter utama untuk melakukan prediksi:

1. **Bounce Rates** (0-1): Tingkat bounce rate halaman
2. **Exit Rates** (0-1): Tingkat exit rate halaman
3. **Page Values**: Nilai rata-rata halaman yang dikunjungi
4. **Product Related**: Jumlah halaman produk yang dilihat
5. **Product Related Duration**: Durasi waktu di halaman produk (detik)
6. **Administrative**: Jumlah halaman administratif yang dilihat
7. **Administrative Duration**: Durasi waktu di halaman administratif (detik)

## 🚀 Instalasi dan Menjalankan Aplikasi

### Prasyarat

- Python 3.7 atau lebih tinggi
- pip (Python package installer)

### Langkah Instalasi

1. **Clone atau download repository ini**
   ```bash
   git clone <repository-url>
   cd shopper-app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Pastikan file model tersedia**
   
   Aplikasi membutuhkan file-file berikut di direktori root:
   - `model.pkl` - Model machine learning yang telah dilatih
   - `scaler.pkl` - Scaler untuk normalisasi data
   - `meta.pkl` - Metadata model (termasuk threshold)

4. **Jalankan aplikasi**
   ```bash
   python app.py
   ```

5. **Akses aplikasi**
   
   Buka browser dan kunjungi: `http://localhost:5000`

## 📁 Struktur Proyek

```
shopper-app/
├── app.py                          # Aplikasi Flask utama
├── requirements.txt                # Dependencies Python
├── model.pkl                       # Model machine learning
├── scaler.pkl                      # Data scaler
├── meta.pkl                        # Metadata model
├── templates/
│   └── index.html                  # Template HTML utama
├── static/
│   └── style.css                   # Stylesheet CSS
└── README.md                       
```

## 🎯 Cara Penggunaan

### 1. Prediksi Manual

1. Buka aplikasi di browser
2. Isi form dengan data sesi pengunjung:
   - Masukkan nilai untuk setiap parameter yang diminta
   - Pastikan nilai sesuai dengan rentang yang valid
3. Klik tombol "Prediksi Sekarang"
4. Lihat hasil prediksi dan probabilitas pembelian

### 2. Prediksi via Upload CSV

1. Siapkan file CSV dengan kolom-kolom berikut:
   ```
   BounceRates,Administrative_Duration,ProductRelated,ProductRelated_Duration,Administrative,ExitRates,PageValues
   ```

2. Upload file CSV melalui form upload
3. Sistem akan memproses semua baris data
4. Download file hasil yang berisi prediksi untuk setiap baris

### Contoh Format CSV

```csv
BounceRates,Administrative_Duration,ProductRelated,ProductRelated_Duration,Administrative,ExitRates,PageValues
0.02,80.0,1,0.0,1,0.02,0.0
0.00,64.0,2,2.666667,3,0.014286,0.0
0.20,0.0,1,0.0,1,0.20,0.0
```

## 📈 Output Prediksi

Aplikasi memberikan dua jenis output:

1. **Hasil Prediksi**: "Akan Membeli" atau "Tidak Membeli"
2. **Probabilitas**: Persentase kemungkinan pembelian (0-100%)