# ShopPredict — Prediksi Niat Pembelian Online

Aplikasi web berbasis Flask yang menggunakan model machine learning (Random Forest) untuk memprediksi niat pembelian pelanggan berdasarkan data interaksi pengguna di halaman web.

## Deskripsi

Aplikasi ini menganalisis perilaku pengunjung website untuk memprediksi kemungkinan mereka melakukan pembelian. Model telah dilatih dengan 12.330 data historis dari [UCI ML Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) dan menggunakan threshold optimal untuk klasifikasi.

## Fitur Utama

- **Prediksi Manual** — Input 7 parameter sesi pengunjung melalui form, hasil langsung ditampilkan di halaman
- **Prediksi Massal (CSV)** — Upload file CSV, hasil ditampilkan sebagai tabel dengan summary statistik
- **Download Hasil** — Unduh hasil prediksi CSV langsung dari halaman hasil
- **Validasi Form (WTForms)** — Validasi server-side dengan CSRF protection untuk semua form
- **Dark Mode UI** — Interface modern dengan Tailwind CSS dan design language Radix UI
- **Unit Test 100%** — 52 test cases dengan 100% code coverage

## Teknologi

| Komponen | Teknologi |
|----------|-----------|
| Backend | Flask 3.x (Python) |
| Form & Validasi | Flask-WTF, WTForms (CSRF + server-side validation) |
| Machine Learning | scikit-learn 1.8+ (Random Forest), joblib |
| Data Processing | pandas 3.x, numpy 2.x |
| Frontend | HTML5, Tailwind CSS (CDN), JavaScript |
| Icon | Font Awesome 6.x |
| Testing | pytest 9.x, coverage 7.x, unittest |
| Linting | black |

## Parameter Input

Aplikasi membutuhkan 7 parameter untuk melakukan prediksi:

| Parameter | Rentang | Deskripsi |
|-----------|---------|-----------|
| Bounce Rates | 0 - 1 | Rasio pengunjung yang langsung pergi |
| Exit Rates | 0 - 1 | Rasio keluar dari halaman terakhir |
| Page Values | >= 0 | Nilai rata-rata halaman yang dikunjungi |
| Product Related | >= 0 | Jumlah halaman produk yang dilihat |
| Product Related Duration | >= 0 | Durasi di halaman produk (detik) |
| Administrative | >= 0 | Jumlah halaman administratif yang dilihat |
| Administrative Duration | >= 0 | Durasi di halaman administratif (detik) |

## Instalasi

### Prasyarat

- Python 3.8+
- pip

### Langkah Instalasi

```bash
# 1. Clone repository
git clone https://github.com/Packooo/shopper-app.git
cd shopper-app

# 2. Buat virtual environment (opsional, disarankan)
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Jalankan aplikasi
python app.py
```

Buka browser: **http://localhost:5000**

### File Model

Aplikasi membutuhkan 3 file model di direktori root:

| File | Deskripsi |
|------|-----------|
| `model.pkl` | Model Random Forest yang telah dilatih |
| `scaler.pkl` | MinMaxScaler untuk normalisasi fitur |
| `meta.pkl` | Metadata berisi threshold optimal |

## Struktur Proyek

```
shopper-app/
├── app.py                 # Aplikasi Flask utama (refactored, berkomentar)
├── forms.py               # Definisi form WTForms (validasi input)
├── test_app.py            # Unit test (52 test, 100% coverage)
├── requirements.txt       # Dependencies Python
├── model.pkl              # Model machine learning
├── scaler.pkl             # Data scaler
├── meta.pkl               # Metadata model (threshold)
├── templates/
│   └── index.html         # Template HTML (Tailwind CSS + dark mode)
├── static/
│   └── style.css          # Stylesheet (legacy, UI via Tailwind CDN)
└── README.md
```

## Cara Penggunaan

### 1. Prediksi Manual

1. Buka aplikasi di browser
2. Isi form dengan data sesi pengunjung
3. Klik **"Prediksi Sekarang"**
4. Lihat hasil: "Akan Membeli" atau "Tidak Membeli" beserta probabilitas

### 2. Prediksi via Upload CSV

1. Siapkan file CSV dengan kolom berikut:
   ```csv
   BounceRates,Administrative_Duration,ProductRelated,ProductRelated_Duration,Administrative,ExitRates,PageValues
   0.02,80.0,1,0.0,1,0.02,0.0
   0.00,64.0,2,2.666667,3,0.014286,0.0
   0.20,0.0,1,0.0,1,0.20,0.0
   ```
2. Upload file melalui form **"Prediksi via CSV"**
3. Hasil ditampilkan langsung di halaman:
   - Summary cards (Total, Akan Membeli, Tidak Membeli)
   - Tabel detail per-baris
4. Klik **"Download CSV"** untuk mengunduh hasil

## Output Prediksi

| Output | Deskripsi |
|--------|-----------|
| Hasil Prediksi | "Akan Membeli" atau "Tidak Membeli" |
| Probabilitas | Persentase kemungkinan pembelian (0-100%) |

## Testing

```bash
# Jalankan semua test
python -m pytest test_app.py -v

# Jalankan dengan coverage report
python -m coverage run --source=app,forms -m pytest test_app.py
python -m coverage report -m
```

**Hasil saat ini:** 52 test passed, 100% coverage (app.py + forms.py)

## Linting

```bash
# Format kode dengan black
black app.py forms.py test_app.py

# Cek tanpa mengubah file
black --check app.py forms.py test_app.py
```

## Arsitektur Kode

### `app.py` — Aplikasi Flask utama

| Fungsi | Deskripsi |
|--------|-----------|
| `muat_model()` | Memuat model, scaler, dan threshold dari file |
| `buat_aplikasi()` | Factory function untuk membuat instance Flask |
| `model_siap()` | Memeriksa ketersediaan semua komponen model |
| `prediksi_dari_array()` | Logika prediksi: scaling → predict_proba → threshold |
| `home()` | GET / — Halaman utama |
| `predict()` | POST /predict — Prediksi manual dari form |
| `predict_csv()` | POST /upload — Prediksi massal dari CSV |
| `download_csv()` | GET /download-csv — Download hasil CSV |
| `jalankan_server()` | Entry point untuk menjalankan server |

### `forms.py` — Definisi form WTForms

| Class | Deskripsi |
|-------|-----------|
| `FormPrediksiManual` | Form 7 field numerik dengan validasi rentang |
| `FormUploadCSV` | Form upload file dengan validasi ekstensi .csv |

## Dataset

| Atribut | Detail |
|---------|--------|
| Nama | Online Shoppers Purchasing Intention Dataset |
| Sumber | [UCI ML Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) |
| Jumlah Data | 12.330 sesi pengunjung |
| Referensi | Sakar, C.O., Polat, S.O., Katircioglu, M. et al. (2019) |

## Lisensi

Lihat repository asli untuk informasi lisensi.
