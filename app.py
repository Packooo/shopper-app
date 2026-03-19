"""
Aplikasi Prediksi Niat Pembelian Online
=======================================
Aplikasi web Flask yang menggunakan model machine learning (Random Forest)
untuk memprediksi apakah pengunjung website akan melakukan pembelian
berdasarkan data perilaku browsing mereka.

Fitur:
- Prediksi manual melalui form input (validasi WTForms)
- Prediksi massal melalui upload file CSV (validasi WTForms)
- Download hasil prediksi dalam format CSV
"""

# --- Impor library yang dibutuhkan ---
from flask import (
    Flask,  # Framework web utama
    render_template,  # Merender template HTML Jinja2
    request,  # Mengakses data dari HTTP request
    make_response,  # Membuat custom HTTP response
    redirect,  # Mengarahkan pengguna ke URL lain
    url_for,  # Menghasilkan URL dari nama fungsi
    session,  # Menyimpan data sementara per-pengguna
)
import joblib  # Memuat model dan scaler dari file (dibutuhkan oleh project ini)
import numpy as np  # Operasi numerik untuk array fitur
import pandas as pd  # Membaca dan memproses data CSV
import io  # Membuat stream di memori untuk output CSV
import uuid  # Generate ID unik untuk download CSV

# Impor form WTForms untuk validasi input
from forms import FormPrediksiManual, FormUploadCSV


def muat_model():
    """
    Memuat artefak model machine learning dari file.

    Mengembalikan:
        tuple: (model, scaler, threshold) jika berhasil,
               (None, None, None) jika file tidak ditemukan.
    """
    try:
        # Memuat model klasifikasi (Random Forest)
        model = joblib.load("model.pkl")
        # Memuat scaler untuk normalisasi fitur input
        scaler = joblib.load("scaler.pkl")
        # Memuat metadata yang berisi threshold optimal
        meta = joblib.load("meta.pkl")
        threshold = meta["threshold"]
        return model, scaler, threshold
    except FileNotFoundError as e:
        # Jika file model tidak ditemukan, aplikasi tetap jalan
        # tetapi semua endpoint prediksi akan mengembalikan error
        print(f"PERINGATAN: Tidak dapat memuat file model: {e}")
        return None, None, None


def buat_aplikasi():
    """
    Factory function untuk membuat instance aplikasi Flask.
    Memisahkan pembuatan app dari konfigurasi global agar
    lebih mudah di-test dan di-maintain.

    Mengembalikan:
        Flask: Instance aplikasi Flask yang sudah dikonfigurasi.
    """
    app = Flask(__name__)
    # Secret key untuk session (cookie-based)
    app.secret_key = "shopper-app-secret-key"
    # Nonaktifkan CSRF karena app ini tidak memiliki autentikasi/user accounts
    # CSRF protection hanya diperlukan untuk app dengan login/session user
    # Azure App Service proxy bisa mengganggu session cookies sehingga CSRF gagal 400
    app.config["WTF_CSRF_ENABLED"] = False
    # Batas ukuran file upload: 5MB
    app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
    return app


# --- Inisialisasi aplikasi dan model ---
app = buat_aplikasi()
model, scaler, threshold = muat_model()

# Penyimpanan sementara hasil CSV di memory (menghindari session cookie yang terbatas ~4KB)
_csv_download_store = {}


@app.errorhandler(413)
def file_terlalu_besar(e):
    """Menangani error saat file upload melebihi MAX_CONTENT_LENGTH (5MB)."""
    # Tidak bisa membuat WTForms di sini karena request.files akan re-raise 413.
    # Gunakan formdata=None agar Flask-WTF tidak mencoba membaca request data.
    from werkzeug.datastructures import MultiDict

    kosong = MultiDict()
    return render_template(
        "index.html",
        form_prediksi=FormPrediksiManual(formdata=kosong),
        form_upload=FormUploadCSV(formdata=kosong),
        csv_error="Error: Ukuran file melebihi batas maksimum (5MB)",
    ), 413


# --- Daftar kolom fitur yang digunakan model ---
# Urutan ini HARUS sama dengan urutan saat model dilatih
FEATURE_COLS = [
    "BounceRates",  # Rasio pengunjung yang langsung pergi (0-1)
    "Administrative_Duration",  # Durasi di halaman administratif (detik)
    "ProductRelated",  # Jumlah halaman produk yang dikunjungi
    "ProductRelated_Duration",  # Durasi di halaman produk (detik)
    "Administrative",  # Jumlah halaman administratif yang dikunjungi
    "ExitRates",  # Rasio keluar dari halaman terakhir (0-1)
    "PageValues",  # Nilai rata-rata halaman yang dikunjungi
]


def model_siap():
    """
    Memeriksa apakah semua komponen model sudah dimuat.

    Mengembalikan:
        bool: True jika model, scaler, dan threshold tersedia.
    """
    return all([model is not None, scaler is not None, threshold is not None])


def prediksi_dari_array(fitur_array):
    """
    Melakukan prediksi dari array fitur numerik.

    Parameter:
        fitur_array (np.ndarray): Array 2D berisi nilai fitur.

    Mengembalikan:
        tuple: (probabilitas, prediksi)
            - probabilitas (float): Probabilitas kelas positif (0-1)
            - prediksi (int): 1 = Akan Membeli, 0 = Tidak Membeli
    """
    # Normalisasi fitur menggunakan scaler yang sudah dilatih
    fitur_scaled = scaler.transform(fitur_array)
    # Hitung probabilitas kelas positif (kolom ke-1)
    probabilitas = model.predict_proba(fitur_scaled)[:, 1]
    # Terapkan threshold untuk menentukan keputusan akhir
    prediksi = [1 if p >= threshold else 0 for p in probabilitas]
    return probabilitas, prediksi


# ======================================
# ROUTE: Halaman Utama
# ======================================
@app.route("/")
def home():
    """Merender halaman utama dengan form WTForms kosong."""
    # Buat instance form untuk di-render di template
    form_prediksi = FormPrediksiManual()
    form_upload = FormUploadCSV()
    return render_template(
        "index.html", form_prediksi=form_prediksi, form_upload=form_upload
    )


# ======================================
# ROUTE: Prediksi Manual (Form Input)
# ======================================
@app.route("/predict", methods=["POST"])
def predict():
    """
    Menangani prediksi dari input form manual.
    WTForms memvalidasi semua field sebelum prediksi dilakukan.
    """
    # Buat instance form dengan data dari request
    form_prediksi = FormPrediksiManual()
    form_upload = FormUploadCSV()

    # Periksa apakah model sudah siap digunakan
    if not model_siap():
        return render_template(
            "index.html",
            form_prediksi=form_prediksi,
            form_upload=form_upload,
            error="Error: Model tidak siap. Periksa log server.",
        )

    # Validasi form menggunakan WTForms validators
    if not form_prediksi.validate_on_submit():
        # Kumpulkan semua pesan error dari setiap field
        error_messages = []
        for field_name, errors in form_prediksi.errors.items():
            # Abaikan CSRF token error
            if field_name == "csrf_token":
                continue
            for err in errors:
                error_messages.append(err)
        # Gabungkan pesan error menjadi satu string
        pesan_error = "; ".join(error_messages) if error_messages else "Validasi gagal"
        return render_template(
            "index.html",
            form_prediksi=form_prediksi,
            form_upload=form_upload,
            error=f"Error: {pesan_error}",
        )

    # Ambil nilai fitur dari form WTForms (sudah tervalidasi)
    fitur = [
        form_prediksi.BounceRates.data,
        form_prediksi.Administrative_Duration.data,
        form_prediksi.ProductRelated.data,
        form_prediksi.ProductRelated_Duration.data,
        form_prediksi.Administrative.data,
        form_prediksi.ExitRates.data,
        form_prediksi.PageValues.data,
    ]

    # Reshape menjadi array 2D (1 baris, 7 kolom)
    fitur_array = np.array(fitur).reshape(1, -1)

    # Lakukan prediksi
    probabilitas, prediksi = prediksi_dari_array(fitur_array)
    prob = probabilitas[0]
    pred = prediksi[0]

    # Tentukan label hasil berdasarkan prediksi
    hasil = "Akan Membeli" if pred == 1 else "Tidak Membeli"

    # Kembalikan halaman dengan hasil (form tetap terisi)
    return render_template(
        "index.html",
        form_prediksi=form_prediksi,
        form_upload=form_upload,
        result=hasil,
        prob=round(prob, 4),
    )


# ======================================
# ROUTE: Prediksi CSV (Upload File)
# ======================================
@app.route("/upload", methods=["POST"])
def predict_csv():
    """
    Menangani prediksi massal dari file CSV yang di-upload.
    WTForms memvalidasi file sebelum diproses.
    """
    # Buat instance form dengan data dari request
    form_prediksi = FormPrediksiManual()
    form_upload = FormUploadCSV()

    # Periksa apakah model sudah siap digunakan
    if not model_siap():
        return render_template(
            "index.html",
            form_prediksi=form_prediksi,
            form_upload=form_upload,
            csv_error="Error: Model tidak siap. Periksa log server.",
        )

    # Validasi form upload menggunakan WTForms validators
    if not form_upload.validate_on_submit():
        # Kumpulkan semua pesan error
        error_messages = []
        for field_name, errors in form_upload.errors.items():
            if field_name == "csrf_token":
                continue
            for err in errors:
                error_messages.append(err)
        pesan_error = "; ".join(error_messages) if error_messages else "Validasi gagal"
        return render_template(
            "index.html",
            form_prediksi=form_prediksi,
            form_upload=form_upload,
            csv_error=f"Error: {pesan_error}",
        )

    # Ambil file dari form WTForms (sudah tervalidasi)
    file = form_upload.file.data

    try:
        # Baca file CSV menjadi DataFrame
        df = pd.read_csv(file)

        # Validasi: CSV tidak boleh kosong (hanya header tanpa data)
        if df.empty:
            return render_template(
                "index.html",
                form_prediksi=form_prediksi,
                form_upload=form_upload,
                csv_error="Error: File CSV kosong (tidak ada baris data)",
            )

        # Validasi: periksa apakah semua kolom fitur ada di CSV
        kolom_hilang = [col for col in FEATURE_COLS if col not in df.columns]
        if kolom_hilang:
            return render_template(
                "index.html",
                form_prediksi=form_prediksi,
                form_upload=form_upload,
                csv_error=f"File CSV tidak memiliki kolom: {', '.join(kolom_hilang)}",
            )

        # Ambil kolom fitur dengan urutan yang benar
        fitur_df = df[FEATURE_COLS]

        # Validasi: semua kolom fitur harus berisi angka (numerik)
        for col in FEATURE_COLS:
            if not pd.to_numeric(fitur_df[col], errors="coerce").notna().all():
                return render_template(
                    "index.html",
                    form_prediksi=form_prediksi,
                    form_upload=form_upload,
                    csv_error=f"Error: Kolom '{col}' mengandung nilai non-numerik atau kosong",
                )

        # Konversi ke numerik (menangani string angka)
        fitur_df = fitur_df.apply(pd.to_numeric)

        # Lakukan prediksi untuk semua baris sekaligus
        probabilitas, prediksi = prediksi_dari_array(fitur_df)

        # Tambahkan kolom hasil ke DataFrame
        df["Hasil_Prediksi"] = [
            "Akan Membeli" if p == 1 else "Tidak Membeli" for p in prediksi
        ]
        # Format probabilitas sebagai persentase
        df["Probabilitas_Pembelian"] = [f"{p:.2%}" for p in probabilitas]

        # Siapkan data untuk ditampilkan di tabel HTML
        csv_results = df.to_dict("records")  # List of dicts per-baris
        csv_columns = df.columns.tolist()  # Nama kolom untuk header tabel

        # Simpan CSV hasil di memory server (bukan session cookie)
        download_id = str(uuid.uuid4())
        output_stream = io.StringIO()
        df.to_csv(output_stream, index=False, encoding="utf-8")
        _csv_download_store.clear()  # Bersihkan data lama
        _csv_download_store[download_id] = output_stream.getvalue()
        session["csv_download_id"] = download_id

        # Hitung ringkasan statistik prediksi
        ringkasan = {
            "total": len(prediksi),
            "beli": sum(prediksi),
            "tidak": len(prediksi) - sum(prediksi),
        }

        # Render halaman dengan tabel hasil
        return render_template(
            "index.html",
            form_prediksi=form_prediksi,
            form_upload=form_upload,
            csv_results=csv_results,
            csv_columns=csv_columns,
            csv_summary=ringkasan,
        )

    except Exception as e:
        # Tangani error parsing CSV atau error lainnya
        return render_template(
            "index.html",
            form_prediksi=form_prediksi,
            form_upload=form_upload,
            csv_error=f"Terjadi kesalahan: {e}",
        )


# ======================================
# ROUTE: Download Hasil CSV
# ======================================
@app.route("/download-csv")
def download_csv():
    """
    Mengirim file CSV hasil prediksi untuk di-download.
    Data CSV diambil dari memory server berdasarkan ID di session.
    """
    # Ambil ID download dari session, lalu ambil data dari memory
    download_id = session.get("csv_download_id")
    csv_data = _csv_download_store.get(download_id) if download_id else None

    # Jika tidak ada data (belum upload), redirect ke halaman utama
    if not csv_data:
        return redirect(url_for("home"))

    # Buat response dengan header download file
    response = make_response(csv_data)
    response.headers["Content-Disposition"] = "attachment; filename=hasil_prediksi.csv"
    response.headers["Content-type"] = "text/csv; charset=utf-8"
    return response


# ======================================
# Entry Point: Menjalankan Aplikasi
# ======================================
def jalankan_server():
    """Memulai server Flask di port 5000 dengan mode debug."""
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":  # pragma: no cover
    jalankan_server()
