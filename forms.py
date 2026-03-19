"""
Definisi Form WTForms untuk Aplikasi ShopPredict
=================================================
Menggunakan Flask-WTF untuk validasi input di sisi server.
Setiap field memiliki validator yang sesuai dengan
kebutuhan model machine learning.
"""

from flask_wtf import FlaskForm  # Base class form dengan CSRF protection
from flask_wtf.file import FileField, FileRequired, FileAllowed  # Validasi file upload
from wtforms import FloatField, SubmitField  # Tipe field untuk input numerik
from wtforms.validators import (
    DataRequired,  # Field wajib diisi
    NumberRange,  # Validasi rentang angka (min/max)
    InputRequired,  # Memastikan input tidak kosong
)


class FormPrediksiManual(FlaskForm):
    """
    Form untuk prediksi manual melalui input 7 parameter fitur.

    Setiap field memiliki validator:
    - DataRequired: wajib diisi
    - NumberRange: membatasi rentang nilai sesuai domain
    """

    # Rasio pengunjung yang langsung pergi dari halaman (0-1)
    # Menggunakan InputRequired (bukan DataRequired) karena nilai 0 valid
    BounceRates = FloatField(
        "Bounce Rates",
        validators=[
            InputRequired(message="Bounce Rates wajib diisi"),
            NumberRange(min=0, max=1, message="Bounce Rates harus antara 0 dan 1"),
        ],
    )

    # Rasio keluar dari halaman terakhir yang dikunjungi (0-1)
    # Menggunakan InputRequired (bukan DataRequired) karena nilai 0 valid
    ExitRates = FloatField(
        "Exit Rates",
        validators=[
            InputRequired(message="Exit Rates wajib diisi"),
            NumberRange(min=0, max=1, message="Exit Rates harus antara 0 dan 1"),
        ],
    )

    # Nilai rata-rata halaman yang dikunjungi sebelum transaksi
    PageValues = FloatField(
        "Page Values",
        validators=[
            InputRequired(message="Page Values wajib diisi"),
            NumberRange(min=0, message="Page Values tidak boleh negatif"),
        ],
    )

    # Jumlah halaman produk yang dikunjungi dalam sesi
    ProductRelated = FloatField(
        "Halaman Produk",
        validators=[
            InputRequired(message="Halaman Produk wajib diisi"),
            NumberRange(min=0, message="Halaman Produk tidak boleh negatif"),
        ],
    )

    # Total durasi waktu di halaman produk (dalam detik)
    ProductRelated_Duration = FloatField(
        "Durasi Produk (detik)",
        validators=[
            InputRequired(message="Durasi Produk wajib diisi"),
            NumberRange(min=0, message="Durasi Produk tidak boleh negatif"),
        ],
    )

    # Jumlah halaman administratif yang dikunjungi dalam sesi
    Administrative = FloatField(
        "Halaman Admin",
        validators=[
            InputRequired(message="Halaman Admin wajib diisi"),
            NumberRange(min=0, message="Halaman Admin tidak boleh negatif"),
        ],
    )

    # Total durasi waktu di halaman administratif (dalam detik)
    Administrative_Duration = FloatField(
        "Durasi Admin (detik)",
        validators=[
            InputRequired(message="Durasi Admin wajib diisi"),
            NumberRange(min=0, message="Durasi Admin tidak boleh negatif"),
        ],
    )

    # Tombol submit form
    submit = SubmitField("Prediksi Sekarang")


class FormUploadCSV(FlaskForm):
    """
    Form untuk upload file CSV prediksi massal.

    Validasi:
    - FileRequired: file wajib dipilih
    - FileAllowed: hanya menerima file berekstensi .csv
    """

    # Input file CSV dengan validasi ekstensi
    file = FileField(
        "File CSV",
        validators=[
            FileRequired(message="File CSV wajib dipilih"),
            FileAllowed(["csv"], message="Hanya file CSV yang diperbolehkan"),
        ],
    )

    # Tombol submit upload
    submit = SubmitField("Upload dan Prediksi")
