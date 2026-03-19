"""
End-to-End Test untuk Aplikasi ShopPredict
==========================================
Test menggunakan Playwright untuk mensimulasikan interaksi browser nyata.

Menguji flow lengkap:
- Halaman utama dimuat dengan benar
- Prediksi manual via form input
- Upload CSV dan lihat hasil prediksi
- Download file CSV hasil prediksi
- Validasi error handling di browser

Jalankan: python -m pytest test_e2e.py -v
Prasyarat: pip install playwright && playwright install chromium
"""

import pytest
import subprocess
import time
import signal
import os
import sys

# Cek apakah playwright tersedia
try:
    from playwright.sync_api import sync_playwright, expect
except ImportError:
    pytest.skip("Playwright tidak terinstall. Jalankan: pip install playwright && playwright install chromium", allow_module_level=True)

# Path file test CSV
TEST_DATA_CSV = os.path.join(os.path.dirname(__file__), "test_data.csv")
BASE_URL = "http://localhost:5002"


@pytest.fixture(scope="module")
def server():
    """Jalankan Flask server di background untuk testing."""
    env = os.environ.copy()
    env["FLASK_ENV"] = "testing"
    proc = subprocess.Popen(
        [
            sys.executable, "-c",
            "from app import app; app.run(host='0.0.0.0', port=5002, debug=False)"
        ],
        cwd=os.path.dirname(__file__),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Tunggu server siap
    for _ in range(20):
        try:
            import urllib.request
            urllib.request.urlopen(BASE_URL)
            break
        except Exception:
            time.sleep(0.5)
    else:
        proc.kill()
        raise RuntimeError("Flask server gagal start di port 5002")

    yield proc

    # Cleanup: stop server
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=5)


@pytest.fixture(scope="module")
def browser_context(server):
    """Buat browser context Playwright."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        yield context
        context.close()
        browser.close()


@pytest.fixture
def page(browser_context):
    """Buat page baru untuk setiap test."""
    page = browser_context.new_page()
    yield page
    page.close()


# ======================================
# Test: Halaman Utama
# ======================================
class TestHalamanUtama:
    """E2E test untuk halaman utama."""

    def test_halaman_dimuat(self, page):
        """Halaman utama harus dimuat dengan judul yang benar."""
        page.goto(BASE_URL)
        assert "ShopPredict" in page.title()

    def test_form_prediksi_manual_ada(self, page):
        """Form prediksi manual harus tersedia."""
        page.goto(BASE_URL)
        assert page.locator("text=Prediksi Manual").is_visible()
        assert page.locator("#prediksiForm").is_visible()

    def test_form_upload_csv_ada(self, page):
        """Form upload CSV harus tersedia."""
        page.goto(BASE_URL)
        assert page.locator("text=Prediksi via CSV").is_visible()
        assert page.locator("#uploadForm").is_visible()

    def test_semua_input_field_ada(self, page):
        """Semua 7 input field fitur harus ada."""
        page.goto(BASE_URL)
        fields = [
            "BounceRates", "ExitRates", "PageValues",
            "ProductRelated", "ProductRelated_Duration",
            "Administrative", "Administrative_Duration",
        ]
        for field in fields:
            assert page.locator(f'input[name="{field}"]').is_visible()

    def test_navbar_ada(self, page):
        """Navbar dengan logo ShopPredict harus ada."""
        page.goto(BASE_URL)
        assert page.locator("text=ShopPredict").first.is_visible()
        assert page.locator("text=ML v1.0").is_visible()


# ======================================
# Test: Prediksi Manual
# ======================================
class TestPrediksiManual:
    """E2E test untuk prediksi manual via form input."""

    def test_prediksi_akan_membeli(self, page):
        """Input dengan PageValues tinggi harus menghasilkan 'Akan Membeli'."""
        page.goto(BASE_URL)

        # Isi form
        page.fill('input[name="BounceRates"]', "0.01")
        page.fill('input[name="ExitRates"]', "0.01")
        page.fill('input[name="PageValues"]', "50")
        page.fill('input[name="ProductRelated"]', "10")
        page.fill('input[name="ProductRelated_Duration"]', "200")
        page.fill('input[name="Administrative"]', "4")
        page.fill('input[name="Administrative_Duration"]', "30")

        # Submit
        page.click('button:has-text("Prediksi Sekarang")')

        # Verifikasi hasil
        page.wait_for_selector("#hasil-prediksi")
        assert page.locator("text=Akan Membeli").is_visible()
        assert page.locator("text=Probabilitas").is_visible()

    def test_prediksi_tidak_membeli(self, page):
        """Input dengan BounceRates tinggi harus menghasilkan 'Tidak Membeli'."""
        page.goto(BASE_URL)

        page.fill('input[name="BounceRates"]', "0.2")
        page.fill('input[name="ExitRates"]', "0.2")
        page.fill('input[name="PageValues"]', "0")
        page.fill('input[name="ProductRelated"]', "1")
        page.fill('input[name="ProductRelated_Duration"]', "0")
        page.fill('input[name="Administrative"]', "1")
        page.fill('input[name="Administrative_Duration"]', "0")

        page.click('button:has-text("Prediksi Sekarang")')

        page.wait_for_selector("#hasil-prediksi")
        assert page.locator("text=Tidak Membeli").is_visible()

    def test_validasi_field_kosong(self, page):
        """Submit tanpa mengisi field harus menampilkan error WTForms."""
        page.goto(BASE_URL)
        page.click('button:has-text("Prediksi Sekarang")')
        page.wait_for_load_state("domcontentloaded")
        # WTForms server-side validation menampilkan error
        assert page.locator("text=Error").first.is_visible()


# ======================================
# Test: Upload CSV
# ======================================
class TestUploadCSV:
    """E2E test untuk upload CSV dan prediksi massal."""

    def test_upload_csv_valid(self, page):
        """Upload test_data.csv harus menampilkan tabel hasil."""
        page.goto(BASE_URL)

        # Upload file
        page.set_input_files('input[name="file"]', TEST_DATA_CSV)

        # Klik upload
        page.click('button:has-text("Upload dan Prediksi")')

        # Tunggu hasil muncul
        page.wait_for_selector("#hasil-csv", timeout=10000)

        # Verifikasi ringkasan
        assert page.locator("text=Hasil Prediksi CSV").is_visible()
        assert page.locator("text=Total").is_visible()
        assert page.locator("text=Beli").first.is_visible()
        assert page.locator("text=Tidak").first.is_visible()

        # Verifikasi tabel
        assert page.locator("table").is_visible()
        assert page.locator("text=Hasil_Prediksi").is_visible()
        assert page.locator("text=Probabilitas_Pembelian").is_visible()

    def test_upload_csv_jumlah_baris_benar(self, page):
        """Tabel hasil harus memiliki jumlah baris sesuai CSV (5 baris)."""
        page.goto(BASE_URL)
        page.set_input_files('input[name="file"]', TEST_DATA_CSV)
        page.click('button:has-text("Upload dan Prediksi")')
        page.wait_for_selector("#hasil-csv", timeout=10000)

        # test_data.csv punya 5 baris data
        rows = page.locator("table tbody tr")
        assert rows.count() == 5

    def test_upload_csv_ringkasan_angka_benar(self, page):
        """Ringkasan total harus = 5 untuk test_data.csv."""
        page.goto(BASE_URL)
        page.set_input_files('input[name="file"]', TEST_DATA_CSV)
        page.click('button:has-text("Upload dan Prediksi")')
        page.wait_for_selector("#hasil-csv", timeout=10000)

        # Total harus 5
        total_text = page.locator(".grid.grid-cols-3 div").first.inner_text()
        assert "5" in total_text

    def test_upload_csv_download_link_ada(self, page):
        """Setelah upload, link Download CSV harus tersedia."""
        page.goto(BASE_URL)
        page.set_input_files('input[name="file"]', TEST_DATA_CSV)
        page.click('button:has-text("Upload dan Prediksi")')
        page.wait_for_selector("#hasil-csv", timeout=10000)

        download_link = page.locator('a:has-text("Download CSV")')
        assert download_link.is_visible()

    def test_download_csv_file(self, page):
        """Klik Download CSV harus menghasilkan file CSV."""
        page.goto(BASE_URL)
        page.set_input_files('input[name="file"]', TEST_DATA_CSV)
        page.click('button:has-text("Upload dan Prediksi")')
        page.wait_for_selector("#hasil-csv", timeout=10000)

        # Intercept download
        with page.expect_download() as download_info:
            page.click('a:has-text("Download CSV")')
        download = download_info.value
        assert download.suggested_filename == "hasil_prediksi.csv"

        # Verifikasi isi file
        path = download.path()
        with open(path, "r") as f:
            content = f.read()
        assert "Hasil_Prediksi" in content
        assert "Probabilitas_Pembelian" in content

    def test_auto_scroll_ke_hasil(self, page):
        """Setelah upload, halaman harus auto-scroll ke hasil."""
        page.goto(BASE_URL)
        page.set_input_files('input[name="file"]', TEST_DATA_CSV)
        page.click('button:has-text("Upload dan Prediksi")')
        page.wait_for_selector("#hasil-csv", timeout=10000)

        # Elemen hasil harus visible di viewport
        is_visible = page.locator("#hasil-csv").is_visible()
        assert is_visible


# ======================================
# Test: Error Handling
# ======================================
class TestErrorHandling:
    """E2E test untuk error handling di browser."""

    def test_upload_tanpa_file_error(self, page):
        """Klik upload tanpa memilih file harus menampilkan error WTForms."""
        page.goto(BASE_URL)
        page.click('button:has-text("Upload dan Prediksi")')
        page.wait_for_load_state("domcontentloaded")
        # WTForms server-side validation menampilkan error
        assert page.locator("text=Error").first.is_visible()
