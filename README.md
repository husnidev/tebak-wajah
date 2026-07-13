Yang Akan Kita Bangun

Tahap demi tahap:

✅ Tahap 1

Struktur project
requirements.txt
config.py
Flask
Upload API

✅ Tahap 2

OpenCV
MediaPipe
Landmark Detection

✅ Tahap 3

Face Shape Classifier
Rule Engine

✅ Tahap 4

SQLite
History

✅ Tahap 5

REST API lengkap
Dokumentasi Swagger/OpenAPI
Unit Test

Struktur projek
1. .VENV
2. DATABASE
  - db.py
3. MODELS
4. RESULTS
5. STATIC 
  - CSS
  - JS
  - IMAGES
6. TEMPLATES
7. TMP
  - UPLOADS

apilkasi memiliki fitur diantaranya:
prediksi wajah yang lebih realistis dengan OpenCV/MediaPipe
tombol hapus riwayat
halaman detail hasil
API JSON yang lebih lengkap
dan tambahkan api model yang bisa memanmpilkan sifat manusia berdasarkan bentuk rupa wajah yang diupload.

setting environmet pada saat dijalankan di local dan vercel

import os
from pathlib import Path

if os.environ.get("VERCEL"):
    UPLOAD_FOLDER = Path("/tmp/uploads")
else:
    BASE_DIR = Path(__file__).resolve().parent
    UPLOAD_FOLDER = BASE_DIR / "uploads"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
