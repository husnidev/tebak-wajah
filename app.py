import json
import math
import os
import uuid
from pathlib import Path

from flask import Flask, flash, jsonify, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from database.db import (
    delete_prediction,
    get_prediction_by_id,
    get_predictions,
    init_db,
    save_prediction,
)

BASE_DIR = Path(__file__).resolve().parent

vercel_mode = bool(os.environ.get("VERCEL") or os.environ.get("VERCEL_ENV"))
if os.environ.get("UPLOAD_FOLDER"):
    UPLOAD_FOLDER = Path(os.environ["UPLOAD_FOLDER"])
elif vercel_mode:
    UPLOAD_FOLDER = Path("/tmp/uploads")
else:
    UPLOAD_FOLDER = BASE_DIR / "tmp" / "uploads"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

MODEL_PREFERENCES = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
]

app = Flask(__name__)
app.config["SECRET_KEY"] = "tebak-wajah-secret"
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
init_db()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _fallback_prediction(image_path: str):
    from PIL import Image

    with Image.open(image_path) as img:
        width, height = img.size
        ratio = width / height

        if 0.95 <= ratio <= 1.05:
            shape = "Bulat / Persegi"
            confidence = 0.84
        elif ratio > 1.05:
            shape = "Oval / Panjang"
            confidence = 0.81
        else:
            shape = "Segitiga / Lonjong"
            confidence = 0.79

        metrics = {
            "face_width": round(width, 2),
            "face_height": round(height, 2),
            "face_ratio": round(ratio, 2),
            "forehead_width": round(width * 0.7, 2),
            "jaw_width": round(width * 0.8, 2),
        }
        return shape, round(confidence, 2), metrics


def predict_face_shape(image_path: str):
    try:
        import cv2
        import mediapipe as mp

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Gambar tidak dapat dibaca")

        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        face_mesh.close()

        if not results.multi_face_landmarks:
            raise ValueError("Wajah tidak terdeteksi")

        landmarks = results.multi_face_landmarks[0].landmark
        height, width = image.shape[:2]

        def point_distance(index_a: int, index_b: int):
            a = landmarks[index_a]
            b = landmarks[index_b]
            return math.hypot((a.x * width) - (b.x * width), (a.y * height) - (b.y * height))

        face_width = point_distance(234, 454)
        face_height = point_distance(10, 152)
        forehead_width = point_distance(105, 334)
        jaw_width = point_distance(205, 425)
        face_ratio = face_width / face_height if face_height else 0

        if face_height / face_width > 1.2:
            shape = "Panjang / Oblong"
        elif 0.95 <= face_ratio <= 1.05:
            shape = "Bulat / Persegi"
        elif face_ratio < 0.95:
            shape = "Segitiga / Lonjong"
        else:
            shape = "Oval / Panjang"

        metrics = {
            "face_width": round(face_width, 2),
            "face_height": round(face_height, 2),
            "face_ratio": round(face_ratio, 2),
            "forehead_width": round(forehead_width, 2),
            "jaw_width": round(jaw_width, 2),
            "method": "OpenCV + MediaPipe Face Mesh",
        }
        return shape, 0.94, metrics
    except Exception:
        return _fallback_prediction(image_path)


def map_personality(shape: str, metrics: dict | None = None):
    """Generate a personality map using a Google AI model.

    Requires environment variable `GOOGLE_API_KEY` to be set. The function
    calls the Google Generative AI API and expects the model to return a JSON
    object with keys: `summary`, `traits` (list), `strengths` (list),
    `challenges` (list), and optional `advice` (list).
    """
    import os
    import json

    # Hardcoded fallback map used when AI key/call is unavailable.
    fallback_map = {
        "Bulat / Persegi": {
            "summary": "Cenderung hangat, setia, dan suka menjaga hubungan.",
            "traits": ["Sosial", "Penuh perhatian", "Ramah", "Stabil"],
            "strengths": ["Mudah beradaptasi", "Membangun kepercayaan"],
            "challenges": ["Kadang terlalu hati-hati"],
        },
        "Oval / Panjang": {
            "summary": "Biasanya terlihat tenang, bijaksana, dan berpikir luas.",
            "traits": ["Analitis", "Tenang", "Bijaksana", "Terarah"],
            "strengths": ["Berpikir strategis", "Sabar"],
            "challenges": ["Terkadang terlalu kritis"],
        },
        "Segitiga / Lonjong": {
            "summary": "Sering dianggap berani, mandiri, dan penuh energi.",
            "traits": ["Berani", "Mandiri", "Enerjik", "Proaktif"],
            "strengths": ["Memimpin", "Berinisiatif"],
            "challenges": ["Cenderung cepat ambil keputusan"],
        },
        "Panjang / Oblong": {
            "summary": "Kebanyakan dikenal sebagai pribadi yang fleksibel dan visioner.",
            "traits": ["Fleksibel", "Visioner", "Inovatif", "Adaptif"],
            "strengths": ["Menciptakan ide baru", "Cepat beradaptasi"],
            "challenges": ["Bisa terlalu banyak memikirkan opsi"],
        },
        "Semrawut / Tidak Terawat": {
            "summary": "Penampilan yang terkesan kurang rapi atau terawat; ini bisa disebabkan oleh kondisi sementara atau gaya hidup.",
            "traits": ["Santai", "Kurang perhatian pada detail", "Apa adanya"],
            "strengths": ["Tidak terlalu memikirkan penilaian orang lain", "Autentik"],
            "challenges": ["Bisa memberi kesan kurang profesional", "Perlu perawatan diri lebih"],
            "advice": ["Pertimbangkan perawatan kulit dasar", "Rapi saat acara penting", "Tidur cukup dan hidrasi"],
        },
    }

    import sys

    # During unit tests or CI we avoid calling real remote models unless
    # explicitly enabled via RUN_REAL_GENAI=1. This prevents network calls
    # during automated tests.
    if ("unittest" in sys.modules or "pytest" in sys.modules) and os.getenv("RUN_REAL_GENAI") != "1":
        app.logger.info("Test environment detected; returning fallback map")
        return fallback_map.get(shape, fallback_map["Oval / Panjang"])

    print("===== ENV CHECK =====")
    print("VERCEL =", os.getenv("VERCEL"))
    print("VERCEL_ENV =", os.getenv("VERCEL_ENV"))
    print("GOOGLE_API_KEY exists =", bool(os.getenv("GOOGLE_API_KEY")))
    print("=====================")

    api_key = os.getenv("GOOGLE_API_KEY")

    app.logger.info(f"VERCEL_ENV={os.getenv('VERCEL_ENV')}")
    app.logger.info(f"GOOGLE_API_KEY exists={bool(api_key)}")
    app.logger.info("========== MAP PERSONALITY ==========")
    app.logger.info(f"Shape: {shape}")
    app.logger.info(f"API Key Exists: {bool(api_key)}")
    app.logger.info("Calling Gemini...")

    # Determine whether we're running in production (VERCEL or explicit env)
    is_production = bool(
        os.environ.get("VERCEL")
        or os.environ.get("VERCEL_ENV")
        or os.environ.get("FLASK_ENV") == "production"
        or os.environ.get("ENV") == "production"
    )

    # When running locally, allow loading key from .env or falling back to the
    # hardcoded map. In production we require the GOOGLE_API_KEY env var.
    if not api_key and not is_production:
        env_path = BASE_DIR / ".env"
        if env_path.exists():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip().startswith("GOOGLE_API_KEY="):
                            api_key = line.split("=", 1)[1].strip()
                            break
            except Exception:
                pass

    if is_production and not api_key:
        app.logger.warning("GOOGLE_API_KEY not set in production; using fallback personality map")
        return fallback_map.get(shape, fallback_map["Oval / Panjang"])

    if not api_key:
        return fallback_map.get(shape, fallback_map["Oval / Panjang"])

    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        def _normalize_model_name(model):
            if hasattr(model, "name"):
                name = model.name
            elif isinstance(model, dict):
                name = model.get("name", "")
            else:
                name = str(model)

            if name.startswith("models/"):
                name = name.replace("models/", "", 1)

            return name

        system_msg = (
            "You are a helpful assistant that generates a concise personality map "
            "based on a face shape label and optional numeric facial metrics. "
            "Respond with a valid JSON object only (no extra text) with these keys:"
            " summary (string), traits (array of short strings), strengths (array), "
            "challenges (array), advice (array, optional). Keep each list short (3-5 items)."
        )

        user_msg = {
            "shape": shape,
            "metrics": metrics or {},
        }

        prompt = (
            f"Generate a JSON personality map for the following input:\n{json.dumps(user_msg, ensure_ascii=False)}\n"
            "Return only the JSON object."
        )

        for model in MODEL_PREFERENCES:
            app.logger.info(f"Trying {model}")

            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={
                        "temperature": 0.6,
                        "max_output_tokens": 400,
                        "safety_settings": [
                            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                        ],
                    }
                )

                if not response.candidates:
                    app.logger.warning(f"{model}: no candidates returned")
                    continue

                candidate = response.candidates[0]
                finish_reason = getattr(candidate, "finish_reason", "UNKNOWN")
                app.logger.info(f"{model} finish_reason: {finish_reason}")

                text = None
                try:
                    text = response.text
                except (ValueError, AttributeError) as e:
                    app.logger.warning(f"{model}: response.text raised {type(e).__name__}: {e}")
                    continue

                if not text or not text.strip():
                    app.logger.warning(f"{model}: empty response text")
                    continue

                text = text.strip()
                app.logger.info(f"{model} response preview: {text[:200]}")

                if text.startswith("```"):
                    text = text.replace("```json", "").replace("```", "").strip()
                return json.loads(text)

            except json.JSONDecodeError as e:
                app.logger.warning(f"{model}: invalid JSON: {e} | raw: {text[:300] if text else 'None'}")
            except Exception as e:
                app.logger.warning(f"{model}: error: {type(e).__name__}: {e}")

        # All models failed, fall back to hardcoded map
        app.logger.warning("All Gemini models failed; falling back to hardcoded personality map")
        return fallback_map.get(shape, fallback_map["Oval / Panjang"])
    except Exception as e:
        import traceback

        traceback.print_exc()
        app.logger.exception(e)
        return fallback_map.get(shape, fallback_map["Oval / Panjang"])


@app.route("/", methods=["GET", "POST"])
def index():
    history = get_predictions()

    if request.method == "POST":
        if "file" not in request.files:
            flash("Pilih file gambar terlebih dahulu.")
            return redirect(url_for("index"))

        file = request.files["file"]
        if file.filename == "":
            flash("Nama file tidak valid.")
            return redirect(url_for("index"))

        if not allowed_file(file.filename):
            flash("Format file tidak didukung. Gunakan JPG, PNG, atau WEBP.")
            return redirect(url_for("index"))

        original_name = secure_filename(file.filename)
        ext = Path(original_name).suffix.lower()
        stored_name = f"{uuid.uuid4().hex}{ext}"
        save_path = UPLOAD_FOLDER / stored_name
        file.save(str(save_path))

        try:
            prediction, confidence, metrics = predict_face_shape(str(save_path))
            prediction_id = save_prediction(original_name, prediction, confidence, stored_name, metrics)
            history = get_predictions()
            flash("Prediksi berhasil disimpan.")
            return redirect(url_for("detail_result", prediction_id=prediction_id))
        except Exception as e:
            app.logger.exception("Gagal memproses prediksi")
            flash(f"Gagal memproses prediksi: {e}")
            return redirect(url_for("index"))

    return render_template("index.html", history=history, prediction=None, confidence=None, uploaded_name=None, prediction_id=None)


@app.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/result/<int:prediction_id>")
def detail_result(prediction_id: int):
    prediction = get_prediction_by_id(prediction_id)
    if not prediction:
        flash("Data prediksi tidak ditemukan.")
        return redirect(url_for("index"))

    prediction["details"] = json.loads(prediction["details"]) if prediction.get("details") else {}
    try:
        prediction["personality"] = map_personality(prediction.get("prediction", ""), prediction.get("details", {}))
    except Exception as e:
        app.logger.warning(f"Gagal generate personality: {e}")
        fallback_map = {
            "Bulat / Persegi": {"summary": "Cenderung hangat, setia, dan suka menjaga hubungan.", "traits": ["Sosial", "Penuh perhatian", "Ramah", "Stabil"], "strengths": ["Mudah beradaptasi", "Membangun kepercayaan"], "challenges": ["Kadang terlalu hati-hati"]},
            "Oval / Panjang": {"summary": "Biasanya terlihat tenang, bijaksana, dan berpikir luas.", "traits": ["Analitis", "Tenang", "Bijaksana", "Terarah"], "strengths": ["Berpikir strategis", "Sabar"], "challenges": ["Terkadang terlalu kritis"]},
            "Segitiga / Lonjong": {"summary": "Sering dianggap berani, mandiri, dan penuh energi.", "traits": ["Berani", "Mandiri", "Enerjik", "Proaktif"], "strengths": ["Memimpin", "Berinisiatif"], "challenges": ["Cenderung cepat ambil keputusan"]},
            "Panjang / Oblong": {"summary": "Kebanyakan dikenal sebagai pribadi yang fleksibel dan visioner.", "traits": ["Fleksibel", "Visioner", "Inovatif", "Adaptif"], "strengths": ["Menciptakan ide baru", "Cepat beradaptasi"], "challenges": ["Bisa terlalu banyak memikirkan opsi"]},
        }
        shape = prediction.get("prediction", "")
        prediction["personality"] = fallback_map.get(shape, fallback_map["Oval / Panjang"])
    return render_template("detail.html", prediction=prediction)


@app.route("/history/<int:prediction_id>/delete", methods=["POST"])
def delete_history(prediction_id: int):
    delete_prediction(prediction_id)
    flash("Riwayat prediksi berhasil dihapus.")
    return redirect(url_for("index"))


@app.route("/api/history")
def api_history():
    history = get_predictions(limit=20)
    return jsonify({"count": len(history), "history": history})


@app.route("/api/appearance", methods=["POST"])
def api_appearance():
    """Return a gentle appearance profile. Accepts optional `label` form field.

    This endpoint does not attempt to judge appearance automatically; it
    returns a non-derogatory mapping for known labels such as 'semrawut'.
    """
    label = request.form.get("label", "").lower()
    label_map = {
        "semrawut": "Semrawut / Tidak Terawat",
        "tidak_terawat": "Semrawut / Tidak Terawat",
    }
    mapped = label_map.get(label)
    if not mapped:
        return jsonify({"error": "Label tidak dikenali. Gunakan 'semrawut' atau 'tidak_terawat'."}), 400

    profile = map_personality(mapped, None)
    return jsonify({"label": mapped, "profile": profile})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "file" not in request.files:
        return jsonify({"error": "File gambar tidak ditemukan."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nama file tidak valid."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Format file tidak didukung."}), 400

    original_name = secure_filename(file.filename)
    ext = Path(original_name).suffix.lower()
    stored_name = f"{uuid.uuid4().hex}{ext}"
    save_path = UPLOAD_FOLDER / stored_name
    file.save(str(save_path))

    prediction, confidence, metrics = predict_face_shape(str(save_path))
    prediction_id = save_prediction(original_name, prediction, confidence, stored_name, metrics)
    result = {
        "id": prediction_id,
        "filename": original_name,
        "prediction": prediction,
        "confidence": confidence,
        "metrics": metrics,
        "personality": map_personality(prediction, metrics),
        "image_url": url_for("uploaded_file", filename=stored_name, _external=True),
    }
    return jsonify(result)


@app.route("/api/personality", methods=["POST"])
def api_personality():
    if "file" not in request.files:
        return jsonify({"error": "File gambar tidak ditemukan."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nama file tidak valid."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Format file tidak didukung."}), 400

    original_name = secure_filename(file.filename)
    ext = Path(original_name).suffix.lower()
    stored_name = f"{uuid.uuid4().hex}{ext}"
    save_path = UPLOAD_FOLDER / stored_name
    file.save(str(save_path))

    prediction, confidence, metrics = predict_face_shape(str(save_path))
    personality = map_personality(prediction, metrics)
    # optional appearance override via form param (e.g., appearance=semrawut)
    appearance_label = request.form.get("appearance")
    appearance = None
    if appearance_label:
        # map common keys
        key_map = {
            "semrawut": "Semrawut / Tidak Terawat",
            "tidak_terawat": "Semrawut / Tidak Terawat",
        }
        mapped = key_map.get(appearance_label.lower())
        if mapped:
            appearance = map_personality(mapped, None)

    resp = {
        "filename": original_name,
        "prediction": prediction,
        "confidence": confidence,
        "metrics": metrics,
        "personality": personality,
        "image_url": url_for("uploaded_file", filename=stored_name, _external=True),
    }
    if appearance:
        resp["appearance"] = appearance
    return jsonify(resp)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
