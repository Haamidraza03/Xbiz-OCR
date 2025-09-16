# api.py
import json
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, abort
from werkzeug.utils import secure_filename
from paddleocr import PaddleOCR

# ----------------- Config -----------------
BASE_DIR = Path(__file__).parent.resolve()
IMAGES_DIR = BASE_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB upload limit (adjust if needed)

# PaddleOCR config
OCR_LANG = "en"   # change to 'ch', 'en|ch', etc. as required
USE_ANGLE_CLS = True  # whether to enable angle classifier

# ----------------- Helper utilities -----------------
def allowed_filename(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def extract_texts_from_predict(result):
    """
    Parse paddleocr.predict() output robustly and return a list of text lines.
    """
    texts = []

    def rec(obj):
        if isinstance(obj, str):
            if obj.strip(): texts.append(obj.strip())
            return
        if isinstance(obj, (int, float)): return
        if isinstance(obj, (list, tuple)):
            # Pattern: (text, confidence)
            if len(obj) == 2 and isinstance(obj[0], str) and isinstance(obj[1], (int, float)):
                if obj[0].strip(): texts.append(obj[0].strip())
                return
            # Pattern: [box, (text, conf)] or nested lists
            if len(obj) >= 2:
                sec = obj[1]
                if isinstance(sec, (list, tuple)):
                    # sec could be (text, conf) or list of (text,conf)
                    if len(sec) >= 1 and isinstance(sec[0], str):
                        if sec[0].strip(): texts.append(sec[0].strip())
                        return
                    else:
                        for item in sec:
                            rec(item)
                        return
            # fallback: walk children
            for item in obj:
                rec(item)
            return
        if isinstance(obj, dict):
            for v in obj.values():
                rec(v)
            return
        # otherwise ignore
        return

    rec(result)
    return [t for t in texts if t]

def ocr_image_to_txt(ocr, image_path: Path, out_path: Path = None):
    """
    Run ocr.predict(image_path) and save recognized text to out_path (defaults to image_path with .txt).
    If text extraction fails, save debug JSON dump to the file so you can inspect it.
    Returns the Path of the written file and the extracted text list.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Use predict() (recommended)
    result = ocr.predict(str(image_path))

    texts = extract_texts_from_predict(result)

    if out_path is None:
        out_path = image_path.with_suffix(".txt")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not texts:
        # Write a debug dump if nothing extracted
        debug_txt = (
            "# No text extracted by parser. Raw predict() output below (JSON-ish):\n\n"
            + json.dumps(result, default=str, ensure_ascii=False, indent=2)
        )
        out_path.write_text(debug_txt, encoding="utf-8")
        return out_path, []
    else:
        out_path.write_text("\n".join(texts), encoding="utf-8")
        return out_path, texts

# ----------------- Flask app -----------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Initialize PaddleOCR once when the app starts
print("Initializing PaddleOCR — this may download model files on first run...")
ocr = PaddleOCR(use_angle_cls=USE_ANGLE_CLS, lang=OCR_LANG)
print("PaddleOCR initialized.")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/upload", methods=["POST"])
def upload_file():
    """
    Accepts multipart form 'file'. Example using curl:
    curl -X POST -F "file=@./images/samp1.png" http://127.0.0.1:5000/upload
    """
    if "file" not in request.files:
        return jsonify({"error": "no file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "no selected file"}), 400

    filename = secure_filename(file.filename)
    if not allowed_filename(filename):
        return jsonify({"error": f"file extension not allowed. Allowed: {ALLOWED_EXTENSIONS}"}), 400

    saved_path = IMAGES_DIR / filename
    file.save(saved_path)

    try:
        out_txt_path, texts = ocr_image_to_txt(ocr, saved_path)
    except Exception as e:
        # on error, remove the saved image? (optional)
        # saved_path.unlink(missing_ok=True)  # Python 3.8+; or try/except
        return jsonify({"error": "ocr_failed", "detail": str(e)}), 500

    # Build response
    resp = {
        "image": str(saved_path),
        "txt_file": str(out_txt_path),
        "text_lines_count": len(texts),
        "text_preview": "\n".join(texts[:20])  # preview first 20 lines
    }
    # If no texts were extracted, indicate debug_dump
    if not texts:
        resp["warning"] = "No text extracted — debug dump saved in txt file."

    return jsonify(resp), 201

@app.route("/download/<path:filename>", methods=["GET"])
def download(filename):
    """
    Download a generated text file by name (filename should be like samp1.txt).
    """
    safe = secure_filename(filename)
    file_path = IMAGES_DIR / safe
    if not file_path.exists():
        return abort(404)
    return send_from_directory(directory=str(IMAGES_DIR), path=safe, as_attachment=True)

if __name__ == "__main__":
    # For development only. For production use waitress/gunicorn/uvicorn + WSGI.
    app.run(host="0.0.0.0", port=5000, debug=True)
