# ocr_api.py
"""
OCR multi-backend Flask API with improved preprocessing + server-side comparison.txt writer.

- Runs Tesseract (pytesseract), EasyOCR, and PaddleOCR on the same preprocessed image
- Returns JSON with Input_Image_Base64 and the three OCR responses
- Writes a comparison text file under ./images2/comparison_<imagename>.txt

Requirements (pip):
  pip install flask pillow numpy opencv-python pytesseract easyocr paddlepaddle paddleocr

Note: Tesseract must be installed on the system (Windows: install .exe and set path).
"""
import io
import time
import base64
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import cv2

# OCR libs
import pytesseract
from pytesseract import Output
import easyocr
from paddleocr import PaddleOCR

# If Tesseract isn't in PATH on Windows, set explicit path:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ----------------- Config -----------------
BASE_DIR = Path(__file__).parent.resolve()
IMAGES_DIR = BASE_DIR / "images2"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}

# Paddle / EasyOCR language (adjust as required)
LANGS = ["en"]
PADDLE_LANG = "en"

# Initialize OCR engines once (may download models on first init)
EASY_OCR_READER = easyocr.Reader(LANGS, gpu=False)
PADDLE_OCR = PaddleOCR(use_angle_cls=True, lang=PADDLE_LANG)

app = Flask(__name__)


# ----------------- Preprocessing helpers -----------------
def pil_to_cv2(img: Image.Image) -> np.ndarray:
    """Convert PIL image to OpenCV BGR."""
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv2_to_pil(img: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR to PIL Image."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def image_to_base64(img: Image.Image) -> str:
    """Encode PIL Image to base64 PNG string."""
    buff = io.BytesIO()
    img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("ascii")


def preprocess_image(pil_img: Image.Image, target_height: int = 1200) -> Image.Image:
    """
    Advanced preprocessing pipeline tuned to help OCR on low-quality images.
    Steps:
      - convert to RGB
      - scale up to target height (preserve aspect ratio) to improve OCR resolution
      - convert to grayscale
      - apply bilateral filter (edge-preserving denoise)
      - apply CLAHE (contrast enhancement)
      - denoise with fastNlMeansDenoising
      - adaptive thresholding (binarization)
      - morphological opening to remove small noise
    Returns a PIL Image (RGB) suitable for OCR.
    """
    img_cv = pil_to_cv2(pil_img)
    h, w = img_cv.shape[:2]

    # Upscale if smaller than target height (but avoid huge upscales)
    if h < target_height and h > 0:
        scale = target_height / h
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_cv = cv2.resize(img_cv, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Edge-preserving denoise
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # CLAHE (adaptive histogram equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Fast Non-local Means Denoising
    gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # Median blur to remove salt-and-pepper noise
    gray = cv2.medianBlur(gray, 3)

    # Adaptive thresholding (binarization)
    block_size = 15 if max(gray.shape) < 2000 else 31
    gray_bin = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 9
    )

    # Morphological opening to remove small dots
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    gray_bin = cv2.morphologyEx(gray_bin, cv2.MORPH_OPEN, kernel)

    processed_bgr = cv2.cvtColor(gray_bin, cv2.COLOR_GRAY2BGR)
    return cv2_to_pil(processed_bgr)


# ----------------- OCR wrappers -----------------
def run_tesseract(pil_img: Image.Image) -> Dict[str, Any]:
    """Run pytesseract and return structured output."""
    start = time.time()
    data = pytesseract.image_to_data(
        pil_img, output_type=Output.DICT, config="--oem 3 --psm 6"
    )
    duration = time.time() - start

    entries = []
    n = len(data.get("level", []))
    for i in range(n):
        text = data["text"][i].strip()
        if not text:
            continue
        obj = {
            "text": text,
            "left": int(data["left"][i]),
            "top": int(data["top"][i]),
            "width": int(data["width"][i]),
            "height": int(data["height"][i]),
            "conf": float(data["conf"][i]) if data["conf"][i] != "-1" else None,
        }
        entries.append(obj)

    full_text = pytesseract.image_to_string(pil_img, config="--oem 3 --psm 6")
    return {"engine": "tesseract", "duration_sec": duration, "full_text": full_text, "words": entries}


def run_easyocr(pil_img: Image.Image) -> Dict[str, Any]:
    """Run EasyOCR and return structured output."""
    start = time.time()
    arr = np.array(pil_img.convert("RGB"))
    results = EASY_OCR_READER.readtext(arr)
    duration = time.time() - start

    entries = []
    for bbox, text, conf in results:
        bbox_list = [[int(x[0]), int(x[1])] for x in bbox]
        entries.append({"text": text, "conf": float(conf), "bbox": bbox_list})

    full_text = "\n".join([r[1] for r in results])
    return {"engine": "easyocr", "duration_sec": duration, "full_text": full_text, "results": entries}


def extract_texts_from_predict(result) -> List[str]:
    """
    Robust extractor for various nested shapes returned by PaddleOCR.predict().
    Returns list of text strings in detection order.
    """
    texts: List[str] = []

    def rec(obj):
        # If it's a string, treat as text
        if isinstance(obj, str):
            if obj.strip():
                texts.append(obj.strip())
            return

        # If numeric, skip
        if isinstance(obj, (int, float)):
            return

        # If tuple/list, try some likely patterns first
        if isinstance(obj, (list, tuple)):
            # Pattern: (text, confidence)  -> (str, float)
            if len(obj) == 2 and isinstance(obj[0], str) and isinstance(obj[1], (int, float)):
                if obj[0].strip():
                    texts.append(obj[0].strip())
                return

            # Pattern: [box, (text, conf)] or [box, [ (text,conf), ... ]]
            if len(obj) >= 2:
                sec = obj[1]
                if isinstance(sec, (list, tuple)) and len(sec) >= 1:
                    # sec may be (text, conf) or [ (text, conf), ... ]
                    if isinstance(sec[0], str):
                        if sec[0].strip():
                            texts.append(sec[0].strip())
                        return
                    else:
                        # iterate inside sec
                        for item in sec:
                            rec(item)
                        return

            # Otherwise, recursively walk children
            for item in obj:
                rec(item)
            return

        # If dict, inspect values
        if isinstance(obj, dict):
            for v in obj.values():
                rec(v)
            return

        # Anything else - ignore
        return

    rec(result)
    # keep unique-ish but preserve order (not removing duplicates aggressively)
    cleaned = [t for t in texts if t]
    return cleaned


def run_paddleocr(pil_img: Image.Image) -> Dict[str, Any]:
    """
    Run PaddleOCR.predict on a numpy BGR image and parse output.
    (This uses PADDLE_OCR.predict which some paddle versions support.)
    """
    start = time.time()
    arr = np.array(pil_img.convert("RGB"))
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # call predict; some paddleocr versions return nested structures
    result = PADDLE_OCR.predict(arr_bgr)
    duration = time.time() - start

    lines = extract_texts_from_predict(result)
    try:
        raw_json = json.loads(json.dumps(result, default=str))
    except Exception:
        raw_json = str(result)

    return {
        "engine": "paddleocr",
        "duration_sec": duration,
        "full_text": "\n".join(lines),
        "lines": lines,
        "raw": raw_json,
    }


# ----------------- Endpoint -----------------
@app.route("/upload", methods=["POST"])
def upload():
    
    img_bytes = None
    original_filename = None

    if "file" in request.files:
        f = request.files["file"]
        original_filename = f.filename or None
        img_bytes = f.read()
    else:
        data = request.get_json(silent=True) or {}
        b64 = data.get("image_base64") or data.get("image")
        if b64:
            try:
                img_bytes = base64.b64decode(b64)
            except Exception as e:
                return jsonify({"error": "invalid_base64", "detail": str(e)}), 400

    if not img_bytes:
        return jsonify({"error": "no_image_provided"}), 400

    # Load PIL
    try:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": "invalid_image", "detail": str(e)}), 400

    input_b64 = image_to_base64(pil_img)
    processed = preprocess_image(pil_img)

    responses: Dict[str, Any] = {}

    # Run engines (each wrapped in try/except to be robust)
    try:
        responses["Tesseract_ocr_response"] = run_tesseract(processed)
    except Exception as e:
        responses["Tesseract_ocr_response"] = {"error": str(e)}

    try:
        responses["EasyOCR_ocr_response"] = run_easyocr(processed)
    except Exception as e:
        responses["EasyOCR_ocr_response"] = {"error": str(e)}

    try:
        responses["PaddleOCR_ocr_response"] = run_paddleocr(processed)
    except Exception as e:
        responses["PaddleOCR_ocr_response"] = {"error": str(e)}

    out = {
        "Input_Image_Base64": input_b64,
        "metadata": {
            "original_size_bytes": len(img_bytes),
            "processed_size_png_bytes": len(base64.b64decode(image_to_base64(processed))),
        },
        **responses,
    }

    # --- Write server-side comparison text file ---
    if original_filename:
        safe_name = Path(original_filename).stem
    else:
        safe_name = datetime.utcnow().strftime("img_%Y%m%d%H%M%S")
    comparison_filename = IMAGES_DIR / f"comparison_{safe_name}.txt"

    try:
        lines = []
        lines.append(f"Comparison file for image: {original_filename or safe_name}")
        lines.append(f"Generated at (UTC): {datetime.utcnow().isoformat()}")
        lines.append("")
        lines.append("METADATA:")
        lines.append(json.dumps(out["metadata"], indent=2))
        lines.append("")

        # Tesseract summary
        t = out.get("Tesseract_ocr_response", {})
        lines.append("=== TESSERACT ===")
        lines.append(f"duration_sec: {t.get('duration_sec')}")
        lines.append("full_text:")
        lines.append(t.get("full_text", "").strip() or "<no text>")
        lines.append("")

        # EasyOCR summary
        e = out.get("EasyOCR_ocr_response", {})
        lines.append("=== EASYOCR ===")
        lines.append(f"duration_sec: {e.get('duration_sec')}")
        lines.append("full_text:")
        lines.append(e.get("full_text", "").strip() or "<no text>")
        lines.append("")

        # PaddleOCR summary
        p = out.get("PaddleOCR_ocr_response", {})
        lines.append("=== PADDLEOCR ===")
        lines.append(f"duration_sec: {p.get('duration_sec')}")
        lines.append("full_text:")
        lines.append(p.get("full_text", "").strip() or "<no text>")
        lines.append("")

        comparison_text = "\n".join(lines)
        comparison_filename.write_text(comparison_text, encoding="utf-8")
    except Exception as e:
        # if writing fails, include error in response but don't crash
        out["comparison_file_write_error"] = str(e)

    return jsonify(out), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
