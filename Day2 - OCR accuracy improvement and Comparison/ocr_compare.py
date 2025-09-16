# ocr_api.py
"""
OCR multi-backend Flask API with improved preprocessing + server-side comparison.json writer.

- Runs Tesseract (pytesseract), EasyOCR, and PaddleOCR on the same preprocessed image
- Returns JSON with Input_Image_Base64 and the three OCR responses
- Writes a comparison JSON file under ./images2/comparison_<imagename>.json

Requirements (pip):
 pip install flask pillow numpy opencv-python pytesseract easyocr paddlepaddle paddleocr

Note: Tesseract must be installed on the system (Windows: install .exe and set path).
"""
import io
import time
import base64
import json
import os
import tempfile
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

# If tesseract is not on PATH, explicitly set it here (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ----------------- Config -----------------
BASE_DIR = Path(__file__).parent.resolve()
IMAGES_DIR = BASE_DIR / "images2"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}

# Paddle / EasyOCR language (adjust as required)
LANGS = ['en']
PADDLE_LANG = 'en'

# Initialize OCR engines once (may download models on first init)
EASY_OCR_READER = easyocr.Reader(LANGS, gpu=False)
PADDLE_OCR = PaddleOCR(use_angle_cls=True, lang=PADDLE_LANG)

app = Flask(__name__)

# ----------------- Preprocessing helpers -----------------
def pil_to_cv2(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert('RGB'))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv2_to_pil(img: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def image_to_base64(img: Image.Image) -> str:
    buff = io.BytesIO()
    img.save(buff, format='PNG')
    return base64.b64encode(buff.getvalue()).decode('ascii')

def preprocess_image(pil_img: Image.Image, target_height: int = 1200) -> Image.Image:
    # Convert to cv2
    img_cv = pil_to_cv2(pil_img)
    h, w = img_cv.shape[:2]
    if h < target_height and h > 0:
        scale = target_height / h
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_cv = cv2.resize(img_cv, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    # Edge-preserving denoise
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    # Non-local means
    gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    # Median blur
    gray = cv2.medianBlur(gray, 3)
    # Adaptive threshold
    block_size = 15 if max(gray.shape) < 2000 else 31
    gray_bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, block_size, 9)
    # Morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    gray_bin = cv2.morphologyEx(gray_bin, cv2.MORPH_OPEN, kernel)
    processed_bgr = cv2.cvtColor(gray_bin, cv2.COLOR_GRAY2BGR)
    return cv2_to_pil(processed_bgr)

# ----------------- OCR wrappers -----------------
def run_tesseract(pil_img: Image.Image) -> Dict[str, Any]:
    start = time.time()
    data = pytesseract.image_to_data(pil_img, output_type=Output.DICT, config='--oem 3 --psm 6')
    duration = time.time() - start
    entries = []
    n = len(data['level'])
    for i in range(n):
        t = data['text'][i].strip()
        if not t:
            continue
        obj = {
            'text': t,
            'left': int(data['left'][i]),
            'top': int(data['top'][i]),
            'width': int(data['width'][i]),
            'height': int(data['height'][i]),
            'conf': float(data['conf'][i]) if data['conf'][i] != '-1' else None
        }
        entries.append(obj)
    full_text = pytesseract.image_to_string(pil_img, config='--oem 3 --psm 6')
    return {'engine': 'tesseract', 'duration_sec': duration, 'full_text': full_text, 'words': entries}

def run_easyocr(pil_img: Image.Image) -> Dict[str, Any]:
    start = time.time()
    arr = np.array(pil_img.convert('RGB'))
    results = EASY_OCR_READER.readtext(arr)
    duration = time.time() - start
    entries = []
    for bbox, text, conf in results:
        bbox_list = [[int(x[0]), int(x[1])] for x in bbox]
        entries.append({'text': text, 'conf': float(conf), 'bbox': bbox_list})
    full_text = "\n".join([r[1] for r in results])
    return {'engine': 'easyocr', 'duration_sec': duration, 'full_text': full_text, 'results': entries}

def extract_texts_from_predict(result) -> List[str]:
    texts = []
    def rec(obj):
        if isinstance(obj, str):
            if obj.strip(): texts.append(obj.strip()); return
        if isinstance(obj, (int,float)): return
        if isinstance(obj, (list,tuple)):
            if len(obj) == 2 and isinstance(obj[0], str) and isinstance(obj[1], (int,float)):
                if obj[0].strip(): texts.append(obj[0].strip()); return
            if len(obj) >= 2:
                sec = obj[1]
                if isinstance(sec, (list,tuple)):
                    if len(sec) >= 1 and isinstance(sec[0], str):
                        if sec[0].strip(): texts.append(sec[0].strip()); return
                    else:
                        for item in sec: rec(item)
                        return
            for item in obj: rec(item)
            return
        if isinstance(obj, dict):
            for v in obj.values(): rec(v)
            return
        return
    rec(result)
    return [t for t in texts if t]

def parse_paddle_result(result) -> (List[str], List[Dict[str,Any]]):
    """
    Parse many common paddleocr return shapes into lines and entries.
    """
    lines = []
    entries = []

    if isinstance(result, list):
        for item in result:
            # item might be [[box], (text, conf)] or [ [box, (text,conf)], [box, (text,conf)], ... ]
            if not item:
                continue
            if isinstance(item, list) and len(item) > 0:
                # If first child is a list like [box,...], assume it's a line with words
                first = item[0]
                if isinstance(first, list) and len(first) >= 2:
                    # iterate possible word-like entries
                    for maybe in item:
                        try:
                            bbox = maybe[0]
                            rec = maybe[1]
                            if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                                text = rec[0]; conf = rec[1]
                            elif isinstance(rec, str):
                                text = rec; conf = None
                            else:
                                text = str(rec); conf = None
                            lines.append(str(text))
                            entries.append({'text': str(text), 'conf': float(conf) if conf is not None else None, 'bbox': bbox})
                        except Exception:
                            continue
                else:
                    # item is likely [box, (text,conf)]
                    try:
                        bbox = item[0]
                        rec = item[1]
                        if isinstance(rec, (list,tuple)) and len(rec) >= 2:
                            text = rec[0]; conf = rec[1]
                        elif isinstance(rec, str):
                            text = rec; conf = None
                        else:
                            text = str(rec); conf = None
                        lines.append(str(text))
                        entries.append({'text': str(text), 'conf': float(conf) if conf is not None else None, 'bbox': bbox})
                    except Exception:
                        # fallback to recursive extractor
                        pass

    if not lines:
        # last-resort: generic extractor
        try:
            extracted = extract_texts_from_predict(result)
            lines = extracted
            entries = [{'text': t} for t in extracted]
        except Exception:
            pass

    return lines, entries

def run_paddleocr(pil_img: Image.Image) -> Dict[str, Any]:
    """
    Robust PaddleOCR runner:
      - saves processed PIL image to a temporary file
      - tries PADDLE_OCR.ocr(tmp_path) first (no extra kwargs)
      - falls back to PADDLE_OCR.predict(arr_bgr) if ocr() fails
      - parses result into text lines and entries
    """
    # prepare numpy BGR
    arr = np.array(pil_img.convert('RGB'))
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # write temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    tmp_path = Path(tmp.name)
    try:
        pil_img.save(tmp_path, format='PNG')
    finally:
        tmp.close()

    result = None
    tried = []
    start = time.time()
    try:
        # Try the simple and widely-supported .ocr(path) first (no cls kwarg).
        try:
            tried.append('ocr(path)')
            result = PADDLE_OCR.ocr(str(tmp_path))
        except TypeError as e:
            tried.append(f'ocr(path) TypeError: {e}')
            # fallback to predict with numpy
            try:
                tried.append('predict(arr)')
                result = PADDLE_OCR.predict(arr_bgr)
            except Exception as e2:
                tried.append(f'predict(arr) error: {e2}')
                # last attempt: try ocr with other signature without cls
                try:
                    tried.append('ocr(path, use_gpu=False)')
                    result = PADDLE_OCR.ocr(str(tmp_path), use_gpu=False)
                except Exception as e3:
                    tried.append(f'ocr(path,use_gpu=False) error: {e3}')
                    # Give up and raise the last exception to upper layer
                    raise
    finally:
        duration = time.time() - start
        # remove temp file
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    # parse result
    lines, entries = parse_paddle_result(result)
    try:
        raw_json = json.loads(json.dumps(result, default=str))
    except Exception:
        raw_json = str(result)

    return {
        'engine': 'paddleocr',
        'duration_sec': duration,
        'full_text': "\n".join(lines),
        'lines': lines,
        'results': entries,
        'raw': raw_json,
        'tried': tried
    }

# ----------------- Endpoint -----------------
@app.route('/upload', methods=['POST'])
def upload():
    """
    Accepts multipart file upload (form field 'file') or JSON {'image_base64': ...}
    Returns JSON with base64 input and OCR outputs from Tesseract, EasyOCR, PaddleOCR.
    Also writes a server-side comparison JSON file into ./images2/
    """
    img_bytes = None
    original_filename = None

    if 'file' in request.files:
        f = request.files['file']
        original_filename = f.filename or None
        img_bytes = f.read()
    else:
        data = request.get_json(silent=True) or {}
        b64 = data.get('image_base64') or data.get('image')
        if b64:
            try:
                img_bytes = base64.b64decode(b64)
            except Exception as e:
                return jsonify({'error': 'invalid_base64', 'detail': str(e)}), 400

    if not img_bytes:
        return jsonify({'error': 'no_image_provided'}), 400

    # Load PIL
    try:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    except Exception as e:
        return jsonify({'error': 'invalid_image', 'detail': str(e)}), 400

    input_b64 = image_to_base64(pil_img)
    processed = preprocess_image(pil_img)

    responses = {}
    # Run engines (each wrapped in try/except to be robust)
    try:
        responses['Tesseract_ocr_response'] = run_tesseract(processed)
    except Exception as e:
        responses['Tesseract_ocr_response'] = {'error': str(e)}
    try:
        responses['EasyOCR_ocr_response'] = run_easyocr(processed)
    except Exception as e:
        responses['EasyOCR_ocr_response'] = {'error': str(e)}
    try:
        responses['PaddleOCR_ocr_response'] = run_paddleocr(processed)
    except Exception as e:
        responses['PaddleOCR_ocr_response'] = {'error': str(e)}

    # Build the comparison JSON structure (exact keys requested)
    comparison_data = {
        "Input_Image_Base64": input_b64,
        "Tesseract_ocr_response": responses.get('Tesseract_ocr_response', {}),
        "EasyOCR_ocr_response": responses.get('EasyOCR_ocr_response', {}),
        "PaddleOCR_ocr_response": responses.get('PaddleOCR_ocr_response', {})
    }

    # Write server-side comparison JSON
    if original_filename:
        safe_name = Path(original_filename).stem
    else:
        safe_name = datetime.utcnow().strftime("img_%Y%m%d%H%M%S")
    comparison_filename = IMAGES_DIR / f"comparison_{safe_name}.json"

    try:
        comparison_filename.write_text(json.dumps(comparison_data, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    except Exception as e:
        comparison_data['_comparison_file_write_error'] = str(e)

    # Also return a richer server response (include metadata if helpful)
    out = {
        'metadata': {
            'original_size_bytes': len(img_bytes),
            'processed_size_png_bytes': len(base64.b64decode(image_to_base64(processed)))
        },
        **comparison_data
    }

    return jsonify(out), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
