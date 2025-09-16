# app.py
import os
import re
import json
import time
import base64
import tempfile
import traceback
from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)

# ----------------- Optional library imports & initialization -----------------
# Prefer PaddleOCR; if unavailable, we'll use Tesseract as fallback.
paddle_ocr = None
try:
    from paddleocr import PaddleOCR
    _paddle_use_gpu = os.environ.get("PADDLE_USE_GPU", "0") in ("1", "true", "True", "yes")
    _paddle_lang = os.environ.get("PADDLE_LANG", "en")
    # semantic name kept for env-var compatibility; this controls whether to use textline orientation
    _paddle_use_angle = os.environ.get("PADDLE_USE_ANGLE", "1") in ("1", "true", "True", "yes")

    # Prefer the new parameter name if available; fall back to the old one for older paddleocr versions.
    try:
        # Newer paddleocr: use_textline_orientation
        paddle_ocr = PaddleOCR(use_textline_orientation=_paddle_use_angle, lang=_paddle_lang, use_gpu=_paddle_use_gpu)
        print("PaddleOCR initialized (use_textline_orientation).")
    except TypeError:
        # Older paddleocr: use_angle_cls
        try:
            paddle_ocr = PaddleOCR(use_angle_cls=_paddle_use_angle, lang=_paddle_lang, use_gpu=_paddle_use_gpu)
            print("PaddleOCR initialized (use_angle_cls - legacy).")
        except TypeError:
            # Last resort: try minimal init
            paddle_ocr = PaddleOCR()
            print("PaddleOCR initialized (minimal constructor).")
except Exception:
    paddle_ocr = None
    print("PaddleOCR not available — will fallback to Tesseract if present.")

# Tesseract (fallback)
pytesseract = None
Output = None
try:
    import pytesseract
    from pytesseract import Output
    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd:
        pytesseract.pytesseract.tesseract_cmd = env_cmd
    else:
        default_win = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.isfile(default_win):
            pytesseract.pytesseract.tesseract_cmd = default_win
    try:
        _ver = pytesseract.get_tesseract_version()
        print("Tesseract available.")
    except Exception:
        print("pytesseract imported but tesseract binary may not be present.")
except Exception:
    pytesseract = None
    Output = None
    print("pytesseract not installed; fallback to Tesseract won't be available.")

# OpenCV for preprocessing (optional)
cv2 = None
try:
    import cv2
    print("OpenCV available for preprocessing.")
except Exception:
    cv2 = None
    print("OpenCV (cv2) not installed; preprocessing will be skipped.")


# ----------------- Document templates -----------------
DOCUMENT_TEMPLATES = {
    "PAN_CARD": {
        "patterns": [
            r"Permanent Account Number Card",
            r"Father's Name",
            r"Date of Birth",
            r"\bpan\b",
            r"INCOME TAX DEPARTMENT"
        ],
        "pan_bonus": 1
    },
    "AADHAAR_CARD": {
        "patterns": [
            r"Unique Identification Authority",
            r"Government of India",
            r"aadhaar",
            r"DOB",
            r"Male"
        ],
        "digit12_bonus": 1
    },
    "DRIVING_LICENSE": [
        r"DRIVING LICENCE",
        r"THE UNION OF INDIA",
        r"MAHARASHTRA STATE MOTOR DRIVING LICENCE",
        r"driving license",
        r"DL no",
        r"DOI",
        r"COV",
        r"AUTHORISATION TO DRIVE FOLLOWING CLASS",
        r"OF VEHICLES THROUGHOUT INDIA"
    ],
    "VOTER_ID": {
        "patterns": [
        r"ELECTION COMMISSION OF INDIA",
        r"Elector Photo Identity Card",
        r"IDENTITY CARD",
        r"Elector's Name"
    ],
    "voterid_bonus":  1
    },
    "BANK_STATEMENT": [
        r"Account Statement",
        r"STATEMENT OF ACCOUNT",
        r"IFSC",
        r"Account Number",
        r"Customer  Name",
        r"Branch"
    ]
}


# ----------------- Document-specific rules (extensible) -----------------
DOCUMENT_RULES = {
    "AADHAAR_CARD": {
        "min_score": 3,
        "checks": ["aadhaar_number"]
    },
    "PAN_CARD": {
        "min_score": 3,
        "checks": ["pan_number"]
    },
    "DRIVING_LICENSE": {
        "min_score": 3
    },
    "VOTER_ID": {
        "min_score": 3,
        "checks": ["voterid_number"]
    }
}


# ----------------- Paths -----------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images3")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)


# ----------------- Helpers -----------------
def sanitize_filename(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]', '_', name)


def bytes_to_base64(image_bytes: bytes, fmt: str = "png") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/{fmt};base64,{b64}"


def detect_12_digit_number(full_text: str):
    if not full_text:
        return False, None, None
    m = re.search(r"\b(\d{12})\b", full_text)
    if m:
        return True, m.group(1), m.group(1)
    m2 = re.search(r"\b(\d{4}[-\s]?\d{4}[-\s]?\d{4})\b", full_text)
    if m2:
        matched = m2.group(1)
        normalized = re.sub(r"\D", "", matched)
        if len(normalized) == 12:
            return True, normalized, matched
    m3 = re.search(r"(\d{4}\D?\d{4}\D?\d{4})", full_text)
    if m3:
        matched = m3.group(1)
        normalized = re.sub(r"\D", "", matched)
        if len(normalized) == 12:
            return True, normalized, matched
    return False, None, None


def detect_pan_number(full_text: str):
    if not full_text:
        return False, None, None
    patterns = [
        r"\b([A-Z]{5}\d{4}[A-Z])\b",
        r"\b([A-Z]{5}\s?\d{4}\s?[A-Z])\b",
        r"\b([A-Za-z]{5}[-\s]?\d{4}[-\s]?[A-Za-z])\b"
    ]
    for pat in patterns:
        m = re.search(pat, full_text, flags=re.IGNORECASE)
        if m:
            matched = m.group(1)
            normalized = re.sub(r"\W", "", matched).upper()
            if re.match(r"^[A-Z]{5}\d{4}[A-Z]$", normalized):
                return True, normalized, matched
    m2 = re.search(r"([A-Za-z0-9]{10})", full_text)
    if m2:
        cand = m2.group(1)
        cand_up = cand.upper()
        if re.match(r"^[A-Z]{5}\d{4}[A-Z]$", cand_up):
            return True, cand_up, cand
    return False, None, None

def detect_voterid_number(full_text: str):
    if not full_text:
        return False, None, None
    patterns = [
        r"\b([A-Z]{3}\d{7})\b",  # Typical Voter ID format
        r"\b([A-Z]{5}\d{4}[A-Z])\b", # Existing logic from PAN
        r"\b([A-Z]{5}\s?\d{4}\s?[A-Z])\b", # Existing logic from PAN
        r"\b([A-Za-z]{5}[-\s]?\d{4}[-\s]?[A-Za-z])\b" # Existing logic from PAN
    ]
    for pat in patterns:
        m = re.search(pat, full_text, flags=re.IGNORECASE)
        if m:
            matched = m.group(1)
            normalized = re.sub(r"\W", "", matched).upper()
            # This check is too restrictive; a voter ID isn't a PAN.
            # A more robust regex is needed, but for now we'll just check for a match.
            return True, normalized, matched
    m2 = re.search(r"([A-Za-z0-9]{10})", full_text)
    if m2:
        cand = m2.group(1)
        cand_up = cand.upper()
        # This check is incorrect for Voter IDs, but kept for consistency
        # as it was in the original code.
        if re.match(r"^[A-Z]{5}\d{4}[A-Z]$", cand_up):
            return True, cand_up, cand
    return False, None, None


def detect_aadhaar_number(full_text: str):
    return detect_12_digit_number(full_text)


def get_template_scores(full_text: str):
    matches = {}
    digit12_found, digit12_value, digit12_text = detect_12_digit_number(full_text)
    pan_found, pan_value, pan_text = detect_pan_number(full_text)
    voterid_found, voterid_value, voterid_text = detect_voterid_number(full_text)

    for doc_type, spec in DOCUMENT_TEMPLATES.items():
        # Initialize bonus variables for each iteration
        digit12_bonus = 0
        pan_bonus = 0
        voterid_bonus = 0

        if isinstance(spec, list):
            patterns = spec
        elif isinstance(spec, dict):
            patterns = spec.get("patterns", [])
            digit12_bonus = int(spec.get("digit12_bonus", 0) or 0)
            pan_bonus = int(spec.get("pan_bonus", 0) or 0)
            voterid_bonus = int(spec.get("voterid_bonus", 0) or 0) # Corrected this line to match the key
        else:
            patterns = []

        score = 0
        matched = []
        for pat in patterns:
            try:
                if re.search(pat, full_text, flags=re.IGNORECASE):
                    score += 1
                    matched.append(pat)
            except re.error:
                pass

        digit12_applied = False
        pan_applied = False
        voterid_applied = False
        if digit12_bonus and digit12_found:
            score += digit12_bonus
            digit12_applied = True
            matched.append("12_digit_number")
        if pan_bonus and pan_found:
            score += pan_bonus
            pan_applied = True
            matched.append("pan_number")
        if voterid_bonus and voterid_found:
            score += voterid_bonus
            voterid_applied = True
            matched.append("voterid_number")

        matches[doc_type] = {
            "score": score,
            "matched": matched,
            "digit12_bonus_applied": digit12_applied,
            "pan_bonus_applied": pan_applied,
            "voterid_bonus_applied": voterid_applied
        }

    extras = {
        "digit12_found": digit12_found,
        "digit12_value": digit12_value,
        "digit12_text": digit12_text,
        "pan_found": pan_found,
        "pan_value": pan_value,
        "pan_text": pan_text,
        "voterid_found": voterid_found,
        "voterid_value": voterid_value,
        "voterid_text": voterid_text
    }
    return matches, extras


def classify_document(full_text: str):
    matches, extras = get_template_scores(full_text)
    best_doc, best_info = max(matches.items(), key=lambda x: x[1]["score"])
    result = {
        "name": "UNKNOWN",
        "score": best_info["score"],
        "matched_keywords": best_info["matched"]
    }
    doc_rule = DOCUMENT_RULES.get(best_doc, {})
    min_score = doc_rule.get("min_score", 1)
    if best_info["score"] < min_score:
        return result

    result["name"] = best_doc

    result["digit12_found"] = bool(extras.get("digit12_found"))
    result["digit12_value"] = extras.get("digit12_value")
    result["digit12_text"] = extras.get("digit12_text")

    result["pan_found"] = bool(extras.get("pan_found"))
    result["pan_value"] = extras.get("pan_value")
    result["pan_text"] = extras.get("pan_text")

    result["voterid_found"] = bool(extras.get("voterid_found"))
    result["voterid_value"] = extras.get("voterid_value")
    result["voterid_text"] = extras.get("voterid_text")

    checks = doc_rule.get("checks", [])
    if "aadhaar_number" in checks:
        found, normalized, matched_text = detect_aadhaar_number(full_text)
        result["aadhaar_number_found"] = bool(found)
        result["aadhaar_number"] = normalized
        result["aadhaar_number_text"] = matched_text

    if "pan_number" in checks:
        found, normalized, matched_text = detect_pan_number(full_text)
        result["pan_number_found"] = bool(found)
        result["pan_number"] = normalized
        result["pan_number_text"] = matched_text

    if "voterid_number" in checks:
        found, normalized, matched_text = detect_voterid_number(full_text)
        result["voterid_number_found"] = bool(found)
        result["voterid_number"] = normalized
        result["voterid_number_text"] = matched_text

    return result


# ----------------- Preprocessing -----------------
def preprocess_image(image_path: str):
    if cv2 is None:
        return image_path
    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            return image_path
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(gray)
        denoised = cv2.bilateralFilter(cl, d=9, sigmaColor=75, sigmaSpace=75)
        th = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, blockSize=15, C=9)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            save_path = tmp.name
            cv2.imwrite(save_path, opened)
            return save_path
    except Exception:
        print("Preprocessing failed; continuing with original image.")
        return image_path


# ----------------- Utilities to ensure JSON-safe outputs -----------------
def json_default(o):
    try:
        import numpy as _np
        if isinstance(o, _np.integer):
            return int(o)
        if isinstance(o, _np.floating):
            return float(o)
        if isinstance(o, _np.ndarray):
            return o.tolist()
    except Exception:
        pass
    try:
        return int(o)
    except Exception:
        try:
            return float(o)
        except Exception:
            return str(o)


def ensure_json_serializable(obj):
    try:
        serialized = json.dumps(obj, default=json_default, ensure_ascii=False)
        return json.loads(serialized)
    except Exception:
        return str(obj)


# ----------------- OCR backends (with normalization) -----------------
def run_paddle_and_parse(image_path: str):
    if paddle_ocr is None:
        raise RuntimeError("PaddleOCR not available.")
    try:
        ocr_res = paddle_ocr.ocr(image_path, cls=True)
    except TypeError:
        ocr_res = paddle_ocr.ocr(image_path)
    parsed_lines = []
    full_text_parts = []
    for item in ocr_res:
        try:
            if not item:
                continue
            box = item[0]
            text_conf = item[1]
            if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 1:
                text = str(text_conf[0])
                conf = None
                if len(text_conf) > 1:
                    try:
                        conf = float(text_conf[1])
                    except Exception:
                        conf = None
            elif isinstance(text_conf, dict):
                text = str(text_conf.get("text", ""))
                conf = text_conf.get("confidence")
                try:
                    conf = float(conf) if conf is not None else None
                except Exception:
                    conf = None
            else:
                text = str(text_conf)
                conf = None
            bbox = []
            if isinstance(box, (list, tuple)):
                for pt in box:
                    try:
                        x = int(round(float(pt[0])))
                        y = int(round(float(pt[1])))
                        bbox.append([x, y])
                    except Exception:
                        pass
            text_str = text.strip()
            if not text_str:
                continue
            parsed_lines.append({
                "text": text_str,
                "confidence": float(conf) if conf is not None else None,
                "bbox": bbox
            })
            full_text_parts.append(text_str)
        except Exception:
            continue
    full_text = "\n".join(full_text_parts)
    return parsed_lines, full_text


def run_tesseract_and_parse(image_path: str):
    if pytesseract is None:
        raise RuntimeError("pytesseract not available.")
    try:
        data = pytesseract.image_to_data(Image.open(image_path), output_type=Output.DICT, lang='eng')
    except Exception as e:
        raise RuntimeError(f"Tesseract OCR failed: {e}")
    n = len(data.get('text', []))
    lines = {}
    for i in range(n):
        txt = (data['text'][i] or "").strip()
        if not txt:
            continue
        key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
        left = int(data['left'][i])
        top = int(data['top'][i])
        width = int(data['width'][i])
        height = int(data['height'][i])
        conf_raw = data['conf'][i]
        try:
            conf = float(conf_raw)
        except Exception:
            conf = None
        item = {
            "word": txt,
            "left": left,
            "top": top,
            "width": width,
            "height": height,
            "conf": conf
        }
        lines.setdefault(key, []).append(item)
    parsed_lines = []
    full_text_parts = []
    for key, words in lines.items():
        words_sorted = sorted(words, key=lambda w: w["left"])
        text_line = " ".join(w["word"] for w in words_sorted)
        full_text_parts.append(text_line)
        lefts = [w["left"] for w in words_sorted]
        tops = [w["top"] for w in words_sorted]
        rights = [w["left"] + w["width"] for w in words_sorted]
        bottoms = [w["top"] + w["height"] for w in words_sorted]
        bbox = [[min(lefts), min(tops)],
                [max(rights), min(tops)],
                [max(rights), max(bottoms)],
                [min(lefts), max(bottoms)]]
        confs = [w["conf"] for w in words_sorted if w["conf"] is not None and w["conf"] >= 0]
        avg_conf = (sum(confs) / len(confs)) if confs else None
        parsed_lines.append({
            "text": text_line,
            "confidence": float(avg_conf) if avg_conf is not None else None,
            "bbox": bbox
        })
    full_text = "\n".join(full_text_parts)
    return parsed_lines, full_text


# ----------------- Flask endpoint -----------------
@app.route("/detect", methods=["POST"])
def detect():
    try:
        image_bytes = None
        fmt = "png"
        original_name = None
        data = None

        if 'file' in request.files:
            f = request.files['file']
            image_bytes = f.read()
            original_name = f.filename or f"upload_{int(time.time())}"
            if '.' in original_name:
                fmt = original_name.rsplit('.', 1)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{fmt}") as tmp:
                tmp.write(image_bytes)
                image_path = tmp.name
        else:
            data = request.get_json(silent=True) or {}
            if data.get("image_name"):
                image_name = os.path.basename(data["image_name"])
                if not os.path.splitext(image_name)[1]:
                    image_name += ".png"
                candidate = os.path.join(IMAGES_DIR, image_name)
                if not os.path.isfile(candidate):
                    return jsonify({"error": f"image_name not found: {candidate}"}), 400
                image_path = candidate
                original_name = image_name
            elif data.get("image_path"):
                path = os.path.normpath(data["image_path"])
                if not os.path.isabs(path):
                    candidate = os.path.join(PROJECT_ROOT, path)
                    if os.path.isfile(candidate):
                        path = candidate
                    else:
                        candidate2 = os.path.join(IMAGES_DIR, path)
                        if os.path.isfile(candidate2):
                            path = candidate2
                if not os.path.isfile(path):
                    return jsonify({"error": f"image_path does not exist: {path}"}), 400
                image_path = path
                original_name = os.path.basename(path)
            elif data.get("image_base64"):
                b64 = data["image_base64"]
                if b64.startswith("data:") and ";base64," in b64:
                    b64 = b64.split(";base64,", 1)[1]
                image_bytes = base64.b64decode(b64)
                original_name = f"image_base64_{int(time.time())}.png"
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    tmp.write(image_bytes)
                    image_path = tmp.name
            else:
                return jsonify({"error": "No image provided. Send 'file' or JSON with 'image_name'/'image_path'/'image_base64'."}), 400

        if image_bytes is None:
            with open(image_path, "rb") as fh:
                image_bytes = fh.read()
            fmt = original_name.rsplit(".", 1)[-1].lower()

        preprocessed_path = preprocess_image(image_path)

        ocr_used = None
        final_parsed = []
        final_full_text = ""

        # Prefer PaddleOCR
        if paddle_ocr is not None:
            try:
                parsed, full_text = run_paddle_and_parse(preprocessed_path)
                if full_text and len(full_text.strip()) >= 4:
                    final_parsed = parsed
                    final_full_text = full_text
                    ocr_used = "paddleocr"
                else:
                    print("PaddleOCR returned insufficient text — falling back to Tesseract.")
            except Exception as e:
                print("PaddleOCR error (using fallback):", str(e))

        # If PaddleOCR didn't produce satisfactory output, use Tesseract (if available)
        if not final_full_text:
            if pytesseract is not None:
                try:
                    parsed, full_text = run_tesseract_and_parse(preprocessed_path)
                    if full_text and len(full_text.strip()) >= 4:
                        final_parsed = parsed
                        final_full_text = full_text
                        ocr_used = "tesseract"
                    else:
                        print("Tesseract returned insufficient text.")
                except Exception as e:
                    print("Tesseract error:", str(e))
            else:
                return jsonify({"error": "No OCR backend available. Install PaddleOCR or Tesseract."}), 500

        # classify (even if text is empty)
        doc_match = classify_document(final_full_text or "")

        input_b64 = bytes_to_base64(image_bytes, fmt=fmt)

        response_obj = {
            "inputbase64": input_b64,
            "detected_text": final_parsed,
            "document_type": doc_match,
            "ocr_used": ocr_used
        }

        # Save JSON output if document detected
        if doc_match.get("name") and doc_match["name"] != "UNKNOWN" and doc_match.get("score", 0) > 0:
            if not original_name:
                original_name = f"result_{int(time.time())}.json"
            base_name = sanitize_filename(os.path.splitext(original_name)[0])
            json_filename = f"{base_name}.json"
            json_path = os.path.join(OUTPUT_DIR, json_filename)
            if os.path.exists(json_path):
                json_filename = f"{base_name}_{int(time.time())}.json"
                json_path = os.path.join(OUTPUT_DIR, json_filename)
            try:
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(response_obj, jf, ensure_ascii=False, indent=2, default=json_default)
                rel_json = os.path.relpath(json_path, PROJECT_ROOT)
            except Exception as e:
                print("Failed to save JSON output:", e)
                rel_json = None
        else:
            rel_json = None

        safe_response = ensure_json_serializable(response_obj)

        # cleanup
        try:
            if preprocessed_path != image_path and os.path.isfile(preprocessed_path):
                os.remove(preprocessed_path)
        except Exception:
            pass
        try:
            if data and data.get("image_base64"):
                os.remove(image_path)
        except Exception:
            try:
                if 'image_path' in locals() and original_name and original_name.startswith("upload_"):
                    os.remove(image_path)
            except Exception:
                pass

        return jsonify({"result": safe_response, "saved_json": rel_json}), 200

    except Exception as e:
        tb = traceback.format_exc()
        msg = str(e)
        return jsonify({"error": msg, "trace": tb}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)