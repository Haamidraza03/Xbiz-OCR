# app.py
import os
import io
import re
import json
import base64
import tempfile
from datetime import datetime
from flask import Flask, request, jsonify, render_template, g
from werkzeug.utils import secure_filename
import uuid
import logging
from typing import List, Dict, Tuple, Any, Optional
import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

# Auth / JWT imports
import time
import jwt
import hmac
from functools import wraps

# pdf2image (may require poppler installed on system)
try:
    from pdf2image import convert_from_bytes
except Exception:
    convert_from_bytes = None

# Optional: allow user to set POPPLER_PATH via env var if needed on Windows
POPPLER_PATH = os.environ.get("POPPLER_PATH", None)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_logger(name, log_file, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Avoid duplicate handlers if logger already configured
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_file) for h in logger.handlers if hasattr(h, "baseFilename")):
        handler = logging.FileHandler(log_file)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
    return logger

FUZZY_THRESHOLD = int(os.environ.get("FUZZY_THRESHOLD", "75"))

# similarity helper (rapidfuzz if available else difflib)
try:
    from rapidfuzz import fuzz, utils
    def similarity(a: str, b: str) -> float:
        a_norm = utils.default_process(a or "")
        b_norm = utils.default_process(b or "")
        return float(fuzz.token_set_ratio(a_norm, b_norm)) if a_norm and b_norm else 0.0
except Exception:
    from difflib import SequenceMatcher
    def similarity(a: str, b: str) -> float:
        return float(SequenceMatcher(None, a or "", b or "").ratio() * 100.0)

ALLOWED = {"png", "jpg", "jpeg", "tiff", "bmp", "pdf"}

# Initialize PaddleOCR (adjust args for your environment if necessary)
ocr = PaddleOCR(use_angle_cls=True, lang="en", rec_batch_num=16)

# Image processing params
BLUR_THRESHOLD = 100.0
UPSCALE_FACTOR_BLUR = 1.6
UNSHARP_WEIGHT = 1.5
UNSHARP_BLUR_WEIGHT = -0.5
UNSHARP_GAUSSIAN_SIGMA = 3

DOCUMENT_TEMPLATES = {
    "PAN_CARD": {"patterns": [r"Permanent Account Number Card", r"Father's Name", r"Date of Birth", r"\bpan\b", r"INCOME TAX DEPARTMENT"], "pan_bonus": 1},
    "AADHAAR_CARD": {"patterns": [r"Unique Identification Authority", r"Government of India", r"DOB", r"Male"], "digit12_bonus": 1},
    "DRIVING_LICENSE": {"patterns": [r"DRIVING LICENCE", r"THE UNION OF INDIA", r"MAHARASHTRA STATE MOTOR DRIVING LICENCE", r"driving license", r"DL no", r"DOI", r"COV", r"AUTHORISATION TO DRIVE FOLLOWING CLASS", r"OF VEHICLES THROUGHOUT INDIA"]},
    "VOTER_ID": {"patterns": [r"ELECTION COMMISSION OF INDIA", r"Elector Photo Identity Card", r"IDENTITY CARD", r"Elector's Name", r"\b[A-Z]{3}\d{7}\b"], "voterid_bonus": 1},
    "BANK_STATEMENT": {"patterns": [r"Account Statement", r"STATEMENT OF ACCOUNT", r"IFSC", r"Account Number", r"Customer Name", r"Branch", r"Statement of Transactions", r"Account Description"], "account11_bonus": 1}
}

DOCUMENT_RULES = {
    "AADHAAR_CARD": {"min_score": 3, "checks": ["aadhaar_number"]},
    "PAN_CARD": {"min_score": 3, "checks": ["pan_number"]},
    "DRIVING_LICENSE": {"min_score": 3},
    "VOTER_ID": {"min_score": 3, "checks": ["voterid_number"]},
    "BANK_STATEMENT": {"min_score": 4, "checks": ["bank_account_number"]}
}

app = Flask(__name__)

# ---------------- JWT / CONFIG ----------------

def load_auth_config():
    """
    Load username/password and jwt secret from config.json or env vars.
    config.json format (optional):
    {
      "username": "admin",
      "password": "mysecret",
      "jwt_secret": "supersecret"
    }
    Environment variables:
      BASIC_AUTH_USER
      BASIC_AUTH_PASS
      JWT_SECRET
      JWT_EXP_SECONDS  (optional, default 3600)
    Returns: (user, pass, jwt_secret, exp_seconds)
    """
    cfg_path = os.environ.get("CONFIG_PATH", "config.json")
    user = None
    pwd = None
    secret = None
    exp_seconds = int(os.environ.get("JWT_EXP_SECONDS", "3600"))
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as cf:
                cfg = json.load(cf)
            user = cfg.get("username") or user
            pwd = cfg.get("password") or pwd
            secret = cfg.get("jwt_secret") or secret
            if "jwt_exp_seconds" in cfg:
                try:
                    exp_seconds = int(cfg.get("jwt_exp_seconds"))
                except Exception:
                    pass
        except Exception:
            logging.warning("Could not read config.json for auth; falling back to env vars.")
    # env fallbacks
    user = user or os.environ.get("BASIC_AUTH_USER")
    pwd = pwd or os.environ.get("BASIC_AUTH_PASS")
    secret = secret or os.environ.get("JWT_SECRET") or "change-me-default-secret"
    try:
        exp_seconds = int(os.environ.get("JWT_EXP_SECONDS", exp_seconds))
    except Exception:
        pass
    return user, pwd, secret, exp_seconds

AUTH_USER, AUTH_PASS, JWT_SECRET, JWT_EXP_SECONDS = load_auth_config()

def generate_jwt_token(username: str) -> str:
    now = int(time.time())
    payload = {
        "sub": username,
        "iat": now,
        "exp": now + int(JWT_EXP_SECONDS)
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")
    # PyJWT may return bytes in older versions; ensure string
    if isinstance(token, bytes):
        token = token.decode("utf-8")
    return token

def jwt_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header:
            return jsonify({"error": "Authorization header missing"}), 401
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return jsonify({"error": "Invalid Authorization header format. Use Bearer <token>"}), 401
        token = parts[1]
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            # attach payload to flask.g for downstream use if needed
            g.jwt_payload = payload
            return f(*args, **kwargs)
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
    return decorated

# -- simple login endpoint to obtain token --
@app.route("/login", methods=["POST"])
def login():
    """
    POST JSON: { "username": "...", "password": "..." }
    Returns: { "access_token": "...", "token_type": "bearer", "expires_in": <seconds> }
    """
    if AUTH_USER is None or AUTH_PASS is None:
        return jsonify({"error": "Server JWT auth not configured. Set BASIC_AUTH_USER and BASIC_AUTH_PASS or provide config.json"}), 500
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400
    username = data.get("username", "") if isinstance(data, dict) else ""
    password = data.get("password", "") if isinstance(data, dict) else ""
    if not username or not password:
        return jsonify({"error": "username and password required"}), 400
    if hmac.compare_digest(username, AUTH_USER) and hmac.compare_digest(password, AUTH_PASS):
        token = generate_jwt_token(username)
        return jsonify({"access_token": token, "token_type": "bearer", "expires_in": int(JWT_EXP_SECONDS)})
    else:
        return jsonify({"error": "Invalid credentials"}), 401

# ---------------- end JWT / CONFIG ----------------

def allowed_file(fn: str) -> bool:
    return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED

def read_image_bytes(b: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(b)).convert("RGB")
    return np.array(img)[:, :, ::-1]  # PIL (RGB) -> OpenCV BGR

def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def detect_skew_angle_using_hough(binary_img: np.ndarray) -> float:
    blurred = cv2.GaussianBlur(binary_img, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=80, minLineLength=max(50, binary_img.shape[1] // 10), maxLineGap=20)
    angles = []
    if lines is not None:
        for (x1, y1, x2, y2) in lines.reshape(-1, 4):
            a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if a < -90: a += 180
            if a > 90: a -= 180
            if abs(a) <= 45: angles.append(a)
    if angles:
        return -float(np.median(angles))
    coords = np.column_stack(np.where(binary_img < 255))
    if coords.shape[0] < 10:
        return 0.0
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    return angle

def deblur_and_sharpen_gray(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    gray_up = cv2.resize(gray, (int(w * UPSCALE_FACTOR_BLUR), int(h * UPSCALE_FACTOR_BLUR)), interpolation=cv2.INTER_CUBIC)
    den = cv2.fastNlMeansDenoising(gray_up, None, h=10, templateWindowSize=7, searchWindowSize=21)
    gauss = cv2.GaussianBlur(den, (0, 0), UNSHARP_GAUSSIAN_SIGMA)
    unsharp = cv2.addWeighted(den, UNSHARP_WEIGHT, gauss, UNSHARP_BLUR_WEIGHT, 0)
    enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(unsharp)
    closed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    return cv2.bilateralFilter(opened, d=9, sigmaColor=75, sigmaSpace=75)

def preprocess_using_tempfile_and_user_logic(img_bgr: np.ndarray, target_width: int = 1200) -> Tuple[np.ndarray, float, np.ndarray]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t:
        orig_tmp = t.name
        cv2.imwrite(orig_tmp, img_bgr)
    img = cv2.imread(orig_tmp, cv2.IMREAD_COLOR)
    try:
        os.remove(orig_tmp)
    except Exception:
        pass
    if img is None:
        return img_bgr, 0.0, img_bgr
    h0, w0 = img.shape[:2]
    if w0 < target_width:
        img = cv2.resize(img, None, fx=target_width / w0, fy=target_width / w0, interpolation=cv2.INTER_CUBIC)
    coarse_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coarse_cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(coarse_gray)
    _, coarse_bin = cv2.threshold(coarse_cl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    angle = detect_skew_angle_using_hough(coarse_bin)
    if abs(angle) > 0.1:
        img = rotate_image(img, angle)
    else:
        angle = 0.0
    blur = cv2.GaussianBlur(img, (5, 5), 2)
    sharp = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    denoised = cv2.fastNlMeansDenoisingColored(sharp, None, 6, 10, 9, 21)
    deskewed_color = denoised.copy()
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    # robust normalization to avoid dtype issues
    bg = cv2.medianBlur(gray, 31).astype(np.float32) + 1e-6
    gray_f = gray.astype(np.float32)
    norm = (gray_f / bg) * 255.0
    norm = np.clip(norm, 0, 255).astype(np.uint8)
    return norm, angle, deskewed_color

# --- detection helpers for numbers and ids ---
def detect_12_digit_number(full_text: str):
    m = re.search(r"\b(\d{12})\b", full_text)
    if m:
        return True, m.group(1), m.group(1)
    m2 = re.search(r"\b(\d{4}[-\s]?\d{4}[-\s]?\d{4})\b", full_text)
    if m2:
        normalized = re.sub(r"\D", "", m2.group(1))
        if len(normalized) == 12:
            return True, normalized, m2.group(1)
    m3 = re.search(r"(\d{4}\D?\d{4}\D?\d{4})", full_text)
    if m3:
        normalized = re.sub(r"\D", "", m3.group(1))
        if len(normalized) == 12:
            return True, normalized, m3.group(1)
    return False, None, None

def detect_pan_number(full_text: str):
    patterns = [r"\b([A-Z]{5}\d{4}[A-Z])\b", r"\b([A-Z]{5}\s?\d{4}\s?[A-Z])\b", r"\b([A-Za-z]{5}[-\s]?\d{4}[-\s]?[A-Za-z])\b"]
    for pat in patterns:
        m = re.search(pat, full_text, flags=re.IGNORECASE)
        if m:
            normalized = re.sub(r"\W", "", m.group(1)).upper()
            if re.match(r"^[A-Z]{5}\d{4}[A-Z]$", normalized):
                return True, normalized, m.group(1)
    m2 = re.search(r"([A-Za-z0-9]{10})", full_text)
    if m2:
        cand_up = m2.group(1).upper()
        if re.match(r"^[A-Z]{5}\d{4}[A-Z]$", cand_up):
            return True, cand_up, m2.group(1)
    return False, None, None

def detect_voterid_number(full_text: str):
    m = re.search(r"\b([A-Z]{3}\d{7})\b", full_text, flags=re.IGNORECASE)
    if m:
        return True, re.sub(r"\W", "", m.group(1)).upper(), m.group(1)
    m2 = re.search(r"(Elector(?:'s)? Name|Elector Photo Identity Card|ELECTION COMMISSION OF INDIA)", full_text, flags=re.IGNORECASE)
    if m2:
        context = full_text[max(0, m2.end() - 120): m2.end() + 120]
        m3 = re.search(r"([A-Z0-9]{8,10})", context, flags=re.IGNORECASE)
        if m3:
            return True, re.sub(r"\W", "", m3.group(1)).upper(), m3.group(1)
    return False, None, None

def detect_aadhaar_number(full_text: str):
    return detect_12_digit_number(full_text)

def detect_bank_account_number_11(full_text: str):
    m = re.search(r"\b(\d{11})\b", full_text)
    if m:
        return True, m.group(1), m.group(1)
    m2 = re.search(r"\b(\d{3}[-\s]?\d{4}[-\s]?\d{4})\b", full_text)
    if m2:
        normalized = re.sub(r"\D", "", m2.group(1))
        if len(normalized) == 11:
            return True, normalized, m2.group(1)
    m3 = re.search(r"((?:\d{2,5}\D?){2,4}\d{2,5})", full_text)
    if m3:
        cand = re.sub(r"\D", "", m3.group(1))
        if len(cand) == 11:
            return True, cand, m3.group(1)
    return False, None, None

def detect_aadhaar_components(full_text: str):
    aadhaar_found, aadhaar_value, aadhaar_text = detect_12_digit_number(full_text)
    male_found = bool(re.search(r"\bMale\b|\b(?:Gender|Sex)[:\s]*M\b", full_text, flags=re.IGNORECASE))
    dob_found = bool(re.search(r"\bDate of Birth\b|\bDOB\b|\b\d{1,2}[\/\-\s](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|[0-9]{1,2})[\/\-\s]\d{2,4}\b|\b\d{1,2}[\/\-\s]\d{1,2}[\/\-\s]\d{2,4}\b", full_text, flags=re.IGNORECASE))
    m_date = re.search(r"(\b\d{1,2}[\/\-\s]\d{1,2}[\/\-\s]\d{2,4}\b|\b\d{4}[\/\-\s]\d{1,2}[\/\-\s]\d{1,2}\b|\b\d{1,2}[\/\-\s](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|[0-9]{1,2})[\/\-\s]\d{2,4}\b)", full_text, flags=re.IGNORECASE)
    gov_found = bool(re.search(r"Government of India|GOVERNMENT OF INDIA", full_text, flags=re.IGNORECASE))
    unique_auth_found = bool(re.search(r"Unique Identification Authority|UIDAI|Unique Identification Authority of India", full_text, flags=re.IGNORECASE))
    address_found = bool(re.search(r"\bAddress\b|\bPermanent Address\b|\bAddress :", full_text, flags=re.IGNORECASE))
    relation_found = bool(re.search(r"\bS\/O\b|\bSO\b|\bD\/O\b|\bDO\b|\bW\/O\b|\bWO\b|\bon of\b|\bwife of\b|\bdaughter of\b", full_text, flags=re.IGNORECASE))
    return {
        "aadhaar_found": aadhaar_found, "aadhaar_value": aadhaar_value or aadhaar_text,
        "male_found": male_found, "male_value": "Male" if re.search(r"\bMale\b", full_text, flags=re.IGNORECASE) else "M" if male_found else None,
        "dob_found": dob_found, "dob_text": m_date.group(1) if m_date else None,
        "gov_found": gov_found, "unique_auth_found": unique_auth_found,
        "address_found": address_found, "relation_found": relation_found
    }

# --- OCR extraction helper (robust for various paddle outputs) ---
def extract_texts_from_ocr_result(result) -> List[str]:
    def extract(obj):
        texts = []
        if isinstance(obj, str) and obj.strip():
            texts.append(obj.strip())
        elif isinstance(obj, dict):
            for key in ("text", "rec_text", "transcription", "sentence", "text_line"):
                if key in obj and isinstance(obj[key], str) and obj[key].strip():
                    texts.append(obj[key].strip())
            for v in obj.values():
                texts.extend(extract(v))
        elif isinstance(obj, (list, tuple)):
            for el in obj:
                texts.extend(extract(el))
        return texts
    all_texts = extract(result)
    filtered = [t for t in all_texts if any(ch.isalnum() for ch in t)]
    # preserve order unique
    seen = set()
    out = []
    for t in filtered:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def _clean_pattern_to_plaintext(pat: str) -> str:
    cleaned = re.sub(r'\\b', ' ', pat)
    cleaned = re.sub(r'[\^\$\.\*\+\?\(\)\[\]\{\}\\\|]', ' ', cleaned)
    cleaned = re.sub(r'[^A-Za-z0-9 ]+', ' ', cleaned)
    return re.sub(r'\s+', ' ', cleaned).strip()

def classify_document(full_text: str, ocr_lines: List[str] = None) -> Dict[str, Any]:
    ocr_lines = ocr_lines or full_text.splitlines()
    digit12_found, digit12_value, digit12_text = detect_12_digit_number(full_text)
    pan_found, pan_norm, pan_orig = detect_pan_number(full_text)
    voter_found, voter_norm, voter_orig = detect_voterid_number(full_text)
    acc_found, acc_norm, acc_orig = detect_bank_account_number_11(full_text)
    aadhaar_components = detect_aadhaar_components(full_text)
    full_text_norm = re.sub(r'\s+', ' ', full_text).strip()
    results = {}
    for dtype, template in DOCUMENT_TEMPLATES.items():
        score = 0
        matched_patterns = []
        patterns = template.get("patterns", [])
        for pat in patterns:
            m = re.search(pat, full_text, flags=re.IGNORECASE)
            if m:
                score += 1
                matched_patterns.append({"pattern": pat, "match": m.group(0), "method": "regex", "score": None})
                continue
            plain = _clean_pattern_to_plaintext(pat)
            if plain:
                best_score = 0.0
                best_line = None
                for ln in ocr_lines:
                    sim = similarity(plain.lower(), ln.lower())
                    if sim > best_score:
                        best_score = sim
                        best_line = ln
                        if best_score >= FUZZY_THRESHOLD:
                            break
                if best_score < FUZZY_THRESHOLD and full_text_norm:
                    sim_full = similarity(plain.lower(), full_text_norm.lower())
                    if sim_full > best_score:
                        best_score = sim_full
                        best_line = full_text_norm
                if best_score >= FUZZY_THRESHOLD:
                    score += 1
                    matched_patterns.append({"pattern": pat, "match": best_line, "method": "fuzzy", "score": int(best_score)})
        if template.get("pan_bonus") and pan_found:
            score += template["pan_bonus"]
            matched_patterns.append({"pattern": "PAN_NUMBER", "match": pan_norm, "method": "detector", "score": None})
        if template.get("digit12_bonus") and digit12_found:
            score += template["digit12_bonus"]
            matched_patterns.append({"pattern": "AADHAAR_NUMBER", "match": digit12_value, "method": "detector", "score": None})
        if template.get("voterid_bonus") and voter_found:
            score += template["voterid_bonus"]
            matched_patterns.append({"pattern": "VOTERID_NUMBER", "match": voter_norm, "method": "detector", "score": None})
        if template.get("account11_bonus") and acc_found:
            score += template["account11_bonus"]
            matched_patterns.append({"pattern": "ACCOUNT11_NUMBER", "match": acc_norm, "method": "detector", "score": None})
        results[dtype] = {"score": score, "matched_patterns": matched_patterns}
    selection_scores = {}
    for dtype, info in results.items():
        rules = DOCUMENT_RULES.get(dtype, {})
        checks = rules.get("checks", [])
        checks_pass = True
        for chk in checks:
            found = False
            if chk == "aadhaar_number":
                found = detect_aadhaar_number(full_text)[0]
            elif chk == "pan_number":
                found = detect_pan_number(full_text)[0]
            elif chk == "voterid_number":
                found = detect_voterid_number(full_text)[0]
            elif chk == "bank_account_number":
                found = detect_bank_account_number_11(full_text)[0]
            if not found:
                checks_pass = False
        min_score = rules.get("min_score", 0)
        acceptable = (info["score"] >= min_score) and checks_pass
        selection_scores[dtype] = {"score": info["score"], "acceptable": acceptable, "checks_pass": checks_pass, "min_score": min_score, "matched_patterns": info["matched_patterns"]}
    acceptable_candidates = [(d, v["score"]) for d, v in selection_scores.items() if v["acceptable"]]
    best = max(acceptable_candidates, key=lambda x: x[1])[0] if acceptable_candidates else "UNKNOWN"
    pan_found2, pan_norm2, pan_orig2 = detect_pan_number(full_text)
    aad_found2, aad_norm2, aad_orig2 = detect_aadhaar_number(full_text)
    voter_found2, voter_norm2, voter_orig2 = detect_voterid_number(full_text)
    acc_found2, acc_norm2, acc_orig2 = detect_bank_account_number_11(full_text)
    aadhaar_components2 = detect_aadhaar_components(full_text)
    result = {
        "best_type": best,
        "scores": {k: v["score"] for k, v in selection_scores.items()},
        "selection_meta": selection_scores,
        "matched": {k: v["matched_patterns"] for k, v in selection_scores.items()},
        "numbers": {
            "pan": {"found": pan_found2, "normalized": pan_norm2, "original": pan_orig2},
            "aadhaar": {"found": aad_found2, "normalized": aad_norm2, "original": aad_orig2},
            "voter": {"found": voter_found2, "normalized": voter_norm2, "original": voter_orig2},
            "bank_account_11": {"found": acc_found2, "normalized": acc_norm2, "original": acc_orig2}
        },
        "aadhaar_components": aadhaar_components2
    }
    ac = aadhaar_components2
    front_count = sum(bool(ac.get(k)) for k in ["aadhaar_found", "male_found", "dob_found", "gov_found"])
    back_count = sum(bool(ac.get(k)) for k in ["unique_auth_found", "address_found", "relation_found"])
    result["aadhaar_front_count"] = front_count
    result["aadhaar_back_count"] = back_count
    result["aadhaar_front_detected"] = front_count >= 2
    result["aadhaar_back_detected"] = back_count >= 1
    result["aadhaar_both_detected"] = front_count >= 2 and back_count >= 1
    doc_rule = DOCUMENT_RULES.get(best, {})
    checks = doc_rule.get("checks", [])
    if "aadhaar_number" in checks:
        found, normalized, matched_text = detect_aadhaar_number(full_text)
        result["aadhaar_number_found"] = found
        result["aadhaar_number"] = normalized
        result["aadhaar_number_text"] = matched_text
    if "pan_number" in checks:
        found, normalized, matched_text = detect_pan_number(full_text)
        result["pan_number_found"] = found
        result["pan_number"] = normalized
        result["pan_number_text"] = matched_text
    if "voterid_number" in checks:
        found, normalized, matched_text = detect_voterid_number(full_text)
        result["voterid_number_found"] = found
        result["voterid_number"] = normalized
        result["voterid_number_text"] = matched_text
    if "bank_account_number" in checks:
        found, normalized, matched_text = detect_bank_account_number_11(full_text)
        result["bank_account_number_found"] = found
        result["bank_account_number"] = normalized
        result["bank_account_number_text"] = matched_text
    return result

# --- extract words/boxes robustly from PaddleOCR output ---
def extract_words_and_boxes_from_paddle(ocr_raw, logger=logging) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    def process_rec(texts, boxes, scores):
        for i, text in enumerate(texts):
            try:
                box = boxes[i]
            except Exception:
                continue
            confidence = (scores[i] if (scores and i < len(scores)) else None)
            vertices = []
            # normalize different box formats
            pts = box.tolist() if hasattr(box, 'tolist') else box
            if isinstance(pts, (list, tuple)) and len(pts) >= 4:
                for point in pts:
                    if isinstance(point, (list, tuple)) and len(point) >= 2:
                        try:
                            x = int(round(float(point[0])))
                            y = int(round(float(point[1])))
                            vertices.append([x, y])
                        except Exception:
                            continue
            if vertices and text:
                out.append({"text": str(text).strip(), "vertices": vertices, "confidence": float(confidence) if confidence is not None else None})

    def traverse(obj):
        if isinstance(obj, dict):
            if 'rec_texts' in obj and 'rec_polys' in obj:
                texts = obj.get('rec_texts', [])
                boxes = obj.get('rec_polys', [])
                scores = obj.get('rec_scores', [])
                process_rec(texts, boxes, scores)
            for v in obj.values():
                traverse(v)
        elif isinstance(obj, (list, tuple)):
            for el in obj:
                traverse(el)

    traverse(ocr_raw)
    return out

def _centroid(vertices):
    if not vertices:
        return (0.0, 0.0)
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    return (sum(xs) / len(xs), sum(ys) / len(ys))

def group_words_to_lines(words: List[Dict[str, Any]], tolerance_multiplier=0.6, min_tolerance_px=10, use_baseline=True, logger=logging) -> List[Tuple[float, List[Tuple[float, str]]]]:
    items = []
    for w in words:
        verts = w.get("vertices") or []
        if not verts:
            continue
        xs = [float(p[0]) for p in verts]
        ys = [float(p[1]) for p in verts]
        x_left = min(xs)
        y_top = min(ys)
        y_bottom = max(ys)
        height = max(1.0, y_bottom - y_top)
        items.append({"text": w.get("text", ""), "x_left": x_left, "y_center": (y_top + y_bottom) / 2.0, "y_top": y_top, "y_bottom": y_bottom, "height": height, "centroid": _centroid(verts)})
    if not items:
        return []
    heights = [it["height"] for it in items if it["height"] > 0]
    median_h = float(np.median(heights)) if heights else 12.0
    tol = max(min_tolerance_px, int(round(median_h * tolerance_multiplier)))
    items.sort(key=lambda e: e["y_center"])
    lines = []
    for it in items:
        placed = False
        for line in lines:
            if abs(it["y_center"] - line["center"]) <= tol:
                line["items"].append(it)
                line["center"] = sum(x["y_center"] for x in line["items"]) / len(line["items"])
                line["y_min"] = min(line["y_min"], it["y_top"])
                line["y_max"] = max(line["y_max"], it["y_bottom"])
                placed = True
                break
        if not placed:
            lines.append({"center": it["y_center"], "y_min": it["y_top"], "y_max": it["y_bottom"], "items": [it]})
    final = []
    for line in lines:
        li_sorted = sorted([(it["x_left"], it["text"]) for it in line["items"]], key=lambda t: t[0])
        final.append((line["center"], li_sorted))
    return sorted(final, key=lambda t: t[0])

def create_sorted_text_file_from_words(words: List[Dict[str, Any]], output_path: str, tolerance_multiplier=0.6, min_tolerance_px: int = 10, use_baseline: bool = True, logger=logging) -> str:
    lines = group_words_to_lines(words, tolerance_multiplier, min_tolerance_px, use_baseline, logger)
    with open(output_path, "w", encoding="utf-8") as fh:
        for _, elems in lines:
            line_text = " ".join(t for _, t in elems).strip()
            if line_text:
                fh.write(line_text + "\n")
    return output_path

def draw_annotations(img: np.ndarray, texts: List[str], boxes: List[List[List[int]]], out_path: str, confidences: Optional[List[float]] = None, color=(0,200,0), logger=logging) -> str:
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_draw = img.copy()
    h, w = img.shape[:2]
    left_margin_x = 8
    left_stack_y = 8
    left_stack_step = 18
    any_drawn = False
    font_scale = max(0.2, min(0.3, w / 900.0))
    text_thickness = max(1, int(round(font_scale * 2)))
    for i, text in enumerate(texts):
        bbox = boxes[i] if i < len(boxes) else None
        conf = confidences[i] if confidences and i < len(confidences) else None
        label_parts = [f"#{i}"]
        if conf is not None:
            try:
                label_parts.append(f"{conf:.2f}")
            except Exception:
                label_parts.append(str(conf))
        label_parts.append(str(text))
        label = " ".join(label_parts)
        if len(label) > 200:
            label = label[:197] + "..."
        (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
        pad_x, pad_y = 6, 3
        if bbox and len(bbox) >= 3:
            pts = np.array(bbox, dtype=np.int32)
            try:
                cv2.polylines(img_draw, [pts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
            except Exception:
                pass
            bx, by, bw_box, bh_box = cv2.boundingRect(pts)
            rect_x1 = max(0, bx)
            rect_y1 = max(0, by)
            rect_x2 = min(w - 1, rect_x1 + max(bw_box, w_text + 2 * pad_x))
            rect_y2 = min(h - 1, rect_y1 + h_text + 2 * pad_y)
            if rect_y2 - rect_y1 > bh_box:
                rect_y1 = max(0, by + bh_box - h_text - 2 * pad_y)
                rect_y2 = min(h - 1, by + bh_box)
            overlay = img_draw.copy()
            cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), color, -1)
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, img_draw, 1 - alpha, 0, img_draw)
            text_color = (0, 0, 0) if sum(color) > 300 else (255, 255, 255)
            text_y = rect_y1 + h_text + pad_y
            cv2.putText(img_draw, label, (rect_x1 + pad_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, text_thickness, cv2.LINE_AA)
            any_drawn = True
        if not any_drawn or (bbox and len(bbox) < 3):
            step = max(left_stack_step, h_text + 2 * pad_y + 2)
            if left_stack_y + step > h - 4:
                left_stack_y = 8
            rect_x1 = left_margin_x
            rect_x2 = min(w - 1, rect_x1 + w_text + 2 * pad_x)
            rect_y1 = left_stack_y
            rect_y2 = rect_y1 + (h_text + 2 * pad_y)
            overlay = img_draw.copy()
            cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), color, -1)
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, img_draw, 1 - alpha, 0, img_draw)
            text_color = (0, 0, 0) if sum(color) > 300 else (255, 255, 255)
            cv2.putText(img_draw, label, (rect_x1 + pad_x, rect_y2 - pad_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, text_thickness, cv2.LINE_AA)
            left_stack_y += step
            any_drawn = True
    if not any_drawn:
        cv2.putText(img_draw, "NO DETECTIONS", (20, max(50, h // 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3, cv2.LINE_AA)
    cv2.imwrite(out_path, img_draw)
    return out_path

# ---------- New helper: split a page image into document crops ----------
def split_page_into_documents(page_bgr: np.ndarray, min_area_ratio: float = 0.02) -> List[np.ndarray]:
    """
    Attempts to detect rectangular document regions on a page image.
    Returns list of cropped images (BGR). If no meaningful crop detected,
    returns a single element list with the original page.
    min_area_ratio: minimum contour area relative to page area to consider.
    """
    h, w = page_bgr.shape[:2]
    page_area = float(h * w)
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    # Enhance and threshold
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 41, 12)
    # Morphological closing to join text blocks into blobs
    kernel_w = max(15, w // 60)
    kernel_h = max(15, h // 60)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    # find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area / page_area < min_area_ratio:
            continue
        boxes.append((x, y, bw, bh))
    if not boxes:
        return [page_bgr]
    # merge overlapping boxes (simple NMS-like merge)
    boxes_sorted = sorted(boxes, key=lambda b: (b[1], b[0]))
    merged = []
    for box in boxes_sorted:
        x, y, bw, bh = box
        if not merged:
            merged.append(box)
            continue
        mx, my, mbw, mbh = merged[-1]
        # if overlapping or close, merge
        if (x < mx + mbw + 10) and (y < my + mbh + 10):
            nx = min(mx, x)
            ny = min(my, y)
            nxb = max(mx + mbw, x + bw) - nx
            nyb = max(my + mbh, y + bh) - ny
            merged[-1] = (nx, ny, nxb, nyb)
        else:
            merged.append(box)
    crops = []
    for idx, (x, y, bw, bh) in enumerate(merged, start=1):
        pad_x = int(round(0.02 * bw))
        pad_y = int(round(0.02 * bh))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)
        crop = page_bgr[y1:y2, x1:x2].copy()
        crops.append(crop)
    return crops

# ---------- Core OCR processing for a single image bytes, but writing into given base_dir ----------
def process_image_bytes_internal(img_bytes: bytes, filename: str, base_dir: str, base: str, ts: str, logs) -> Tuple[Dict[str, Any], Dict[str, Any], List[str], List[Dict[str, Any]], float, str, str, str, str]:
    """
    Processes a single image and writes outputs into the provided base_dir structure.
    Returns: result_dict, classification, lines, words, angle, sorted_txt_path, detection_json_path, txt_detected_path, annotated_image_path
    """
    output_dir = os.path.join(base_dir, "OUTPUT")
    ocr_image_dir = os.path.join(output_dir, "OCR_IMAGE")
    ocr_sorted_dir = os.path.join(output_dir, "OCR_SORTED")
    ocr_text_dir = os.path.join(output_dir, "OCR_TEXT")
    assets_dir = os.path.join(base_dir, "Assets")
    for d in (output_dir, ocr_image_dir, ocr_sorted_dir, ocr_text_dir, assets_dir):
        os.makedirs(d, exist_ok=True)

    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    img = read_image_bytes(img_bytes)
    processed_gray, angle, processed_color = preprocess_using_tempfile_and_user_logic(img)
    tmp_path = os.path.join(base_dir, f"tmp_{base}_{ts}.png")
    try:
        tmp_img_to_write = processed_gray
        if tmp_img_to_write.dtype != np.uint8:
            tmp_img_to_write = np.clip(tmp_img_to_write, 0, 255).astype(np.uint8)
    except Exception:
        tmp_img_to_write = (processed_gray * 255).astype(np.uint8) if hasattr(processed_gray, 'dtype') and processed_gray.dtype == np.float32 else processed_gray
    cv2.imwrite(tmp_path, tmp_img_to_write)
    ocr_raw = ocr.predict(tmp_path) if hasattr(ocr, "predict") else ocr.ocr(tmp_path)
    lines = extract_texts_from_ocr_result(ocr_raw)
    full_text = "\n".join(lines)
    txt_detected_path = os.path.join(ocr_text_dir, f"{base}_{ts}.txt")
    with open(txt_detected_path, "w", encoding="utf-8") as outf:
        outf.write(full_text)
    classification = classify_document(full_text, lines)
    words = extract_words_and_boxes_from_paddle(ocr_raw, logs)
    sorted_txt_path = os.path.join(ocr_sorted_dir, f"{base}_{ts}.txt")
    create_sorted_text_file_from_words(words, sorted_txt_path, logger=logs)
    texts_for_draw = [w["text"] for w in words]
    boxes_for_draw = [w["vertices"] for w in words]
    confidences = [w.get("confidence") for w in words]
    annotated_image_path = os.path.join(ocr_image_dir, f"{base}_{ts}.png")
    draw_annotations(processed_color, texts_for_draw, boxes_for_draw, annotated_image_path, confidences, logger=logs)
    with open(annotated_image_path, "rb") as ann_f:
        ann_b64 = base64.b64encode(ann_f.read()).decode("utf-8")
    json_payload = {
        "filename": filename, "timestamp": ts, "image_base64": img_b64,
        "detected_text": full_text, "lines": lines, "deskew_angle": angle,
        "classification": classification, "words_count": len(words),
        "boxes_count": len([w for w in words if w.get("vertices")]),
        "annotated_image_path": annotated_image_path, "sorted_text_path": sorted_txt_path
    }
    detection_json_path = os.path.join(assets_dir, f"{base}_{ts}_detection.json")
    with open(detection_json_path, "w", encoding="utf-8") as jf:
        json.dump(json_payload, jf, indent=2, ensure_ascii=False)

    # write request JSON with only filename as requested
    try:
        request_json_path = os.path.join(assets_dir, f"{base}_{ts}_request.json")
        with open(request_json_path, "w", encoding="utf-8") as rjf:
            json.dump({"filename": filename}, rjf, indent=2, ensure_ascii=False)
    except Exception:
        try:
            logs.exception("Failed to write request.json")
        except Exception:
            pass

    result_dict = {
        "inputbase64": ann_b64, "document_type": classification.get("best_type"),
        "cleaned_text": [], "detected_text": lines, "sorted_text_path": sorted_txt_path
    }
    response_json_path = os.path.join(assets_dir, f"{base}_{ts}_response.json")
    with open(response_json_path, "w", encoding="utf-8") as respjf:
        json.dump(result_dict, respjf, indent=2)
    try:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
    except Exception:
        pass
    return result_dict, classification, lines, words, angle, sorted_txt_path, detection_json_path, txt_detected_path, annotated_image_path

# ---------- Main process_file now supports PDF and images ----------
def process_file(img_bytes, filename):
    name = secure_filename(filename)
    base = os.path.splitext(name)[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    txnid = uuid.uuid4().hex[:12]
    base_dir = f"{base}_{txnid}"
    os.makedirs(base_dir, exist_ok=True)
    input_dir = os.path.join(base_dir, "INPUT")
    os.makedirs(input_dir, exist_ok=True)
    logs_dir = os.path.join(base_dir, "LOGS")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{base}_{ts}.log")
    logs = setup_logger(f"ocr_{base}_{ts}", log_path)
    logs.info("Request ocr is running")
    logs.info(f"Starting processing for file: {name}")
    original_path = os.path.join(input_dir, name)
    with open(original_path, "wb") as of:
        of.write(img_bytes)

    # Safe defaults
    overall_result: Dict[str, Any] = {"inputbase64": "", "document_type": "ERROR", "cleaned_text": [], "detected_text": [], "sorted_text_path": "", "sub_results": []}
    overall_classification: Dict[str, Any] = {}
    overall_lines: List[str] = []
    overall_words: List[Dict[str, Any]] = []
    overall_angle: float = 0.0
    overall_sorted_txt_path: str = ""
    assets_dir = os.path.join(base_dir, "Assets")
    os.makedirs(assets_dir, exist_ok=True)
    detection_json_path = os.path.join(assets_dir, f"{base}_{ts}_detection_error.json")
    txt_detected_path = ""
    annotated_image_path = ""

    ext = name.rsplit(".", 1)[1].lower() if "." in name else ""
    try:
        if ext == "pdf":
            if convert_from_bytes is None:
                raise RuntimeError("pdf2image.convert_from_bytes unavailable. Install pdf2image and poppler and set POPPLER_PATH if needed.")
            # convert pdf bytes to PIL pages
            try:
                if POPPLER_PATH:
                    pil_pages = convert_from_bytes(img_bytes, dpi=200, poppler_path=POPPLER_PATH)
                else:
                    pil_pages = convert_from_bytes(img_bytes, dpi=200)
            except Exception as e:
                logs.exception("pdf2image failed to convert PDF - make sure poppler is installed")
                raise
            # save pages to INPUT folder and split each page into document crops
            for p_idx, pil_page in enumerate(pil_pages, start=1):
                page_name = f"{base}_page_{p_idx:02d}.png"
                page_path = os.path.join(input_dir, page_name)
                pil_page.save(page_path, "PNG")
                # convert to BGR numpy
                page_bgr = np.array(pil_page.convert("RGB"))[:, :, ::-1]
                # split page into document crops
                crops = split_page_into_documents(page_bgr)
                # for each crop save and process
                for d_idx, crop_bgr in enumerate(crops, start=1):
                    crop_name = f"{base}_page_{p_idx:02d}_doc_{d_idx:02d}.png"
                    crop_path = os.path.join(input_dir, crop_name)
                    cv2.imwrite(crop_path, crop_bgr)
                    with open(crop_path, "rb") as rf:
                        crop_bytes = rf.read()
                    # process this document image (writes outputs into same base_dir)
                    sub_res = process_image_bytes_internal(crop_bytes, crop_name, base_dir, base + f"_page{p_idx:02d}_doc{d_idx:02d}", ts, logs)
                    overall_result["sub_results"].append(sub_res[0])
            if overall_result["sub_results"]:
                overall_result["document_type"] = "PDF_CONTAINS_DOCUMENTS"
        else:
            # single image
            result_dict, classification, lines, words, angle, sorted_txt_path, detection_json_path, txt_detected_path, annotated_image_path = process_image_bytes_internal(img_bytes, name, base_dir, base, ts, logs)
            overall_result = result_dict
            overall_classification = classification
            overall_lines = lines
            overall_words = words
            overall_angle = angle
            overall_sorted_txt_path = sorted_txt_path
    except Exception as e:
        logs.exception("Processing failed (top level)")
        try:
            with open(detection_json_path, "w", encoding="utf-8") as jf:
                json.dump({"error": str(e)}, jf, indent=2)
        except Exception:
            pass

    # ensure logs flushed/closed
    for handler in logs.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logs.removeHandler(handler)
            try:
                handler.flush()
                handler.close()
            except Exception:
                pass

    return overall_result, overall_classification, overall_lines, overall_words, overall_angle, overall_sorted_txt_path, detection_json_path, txt_detected_path, annotated_image_path

# ---- Flask endpoints ----

@app.route("/upload", methods=["POST"])
@jwt_required
def upload():
    files = request.files.getlist("file")
    if not files:
        return jsonify({"error": "no file"}), 400
    results = []
    for f in files:
        if f.filename == "" or not allowed_file(f.filename):
            continue
        img_bytes = f.read()
        result, classification, lines, words, angle, sorted_txt_path, detection_json_path, txt_detected_path, annotated_image_path = process_file(img_bytes, f.filename)
        results.append({"original_name": f.filename, "result": result})
    if not results:
        return jsonify({"error": "no valid files"}), 400
    if len(results) == 1:
        return jsonify({"result": results[0]["result"]})
    else:
        return jsonify({"results": results})

@app.route("/detect", methods=["POST"])
@jwt_required
def detect():
    files = request.files.getlist("file")
    if not files:
        return jsonify({"error": "no file"}), 400
    results = []
    for f in files:
        if f.filename == "" or not allowed_file(f.filename):
            continue
        img_bytes = f.read()
        result, _, _, _, _, sorted_txt_path, _, _, _ = process_file(img_bytes, f.filename)
        if result.get("sub_results"):
            for sr in result["sub_results"]:
                results.append({"original_name": f.filename, "result": sr})
        else:
            results.append({"original_name": f.filename, "result": result})
    if not results:
        return jsonify({"error": "no valid files"}), 400
    return jsonify({"result": results[0]["result"]}) if len(results) == 1 else jsonify({"results": results})

@app.route('/detect_all', methods=['GET'])
@jwt_required
def detect_all():
    folder = 'images_multi'
    if not os.path.exists(folder):
        return jsonify({"error": "Folder not found"}), 404
    results = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if os.path.isfile(path) and allowed_file(filename):
            with open(path, 'rb') as f:
                img_bytes = f.read()
            result, _, _, _, _, _, _, _, _ = process_file(img_bytes, filename)
            results.append({"original_name": filename, "result": result})
    if not results:
        return jsonify({"error": "no valid files in images_multi folder"}), 400
    return jsonify({"results": results})

@app.route('/')
@jwt_required
def index():
    return render_template('index2.html')

@app.route("/get_sorted_text", methods=["POST"])
@jwt_required
def get_sorted_text():
    data = request.get_json(force=True)
    file_path = data.get('file_path', '') if isinstance(data, dict) else ''
    if not file_path:
        return jsonify({"error": "Invalid file path"}), 400
    requested = os.path.abspath(os.path.normpath(file_path))
    allowed_dir = os.path.abspath(os.path.normpath("."))
    if not requested.startswith(allowed_dir + os.sep) and requested != allowed_dir:
        return jsonify({"error": "Invalid file path"}), 400
    if not os.path.exists(requested):
        return jsonify({"error": "File not found"}), 404
    with open(requested, 'r', encoding='utf-8') as f:
        content = f.read()
    return jsonify({"sorted_text": content})

if __name__ == "__main__":
    if convert_from_bytes is None:
        logging.warning("pdf2image.convert_from_bytes is not available. PDF uploads will fail until pdf2image + poppler are installed.")
    # Ensure auth config is loaded (already loaded at import), log if not configured
    if AUTH_USER is None or AUTH_PASS is None:
        logging.warning("BASIC_AUTH_USER / BASIC_AUTH_PASS not configured. Create config.json or set env vars BASIC_AUTH_USER/BASIC_AUTH_PASS.")
    # start the server
    app.run(debug=True, host="0.0.0.0", port=5000)
