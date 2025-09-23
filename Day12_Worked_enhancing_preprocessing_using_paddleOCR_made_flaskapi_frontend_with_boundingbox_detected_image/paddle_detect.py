# app.py
import os
import io
import re
import json
import base64
import tempfile
from datetime import datetime
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

# Environment-tunable fuzzy threshold
FUZZY_THRESHOLD = int(os.environ.get("FUZZY_THRESHOLD", "75"))

# Try to import rapidfuzz (preferred)
_HAS_RAPIDFUZZ = False
try:
    from rapidfuzz import fuzz as _rf_fuzz
    from rapidfuzz import utils as _rf_utils

    def similarity(a: str, b: str) -> float:
        """Return token_set_ratio-like similarity (0..100)."""
        try:
            a_norm = _rf_utils.default_process(a or "")
            b_norm = _rf_utils.default_process(b or "")
            if not a_norm or not b_norm:
                return 0.0
            return float(_rf_fuzz.token_set_ratio(a_norm, b_norm))
        except Exception:
            try:
                return float(_rf_fuzz.ratio(a or "", b or ""))
            except Exception:
                return 0.0

    _HAS_RAPIDFUZZ = True
    print("rapidfuzz available — using token_set_ratio for fuzzy matching.")
except Exception:
    # fallback to difflib
    from difflib import SequenceMatcher as _SM

    def similarity(a: str, b: str) -> float:
        try:
            return float(_SM(None, a or "", b or "").ratio() * 100.0)
        except Exception:
            return 0.0

    print("rapidfuzz not installed — falling back to difflib SequenceMatcher for fuzzy matching.")

ALLOWED = {"png", "jpg", "jpeg", "tiff", "bmp"}
os.makedirs("Output", exist_ok=True)

# Initialize PaddleOCR (v3.x uses .predict()).
ocr = PaddleOCR(use_angle_cls=True, lang="en", rec_batch_num=16)

app = Flask(__name__)

# ---------- TUNABLE: blur detection and sharpening params ----------
BLUR_THRESHOLD = 100.0   # Laplacian variance below this considered "blurry"
UPSCALE_FACTOR_BLUR = 1.6
UNSHARP_WEIGHT = 1.5
UNSHARP_BLUR_WEIGHT = -0.5
UNSHARP_GAUSSIAN_SIGMA = 3
# ------------------------------------------------------------------


# ---------------- Document templates & rules -----------------------
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
            r"DOB",
            r"Male"
        ],
        "digit12_bonus": 1
    },
    "DRIVING_LICENSE": {
        "patterns": [
            r"DRIVING LICENCE",
            r"THE UNION OF INDIA",
            r"MAHARASHTRA STATE MOTOR DRIVING LICENCE",
            r"driving license",
            r"DL no",
            r"DOI",
            r"COV",
            r"AUTHORISATION TO DRIVE FOLLOWING CLASS",
            r"OF VEHICLES THROUGHOUT INDIA"
        ]
    },
    "VOTER_ID": {
        "patterns": [
            r"ELECTION COMMISSION OF INDIA",
            r"Elector Photo Identity Card",
            r"IDENTITY CARD",
            r"Elector's Name",
            r"\b[A-Z]{3}\d{7}\b"
        ],
        "voterid_bonus": 1
    },
    "BANK_STATEMENT": {
        "patterns": [
            r"Account Statement",
            r"STATEMENT OF ACCOUNT",
            r"IFSC",
            r"Account Number",
            r"Customer Name",
            r"Branch",
            r"Statement of Transactions",
            r"Account Description"
        ],
        "account11_bonus": 1
    }
}

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
    },
    "BANK_STATEMENT": {
        "min_score": 4,
        "checks": ["bank_account_number"]
    }
}
# ------------------------------------------------------------------


def allowed_file(fn: str) -> bool:
    return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED


def read_image_bytes(b: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(b)).convert("RGB")
    arr = np.array(img)[:, :, ::-1]  # PIL RGB -> BGR
    return arr


def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def detect_skew_angle_using_hough(binary_img: np.ndarray) -> float:
    blurred = cv2.GaussianBlur(binary_img, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=80,
                            minLineLength=max(50, binary_img.shape[1] // 10),
                            maxLineGap=20)
    angles = []
    if lines is not None:
        lines_arr = np.asarray(lines).reshape(-1, 4)
        for (x1, y1, x2, y2) in lines_arr:
            a = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
            if a < -90:
                a += 180
            if a > 90:
                a -= 180
            if abs(a) <= 45:
                angles.append(a)
    if len(angles) > 0:
        median_angle = float(np.median(angles))
        return -median_angle

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


# ----------------- blur detection + deblur/sharpen ----------------------
def is_blurry(gray: np.ndarray) -> float:
    try:
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(lap.var())
    except Exception:
        return 0.0


def deblur_and_sharpen_gray(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    new_w = max(1, int(w * UPSCALE_FACTOR_BLUR))
    new_h = max(1, int(h * UPSCALE_FACTOR_BLUR))
    gray_up = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    den = cv2.fastNlMeansDenoising(gray_up, None, h=10, templateWindowSize=7, searchWindowSize=21)
    gauss = cv2.GaussianBlur(den, (0, 0), UNSHARP_GAUSSIAN_SIGMA)
    unsharp = cv2.addWeighted(den, UNSHARP_WEIGHT, gauss, UNSHARP_BLUR_WEIGHT, 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(unsharp)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, k_close, iterations=1)
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_open, iterations=1)
    filtered = cv2.bilateralFilter(opened, d=9, sigmaColor=75, sigmaSpace=75)
    return filtered
# ----------------------------------------------------------------------


# ------------------ preprocessing (deskew-first) -----------------------
def preprocess_using_tempfile_and_user_logic(img_bgr: np.ndarray, target_width: int = 1200):
    orig_tmp = None
    proc_tmp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t:
            orig_tmp = t.name
            cv2.imwrite(orig_tmp, img_bgr)

        img = cv2.imread(orig_tmp, cv2.IMREAD_COLOR)
        if img is None:
            return img_bgr, 0.0

        # initial resize for skew detection
        h0, w0 = img.shape[:2]
        if w0 < target_width:
            scale = target_width / float(w0)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        coarse_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        coarse_cl = clahe.apply(coarse_gray)
        _, coarse_bin = cv2.threshold(coarse_cl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # deskew color image first
        angle = detect_skew_angle_using_hough(coarse_bin)
        if abs(angle) > 0.1:
            img = rotate_image(img, angle)
        else:
            angle = 0.0

        # Apply the provided sequence: blur -> unsharp -> denoise -> grayscale -> background normalization
        # Resulting `norm_uint8` will be returned as the processed image for OCR downstream.
        try:
            # Gaussian blur on color image
            blur = cv2.GaussianBlur(img, (5, 5), 2)
            # unsharp/sharpen
            sharp = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
            # colored denoising
            denoised = cv2.fastNlMeansDenoisingColored(sharp, None, 6, 10, 9, 21)
            # to grayscale
            gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
            # background via median blur
            bg = cv2.medianBlur(gray, 31)
            # avoid division by zero
            bg_safe = bg.copy().astype("float32")
            bg_safe[bg_safe == 0] = 1.0
            # normalize by dividing with background and scaling
            norm = cv2.divide(gray.astype("float32"), bg_safe, scale=255.0)
            # normalize full range to 0..255 and convert to uint8
            norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX)
            norm_uint8 = norm.astype("uint8")

            # write out processed image to a temp file for downstream OCR
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t2:
                proc_tmp = t2.name
                cv2.imwrite(proc_tmp, norm_uint8)

            return norm_uint8, float(angle)
        except Exception:
            # fall back to returning grayscale of rotated image if something fails
            try:
                gray_fb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                return gray_fb, float(angle)
            except Exception:
                return img, float(angle)
        # ------------------------------------------------------------------------------

    except Exception:
        try:
            if proc_tmp and os.path.exists(proc_tmp):
                os.remove(proc_tmp)
            if orig_tmp and os.path.exists(orig_tmp):
                os.remove(orig_tmp)
        except Exception:
            pass
        try:
            gray_fb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            return gray_fb, 0.0
        except Exception:
            return img_bgr, 0.0
    finally:
        try:
            if orig_tmp and os.path.exists(orig_tmp):
                os.remove(orig_tmp)
            if proc_tmp and os.path.exists(proc_tmp):
                os.remove(proc_tmp)
        except Exception:
            pass
# ----------------------------------------------------------------------


# ------------------ number detection helpers (unchanged) -------------------
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
    m = re.search(r"\b([A-Z]{3}\d{7})\b", full_text, flags=re.IGNORECASE)
    if m:
        matched = m.group(1)
        normalized = re.sub(r"\W", "", matched).upper()
        return True, normalized, matched
    m2 = re.search(r"(Elector(?:'s)? Name|Elector Photo Identity Card|ELECTION COMMISSION OF INDIA)", full_text, flags=re.IGNORECASE)
    if m2:
        idx = m2.end()
        context = full_text[max(0, idx - 120): idx + 120]
        m3 = re.search(r"([A-Z0-9]{8,10})", context, flags=re.IGNORECASE)
        if m3:
            cand = m3.group(1)
            return True, re.sub(r"\W", "", cand).upper(), cand
    return False, None, None

def detect_aadhaar_number(full_text: str):
    return detect_12_digit_number(full_text)

def detect_bank_account_number_11(full_text: str):
    if not full_text:
        return False, None, None
    m = re.search(r"\b(\d{11})\b", full_text)
    if m:
        return True, m.group(1), m.group(1)
    m2 = re.search(r"\b(\d{3}[-\s]?\d{4}[-\s]?\d{4})\b", full_text)
    if m2:
        matched = m2.group(1)
        normalized = re.sub(r"\D", "", matched)
        if len(normalized) == 11:
            return True, normalized, matched
    m3 = re.search(r"((?:\d{2,5}\D?){2,4}\d{2,5})", full_text)
    if m3:
        cand = re.sub(r"\D", "", m3.group(1))
        if len(cand) == 11:
            return True, cand, m3.group(1)
    return False, None, None

def detect_aadhaar_components(full_text: str):
    if not full_text:
        return {
            "aadhaar_found": False, "aadhaar_value": None,
            "male_found": False,
            "dob_found": False, "dob_text": None,
            "gov_found": False,
            "unique_auth_found": False,
            "address_found": False,
            "relation_found": False
        }
    txt = full_text
    aadhaar_found, aadhaar_value, aadhaar_text = detect_12_digit_number(txt)
    male_found = False
    male_val = None
    if re.search(r"\bMale\b", txt, flags=re.IGNORECASE):
        male_found = True
        male_val = "Male"
    else:
        m_gender_m = re.search(r"\b(?:Gender|Sex)[:\s]*M\b", txt, flags=re.IGNORECASE)
        if m_gender_m:
            male_found = True
            male_val = "M"
    dob_found = False
    dob_text = None
    if re.search(r"\bDate of Birth\b|\bDOB\b", txt, flags=re.IGNORECASE):
        dob_found = True
        m_date = re.search(r"(\b\d{1,2}[\/\-\s]\d{1,2}[\/\-\s]\d{2,4}\b|\b\d{4}[\/\-\s]\d{1,2}[\/\-\s]\d{1,1}\b)", txt)
        if m_date:
            dob_text = m_date.group(1)
    else:
        m_date2 = re.search(r"(\b\d{1,2}[\/\-\s](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|[0-9]{1,2})[\/\-\s]\d{2,4}\b|\b\d{1,2}[\/\-\s]\d{1,2}[\/\-\s]\d{2,4}\b)", txt, flags=re.IGNORECASE)
        if m_date2:
            dob_found = True
            dob_text = m_date2.group(1)
    gov_found = bool(re.search(r"Government of India|GOVERNMENT OF INDIA", txt, flags=re.IGNORECASE))
    unique_auth_found = bool(re.search(r"Unique Identification Authority|UIDAI|Unique Identification Authority of India", txt, flags=re.IGNORECASE))
    address_found = bool(re.search(r"\bAddress\b|\bPermanent Address\b|\bAddress :", txt, flags=re.IGNORECASE))
    relation_found = bool(re.search(r"\bS\/O\b|\bSO\b|\bD\/O\b|\bDO\b|\bW\/O\b|\bWO\b|\bson of\b|\bwife of\b|\bdaughter of\b", txt, flags=re.IGNORECASE))
    return {
        "aadhaar_found": bool(aadhaar_found),
        "aadhaar_value": aadhaar_value or aadhaar_text,
        "male_found": bool(male_found),
        "male_value": male_val,
        "dob_found": bool(dob_found),
        "dob_text": dob_text,
        "gov_found": bool(gov_found),
        "unique_auth_found": bool(unique_auth_found),
        "address_found": bool(address_found),
        "relation_found": bool(relation_found)
    }
# ----------------------------------------------------------------------


def extract_texts_from_ocr_result(result):
    def extract(obj):
        texts = []
        if obj is None:
            return texts
        if isinstance(obj, str):
            s = obj.strip()
            if s:
                texts.append(s)
            return texts
        if isinstance(obj, dict):
            for key in ("text", "rec_text", "transcription", "sentence", "text_line"):
                if key in obj and isinstance(obj[key], str) and obj[key].strip():
                    texts.append(obj[key].strip())
            for v in obj.values():
                texts.extend(extract(v))
            return texts
        if isinstance(obj, (list, tuple)):
            for el in obj:
                texts.extend(extract(el))
            return texts
        return texts

    all_texts = extract(result)
    filtered = [t for t in all_texts if any(ch.isalnum() for ch in t)]
    seen = set()
    out = []
    for t in filtered:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _clean_pattern_to_plaintext(pat: str) -> str:
    if not pat:
        return ""
    cleaned = re.sub(r'\\b', ' ', pat)
    cleaned = re.sub(r'[\^\$\.\*\+\?\(\)\[\]\{\}\\\|]', ' ', cleaned)
    cleaned = re.sub(r'[^A-Za-z0-9 ]+', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def classify_document(full_text: str, ocr_lines: list = None):
    results = {}
    text_for_matching = full_text or ""
    ocr_lines = ocr_lines or (text_for_matching.splitlines() if text_for_matching else [])

    # Basic detectors once
    digit12_found, digit12_value, digit12_text = detect_12_digit_number(text_for_matching)
    pan_found, pan_norm, pan_orig = detect_pan_number(text_for_matching)
    voter_found, voter_norm, voter_orig = detect_voterid_number(text_for_matching)
    acc_found, acc_norm, acc_orig = detect_bank_account_number_11(text_for_matching)
    aadhaar_components = detect_aadhaar_components(text_for_matching)

    full_text_norm = re.sub(r'\s+', ' ', text_for_matching).strip()

    for dtype, template in DOCUMENT_TEMPLATES.items():
        score = 0
        matched_patterns = []
        patterns = template.get("patterns", []) if isinstance(template, dict) else list(template)

        for pat in patterns:
            # regex first
            try:
                m = re.search(pat, text_for_matching, flags=re.IGNORECASE)
            except re.error:
                m = None
            if m:
                score += 1
                matched_patterns.append({"pattern": pat, "match": m.group(0), "method": "regex", "score": None})
                continue

            # fuzzy fallback using similarity() against OCR lines and full_text
            plain = _clean_pattern_to_plaintext(pat)
            if plain:
                best_score = 0.0
                best_line = None
                # check each line
                for ln in ocr_lines:
                    try:
                        sim = similarity(plain.lower(), ln.lower())
                    except Exception:
                        sim = 0.0
                    if sim > best_score:
                        best_score = sim
                        best_line = ln
                        if best_score >= FUZZY_THRESHOLD:
                            break
                # compare to full_text too if not yet matched
                if best_score < FUZZY_THRESHOLD and full_text_norm:
                    try:
                        sim_full = similarity(plain.lower(), full_text_norm.lower())
                    except Exception:
                        sim_full = 0.0
                    if sim_full > best_score:
                        best_score = sim_full
                        best_line = full_text_norm
                if best_score >= FUZZY_THRESHOLD:
                    score += 1
                    matched_patterns.append({"pattern": pat, "match": best_line, "method": "fuzzy", "score": int(best_score)})

        # apply bonuses
        if template.get("pan_bonus") and pan_found:
            score += template.get("pan_bonus", 0)
            matched_patterns.append({"pattern": "PAN_NUMBER", "match": pan_norm, "method": "detector", "score": None})
        if template.get("digit12_bonus") and digit12_found:
            score += template.get("digit12_bonus", 0)
            matched_patterns.append({"pattern": "AADHAAR_NUMBER", "match": digit12_value, "method": "detector", "score": None})
        if template.get("voterid_bonus") and voter_found:
            score += template.get("voterid_bonus", 0)
            matched_patterns.append({"pattern": "VOTERID_NUMBER", "match": voter_norm, "method": "detector", "score": None})
        if template.get("account11_bonus") and acc_found:
            score += template.get("account11_bonus", 0)
            matched_patterns.append({"pattern": "ACCOUNT11_NUMBER", "match": acc_norm, "method": "detector", "score": None})

        results[dtype] = {
            "score": score,
            "matched_patterns": matched_patterns
        }

    # Evaluate candidates — must meet min_score and checks to be acceptable
    selection_scores = {}
    for dtype, info in results.items():
        score = info["score"]
        rules = DOCUMENT_RULES.get(dtype, {})
        checks = rules.get("checks", [])
        checks_pass = True
        for chk in checks:
            if chk == "aadhaar_number":
                found, _, _ = detect_aadhaar_number(text_for_matching)
                if not found:
                    checks_pass = False
            elif chk == "pan_number":
                found, _, _ = detect_pan_number(text_for_matching)
                if not found:
                    checks_pass = False
            elif chk == "voterid_number":
                found, _, _ = detect_voterid_number(text_for_matching)
                if not found:
                    checks_pass = False
            elif chk == "bank_account_number":
                found, _, _ = detect_bank_account_number_11(text_for_matching)
                if not found:
                    checks_pass = False
            else:
                checks_pass = False

        min_score = rules.get("min_score", 0)
        acceptable = (score >= min_score) and checks_pass
        selection_scores[dtype] = {
            "score": score,
            "acceptable": acceptable,
            "checks_pass": checks_pass,
            "min_score": min_score,
            "matched_patterns": info["matched_patterns"]
        }

    acceptable_candidates = [(d, v["score"]) for d, v in selection_scores.items() if v["acceptable"]]
    if acceptable_candidates:
        best = max(acceptable_candidates, key=lambda x: x[1])[0]
    else:
        best = "UNKNOWN"

    # build final classification details:
    pan_found2, pan_norm2, pan_orig2 = detect_pan_number(text_for_matching)
    aad_found2, aad_norm2, aad_orig2 = detect_aadhaar_number(text_for_matching)
    voter_found2, voter_norm2, voter_orig2 = detect_voterid_number(text_for_matching)
    acc_found2, acc_norm2, acc_orig2 = detect_bank_account_number_11(text_for_matching)
    aadhaar_components2 = detect_aadhaar_components(text_for_matching)

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

    # additional flags and aadhar front/back detection
    ac = aadhaar_components2 or {}
    aadhaar_present = bool(ac.get("aadhaar_found"))
    male_present = bool(ac.get("male_found"))
    dob_present = bool(ac.get("dob_found"))
    gov_present = bool(ac.get("gov_found"))
    unique_auth_present = bool(ac.get("unique_auth_found"))
    address_present = bool(ac.get("address_found"))
    relation_present = bool(ac.get("relation_found"))

    front_components = [dob_present, gov_present, aadhaar_present, male_present]
    back_components = [unique_auth_present, address_present, relation_present]
    front_count = sum(1 for x in front_components if bool(x))
    back_count = sum(1 for x in back_components if bool(x))

    front_detected = front_count >= 2
    back_detected = back_count >= 1
    both_detected = front_detected and back_detected

    result["aadhaar_front_count"] = front_count
    result["aadhaar_back_count"] = back_count
    result["aadhaar_front_detected"] = bool(front_detected)
    result["aadhaar_back_detected"] = bool(back_detected)
    result["aadhaar_both_detected"] = bool(both_detected)

    # expose checks if requested
    doc_rule = DOCUMENT_RULES.get(result["best_type"], {})
    checks = doc_rule.get("checks", [])
    if "aadhaar_number" in checks:
        found, normalized, matched_text = detect_aadhaar_number(text_for_matching)
        result["aadhaar_number_found"] = bool(found)
        result["aadhaar_number"] = normalized
        result["aadhaar_number_text"] = matched_text
    if "pan_number" in checks:
        found, normalized, matched_text = detect_pan_number(text_for_matching)
        result["pan_number_found"] = bool(found)
        result["pan_number"] = normalized
        result["pan_number_text"] = matched_text
    if "voterid_number" in checks:
        found, normalized, matched_text = detect_voterid_number(text_for_matching)
        result["voterid_number_found"] = bool(found)
        result["voterid_number"] = normalized
        result["voterid_number_text"] = matched_text
    if "bank_account_number" in checks:
        found, normalized, matched_text = detect_bank_account_number_11(text_for_matching)
        result["bank_account_number_found"] = bool(found)
        result["bank_account_number"] = normalized
        result["bank_account_number_text"] = matched_text

    return result


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    f = request.files["file"]
    if f.filename == "" or not allowed_file(f.filename):
        return jsonify({"error": "bad filename or file type"}), 400

    name = secure_filename(f.filename)
    base = os.path.splitext(name)[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_txt_path = os.path.join("Output", f"{base}_{ts}.txt")
    out_json_path = os.path.join("Output", f"{base}_{ts}.json")

    img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    img = read_image_bytes(img_bytes)

    # preprocess (deskew-first + blur handling)
    processed, angle = preprocess_using_tempfile_and_user_logic(img)

    tmp_path = os.path.join("Output", f"tmp_{base}_{ts}.png")
    cv2.imwrite(tmp_path, processed)

    # run paddleocr (support .predict or .ocr forms)
    try:
        if hasattr(ocr, "predict"):
            ocr_raw = ocr.predict(tmp_path)
        else:
            ocr_raw = ocr.ocr(tmp_path)
    except Exception as e:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return jsonify({"error": "paddleocr failed", "detail": str(e)}), 500

    lines = extract_texts_from_ocr_result(ocr_raw)
    full_text = "\n".join(lines)

    try:
        with open(out_txt_path, "w", encoding="utf-8") as outf:
            outf.write(full_text)
    except Exception as e:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return jsonify({"error": "failed to write output txt", "detail": str(e)}), 500

    # classify using fuzzy similarity and saved OCR lines
    classification = classify_document(full_text, ocr_lines=lines)

    json_payload = {
        "filename": name,
        "timestamp": ts,
        "image_base64": img_b64,
        "detected_text": full_text,
        "lines": lines,
        "deskew_angle": angle,
        "classification": classification
    }

    try:
        with open(out_json_path, "w", encoding="utf-8") as jf:
            json.dump(json_payload, jf, indent=2, ensure_ascii=False)
    except Exception as e:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return jsonify({"error": "failed to write output json", "detail": str(e)}), 500

    try:
        os.remove(tmp_path)
    except Exception:
        pass

    return jsonify({
        "output_txt": out_txt_path,
        "output_json": out_json_path,
        "lines_count": len(lines),
        "deskew_angle": angle,
        "detected_doc_type": classification.get("best_type"),
        "scores": classification.get("scores"),
        "aadhaar_front_detected": classification.get("aadhaar_front_detected"),
        "aadhaar_back_detected": classification.get("aadhaar_back_detected"),
        "aadhaar_both_detected": classification.get("aadhaar_both_detected")
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
