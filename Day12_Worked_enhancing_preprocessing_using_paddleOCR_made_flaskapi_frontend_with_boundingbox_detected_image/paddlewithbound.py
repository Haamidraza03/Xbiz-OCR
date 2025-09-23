import os
import io
import re
import json
import base64
import tempfile
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

from typing import List, Optional, Dict, Any, Tuple

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

# create OCR subfolders
OCR_IMAGE_DIR = os.path.join("Output", "OCR_IMAGE")
OCR_SORTED_DIR = os.path.join("Output", "OCR_SORTED")
OCR_TEXT_DIR = os.path.join("Output", "OCR_TEXT")
os.makedirs(OCR_IMAGE_DIR, exist_ok=True)
os.makedirs(OCR_SORTED_DIR, exist_ok=True)
os.makedirs(OCR_TEXT_DIR, exist_ok=True)

# Initialize PaddleOCR (v3.x uses .predict()).
# rec_batch_num is batch size for recognition stage (helps speed when multiple crops/images)
ocr = PaddleOCR(use_angle_cls=True, lang="en", rec_batch_num=16)

app = Flask(__name__)

# ---------- TUNABLE: blur detection and sharpening params ----------
BLUR_THRESHOLD = 100.0   # Laplacian variance below this considered "blurry"
UPSCALE_FACTOR_BLUR = 1.6
UNSHARP_WEIGHT = 1.5
UNSHARP_BLUR_WEIGHT = -0.5
UNSHARP_GAUSSIAN_SIGMA = 3
# ------------------------------------------------------------------


# ---------------- Document templates & rules (unchanged) -----------------------
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
def preprocess_using_tempfile_and_user_logic(img_bgr: np.ndarray, target_width: int = 1200) -> Tuple[np.ndarray, float, np.ndarray]:
    orig_tmp = None
    proc_tmp = None
    deskewed_color = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t:
            orig_tmp = t.name
            cv2.imwrite(orig_tmp, img_bgr)

        img = cv2.imread(orig_tmp, cv2.IMREAD_COLOR)
        if img is None:
            return img_bgr, 0.0, img_bgr

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
        # Keep deskewed_color after denoise
        try:
            # Gaussian blur on color image
            blur = cv2.GaussianBlur(img, (5, 5), 2)
            # unsharp/sharpen
            sharp = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
            # colored denoising
            denoised = cv2.fastNlMeansDenoisingColored(sharp, None, 6, 10, 9, 21)
            deskewed_color = denoised.copy()
            # to grayscale
            gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
            # background via median blur
            bg = cv2.medianBlur(gray, 31)
            
            # normalize by dividing with background and scaling
            norm = cv2.divide(gray.astype("float32"), bg, scale=255)


            # write out processed image to a temp file for downstream OCR
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t2:
                proc_tmp = t2.name
                cv2.imwrite(proc_tmp, norm)

            return norm, float(angle), deskewed_color
        except Exception:
            # fall back to returning grayscale of rotated image if something fails
            try:
                gray_fb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                return gray_fb, float(angle), img
            except Exception:
                return img, float(angle), img
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
            return gray_fb, 0.0, img_bgr
        except Exception:
            return img_bgr, 0.0, img_bgr
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


def extract_texts_from_ocr_result(result) -> List[str]:
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


def classify_document(full_text: str, ocr_lines: Optional[List[str]] = None) -> Dict[str, Any]:
    # same implementation as before (unchanged) - kept compact here
    results = {}
    text_for_matching = full_text or ""
    ocr_lines = ocr_lines or (text_for_matching.splitlines() if text_for_matching else [])

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
            try:
                m = re.search(pat, text_for_matching, flags=re.IGNORECASE)
            except re.error:
                m = None
            if m:
                score += 1
                matched_patterns.append({"pattern": pat, "match": m.group(0), "method": "regex", "score": None})
                continue
            plain = _clean_pattern_to_plaintext(pat)
            if plain:
                best_score = 0.0
                best_line = None
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
        results[dtype] = {"score": score, "matched_patterns": matched_patterns}

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
        selection_scores[dtype] = {"score": score, "acceptable": acceptable, "checks_pass": checks_pass, "min_score": min_score, "matched_patterns": info["matched_patterns"]}

    acceptable_candidates = [(d, v["score"]) for d, v in selection_scores.items() if v["acceptable"]]
    best = max(acceptable_candidates, key=lambda x: x[1])[0] if acceptable_candidates else "UNKNOWN"

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

    result["aadhaar_front_count"] = front_count
    result["aadhaar_back_count"] = back_count
    result["aadhaar_front_detected"] = bool(front_count >= 2)
    result["aadhaar_back_detected"] = bool(back_count >= 1)
    result["aadhaar_both_detected"] = bool(front_count >= 2 and back_count >= 1)

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


# ---------- FIXED: paddleocr -> words+boxes extractor ----------
def extract_words_and_boxes_from_paddle(ocr_raw) -> List[Dict[str, Any]]:
    """
    Return list of dicts: {"text": str, "vertices": [[x,y],...], "confidence": float|None}
    Handles the specific PaddleOCR output format from your example.
    """
    out: List[Dict[str, Any]] = []
    
    if not ocr_raw:
        return out

    print(f"DEBUG: OCR raw type: {type(ocr_raw)}")
    if isinstance(ocr_raw, list) and ocr_raw:
        print(f"DEBUG: First element type: {type(ocr_raw[0])}")
        if isinstance(ocr_raw[0], dict):
            print(f"DEBUG: First element keys: {ocr_raw[0].keys()}")

    # Handle your specific PaddleOCR output format
    if isinstance(ocr_raw, list) and len(ocr_raw) > 0 and isinstance(ocr_raw[0], dict):
        result_dict = ocr_raw[0]
        
        # Extract texts and boxes from rec_texts and rec_polys
        texts = result_dict.get('rec_texts', [])
        boxes = result_dict.get('rec_polys', [])
        scores = result_dict.get('rec_scores', [])
        
        print(f"DEBUG: Found {len(texts)} texts, {len(boxes)} boxes, {len(scores)} scores")
        
        # Process each detected text element
        for i, text in enumerate(texts):
            if i < len(boxes):
                box = boxes[i]
                confidence = scores[i] if i < len(scores) else None
                
                # Convert numpy array to list of vertices
                vertices = []
                try:
                    if hasattr(box, 'tolist'):
                        box_list = box.tolist()
                    else:
                        box_list = box
                    
                    if isinstance(box_list, list) and len(box_list) >= 4:
                        for point in box_list:
                            if isinstance(point, (list, tuple)) and len(point) >= 2:
                                x = int(round(float(point[0])))
                                y = int(round(float(point[1])))
                                vertices.append([x, y])
                    elif hasattr(box, 'shape') and box.shape == (4, 2):
                        # Handle numpy array with shape (4, 2)
                        for j in range(4):
                            x = int(round(float(box[j, 0])))
                            y = int(round(float(box[j, 1])))
                            vertices.append([x, y])
                except Exception as e:
                    print(f"DEBUG: Error processing box {i}: {e}")
                    continue
                
                if vertices and text:
                    out.append({
                        "text": str(text).strip(),
                        "vertices": vertices,
                        "confidence": float(confidence) if confidence is not None else None
                    })
                    print(f"DEBUG: Added text: '{text}' with {len(vertices)} vertices")
    
    # Fallback: try to extract from typical PaddleOCR format
    if not out:
        def extract_from_nested(obj, results):
            if isinstance(obj, list):
                for item in obj:
                    extract_from_nested(item, results)
            elif isinstance(obj, dict):
                # Check for typical PaddleOCR structure
                if 'rec_texts' in obj and 'rec_polys' in obj:
                    texts = obj.get('rec_texts', [])
                    boxes = obj.get('rec_polys', [])
                    scores = obj.get('rec_scores', [])
                    
                    for i, text in enumerate(texts):
                        if i < len(boxes):
                            box = boxes[i]
                            confidence = scores[i] if i < len(scores) else None
                            
                            vertices = []
                            try:
                                if hasattr(box, 'tolist'):
                                    box_points = box.tolist()
                                else:
                                    box_points = box
                                
                                for point in box_points:
                                    if isinstance(point, (list, tuple)) and len(point) >= 2:
                                        x = int(round(float(point[0])))
                                        y = int(round(float(point[1])))
                                        vertices.append([x, y])
                            except Exception:
                                continue
                            
                            if vertices and text:
                                results.append({
                                    "text": str(text).strip(),
                                    "vertices": vertices,
                                    "confidence": float(confidence) if confidence is not None else None
                                })
                
                # Recursively search nested dictionaries
                for value in obj.values():
                    extract_from_nested(value, results)
        
        extract_from_nested(ocr_raw, out)
    
    print(f"DEBUG: Extracted {len(out)} words")
    return out


# ---------- grouping words into lines and sorting (fixed ordering) ----------
def _centroid(vertices):
    if not vertices:
        return (0.0, 0.0)
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    return (sum(xs) / len(xs), sum(ys) / len(ys))

def group_words_to_lines(words: List[Dict[str, Any]], tolerance_multiplier=0.6, min_tolerance_px=10, use_baseline=True) -> List[Tuple[float, List[Tuple[float, str]]]]:
    """
    words: list of {"text","vertices","confidence"}
    returns ordered list of tuples: (line_center_y, [(x_left, text), ...]) sorted top-to-bottom.
    """
    items = []
    for w in words:
        verts = w.get("vertices") or []
        if not verts:
            continue
        try:
            xs = [float(p[0]) for p in verts]
            ys = [float(p[1]) for p in verts]
            x_left = min(xs)
            y_top = min(ys)
            y_bottom = max(ys)
            height = max(1.0, y_bottom - y_top)
            items.append({
                "text": w.get("text", ""), 
                "x_left": float(x_left), 
                "y_center": (y_top + y_bottom) / 2.0, 
                "y_top": y_top, 
                "y_bottom": y_bottom, 
                "height": float(height), 
                "centroid": _centroid(verts)
            })
        except Exception as e:
            print(f"DEBUG: Error processing word vertices: {e}")
            continue

    if not items:
        print("DEBUG: No valid items found for line grouping")
        return []

    try:
        heights = [it["height"] for it in items if it["height"] > 0]
        median_h = float(np.median(heights)) if heights else 12.0
        tol = max(min_tolerance_px, int(round(median_h * float(tolerance_multiplier))))

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
                lines.append({
                    "center": it["y_center"], 
                    "y_min": it["y_top"], 
                    "y_max": it["y_bottom"], 
                    "items": [it]
                })

        # Build final structure and sort
        final = []
        for line in lines:
            li = line["items"]
            li_sorted = sorted([(it["x_left"], it["text"]) for it in li], key=lambda t: t[0])
            final.append((line["center"], li_sorted))
        # sort by center (top -> bottom)
        final_sorted = sorted(final, key=lambda t: t[0])
        print(f"DEBUG: Grouped into {len(final_sorted)} lines")
        return final_sorted
    except Exception as e:
        print(f"DEBUG: Error in line grouping: {e}")
        return []

def create_sorted_text_file_from_words(words: List[Dict[str, Any]], output_path: str, tolerance_multiplier=0.6, min_tolerance_px: int = 10, use_baseline: bool = True) -> str:
    try:
        lines = group_words_to_lines(words, tolerance_multiplier=tolerance_multiplier, min_tolerance_px=min_tolerance_px, use_baseline=use_baseline)
        with open(output_path, "w", encoding="utf-8") as fh:
            for _, elems in lines:
                line_text = " ".join(t for _, t in elems).strip()
                if line_text:  # Only write non-empty lines
                    fh.write(line_text + "\n")
        print(f"DEBUG: Saved sorted text to {output_path} with {len(lines)} lines")
        return output_path
    except Exception as e:
        print(f"DEBUG: Error creating sorted text file: {e}")
        # Fallback: write raw text
        try:
            with open(output_path, "w", encoding="utf-8") as fh:
                for w in words:
                    text = w.get("text", "").strip()
                    if text:
                        fh.write(text + "\n")
            return output_path
        except Exception:
            return output_path


# ---------- Improved draw bounding boxes + translucent label ----------
def draw_annotations(img: np.ndarray, texts: List[str], boxes: List[List[List[int]]], out_path: str, confidences: Optional[List[float]] = None, color=(0,200,0)) -> str:
    """
    img: BGR numpy array
    texts: list of strings
    boxes: list of polygons (list of [x,y]) or [] placeholders
    confidences: list of floats or None
    """
    if len(img.shape) == 2:  # If grayscale, convert to BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_draw = img.copy()
    h, w = img.shape[:2]

    left_margin_x = 8
    left_stack_y = 8
    left_stack_step = 18

    any_drawn = False
    font_scale = max(0.25, min(0.45, w / 1200.0))
    text_thickness = max(1, int(round(font_scale * 2)))

    print(f"DEBUG: Drawing {len(texts)} texts, {len(boxes)} boxes")

    for i, text in enumerate(texts):
        bbox = boxes[i] if i < len(boxes) else None
        conf = None
        if confidences and i < len(confidences):
            try:
                conf = float(confidences[i]) if confidences[i] is not None else None
            except Exception:
                conf = None

        label_parts = [f"#{i}"]
        if conf is not None:
            label_parts.append(f"{conf:.2f}")
        label_parts.append(str(text))
        label = " ".join(label_parts)
        if len(label) > 200:
            label = label[:197] + "..."

        (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
        pad_x, pad_y = 6, 3

        if bbox and len(bbox) >= 3:  # Need at least 3 points for a polygon
            try:
                pts = np.array(bbox, dtype=np.int32)
                # Draw the bounding polygon
                cv2.polylines(img_draw, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
                
                # Get bounding rect for text background
                bx, by, bw_box, bh_box = cv2.boundingRect(pts)
                
                # Position text label inside the bounding box at top-left
                rect_x1 = max(0, bx)
                rect_y1 = max(0, by)
                rect_x2 = min(w - 1, rect_x1 + max(bw_box, w_text + 2 * pad_x))
                rect_y2 = min(h - 1, rect_y1 + h_text + 2 * pad_y)
                
                # If not enough height at top, try bottom inside
                if rect_y2 - rect_y1 > bh_box:
                    rect_y1 = max(0, by + bh_box - h_text - 2 * pad_y)
                    rect_y2 = min(h - 1, by + bh_box)
                
                # Create semi-transparent background for text
                overlay = img_draw.copy()
                cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), color, -1)
                alpha = 0.65
                cv2.addWeighted(overlay, alpha, img_draw, 1 - alpha, 0, img_draw)
                
                # Draw text (y is bottom of text)
                text_color = (0, 0, 0) if (color[0] + color[1] + color[2]) > 300 else (255, 255, 255)
                text_y = rect_y1 + h_text + pad_y  # Place at top with padding
                cv2.putText(img_draw, label, (rect_x1 + pad_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, text_thickness, cv2.LINE_AA)
                any_drawn = True
                print(f"DEBUG: Drawn box for '{text}'")
                
            except Exception as e:
                print(f"DEBUG: Error drawing box {i}: {e}")
                # Fall through to left column placement

        # If no valid box or error, use left column fallback
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
            alpha = 0.65
            cv2.addWeighted(overlay, alpha, img_draw, 1 - alpha, 0, img_draw)
            
            text_color = (0, 0, 0) if (color[0] + color[1] + color[2]) > 300 else (255, 255, 255)
            cv2.putText(img_draw, label, (rect_x1 + pad_x, rect_y2 - pad_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, text_thickness, cv2.LINE_AA)
            left_stack_y += step
            any_drawn = True
            print(f"DEBUG: Drawn left column for '{text}'")

    if not any_drawn:
        cv2.putText(img_draw, "NO DETECTIONS", (20, max(50, h // 2)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3, cv2.LINE_AA)
        print("DEBUG: No detections drawn")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, img_draw)
    print(f"DEBUG: Saved annotated image to {out_path}")
    return out_path


# ------------------ main upload endpoint (integrates everything) ------------------
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
    processed_gray, angle, processed_color = preprocess_using_tempfile_and_user_logic(img)

    tmp_path = os.path.join("Output", f"tmp_{base}_{ts}.png")
    cv2.imwrite(tmp_path, processed_gray)

    # run paddleocr (support .predict or .ocr forms)
    try:
        if hasattr(ocr, "predict"):
            ocr_raw = ocr.predict(tmp_path)
        else:
            ocr_raw = ocr.ocr(tmp_path)
        print(f"DEBUG: OCR completed, result type: {type(ocr_raw)}")
    except Exception as e:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return jsonify({"error": "paddleocr failed", "detail": str(e)}), 500

    # extract plain lines/text
    lines = extract_texts_from_ocr_result(ocr_raw)
    full_text = "\n".join(lines)
    print(f"DEBUG: Extracted {len(lines)} text lines")

    # --- save plain detected text into Output root AND OCR_TEXT folder ---
    txt_detected_path = os.path.join(OCR_TEXT_DIR, f"{base}_{ts}.txt")
    try:
        # original file (kept for backward compatibility)
        with open(out_txt_path, "w", encoding="utf-8") as outf:
            outf.write(full_text)
        # OCR_TEXT file
        with open(txt_detected_path, "w", encoding="utf-8") as outf:
            outf.write(full_text)
    except Exception as e:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return jsonify({"error": "failed to write output txt", "detail": str(e)}), 500

    # classify using fuzzy similarity and saved OCR lines
    classification = classify_document(full_text, ocr_lines=lines)

    # --- additional: extract words+boxes and save sorted text & annotated image ---
    words = extract_words_and_boxes_from_paddle(ocr_raw)
    print(f"DEBUG: Extracted {len(words)} words with bounding boxes")

    # create sorted text file under OCR_SORTED
    sorted_txt_path = os.path.join(OCR_SORTED_DIR, f"{base}_{ts}.txt")
    try:
        create_sorted_text_file_from_words(words, sorted_txt_path, tolerance_multiplier=0.6, min_tolerance_px=10, use_baseline=True)
    except Exception as e:
        print(f"DEBUG: Error creating sorted file: {e}")
        # fallback: write the naive line list
        try:
            with open(sorted_txt_path, "w", encoding="utf-8") as sf:
                for ln in lines:
                    sf.write(ln + "\n")
        except Exception:
            pass

    # draw annotated image into OCR_IMAGE on processed_color
    texts_for_draw = [w["text"] for w in words]
    boxes_for_draw = [w["vertices"] for w in words]
    confidences = [w.get("confidence") for w in words]
    annotated_image_path = os.path.join(OCR_IMAGE_DIR, f"{base}_{ts}.png")
    try:
        draw_annotations(processed_color, texts_for_draw, boxes_for_draw, annotated_image_path, confidences=confidences)
    except Exception as e:
        print(f"DEBUG: Error drawing annotations: {e}")
        # fallback: copy processed_color
        try:
            cv2.imwrite(annotated_image_path, processed_color)
        except Exception:
            pass

    # For frontend: base64 of annotated image
    with open(annotated_image_path, "rb") as ann_f:
        ann_b64 = base64.b64encode(ann_f.read()).decode("utf-8")

    json_payload = {
        "filename": name,
        "timestamp": ts,
        "image_base64": img_b64,
        "detected_text": full_text,
        "lines": lines,
        "deskew_angle": angle,
        "classification": classification,
        "words_count": len(words),
        "boxes_count": len([w for w in words if w.get("vertices")]),
        "annotated_image_path": annotated_image_path,
        "sorted_text_path": sorted_txt_path  # include for debugging and frontend lazy fetch
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
        "words_count": len(words),
        "deskew_angle": angle,
        "detected_doc_type": classification.get("best_type"),
        "scores": classification.get("scores"),
        "aadhaar_front_detected": classification.get("aadhaar_front_detected"),
        "aadhaar_back_detected": classification.get("aadhaar_back_detected"),
        "aadhaar_both_detected": classification.get("aadhaar_both_detected"),
        # new debug fields (local file locations)
        "ocr_text_file": txt_detected_path,
        "ocr_sorted_file": sorted_txt_path,
        "ocr_annotated_image": annotated_image_path,
        "annotated_image_base64": ann_b64,
        "sorted_text_path": sorted_txt_path  # ADDED
    })


# ------------------ detect endpoint for multiple files ------------------
@app.route("/detect", methods=["POST"])
def detect():
    files = request.files.getlist("file")
    if not files:
        return jsonify({"error": "no file"}), 400

    results = []
    for f in files:
        if f.filename == "" or not allowed_file(f.filename):
            continue

        name = secure_filename(f.filename)
        base = os.path.splitext(name)[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_txt_path = os.path.join("Output", f"{base}_{ts}.txt")
        out_json_path = os.path.join("Output", f"{base}_{ts}.json")

        img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        img = read_image_bytes(img_bytes)

        # preprocess (deskew-first + blur handling)
        processed_gray, angle, processed_color = preprocess_using_tempfile_and_user_logic(img)

        tmp_path = os.path.join("Output", f"tmp_{base}_{ts}.png")
        cv2.imwrite(tmp_path, processed_gray)

        # run paddleocr (support .predict or .ocr forms)
        try:
            if hasattr(ocr, "predict"):
                ocr_raw = ocr.predict(tmp_path)
            else:
                ocr_raw = ocr.ocr(tmp_path)
            print(f"DEBUG: OCR completed, result type: {type(ocr_raw)}")
        except Exception as e:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            continue

        # extract plain lines/text
        lines = extract_texts_from_ocr_result(ocr_raw)
        full_text = "\n".join(lines)
        print(f"DEBUG: Extracted {len(lines)} text lines")

        # --- save plain detected text into Output root AND OCR_TEXT folder ---
        txt_detected_path = os.path.join(OCR_TEXT_DIR, f"{base}_{ts}.txt")
        try:
            # original file (kept for backward compatibility)
            with open(out_txt_path, "w", encoding="utf-8") as outf:
                outf.write(full_text)
            # OCR_TEXT file
            with open(txt_detected_path, "w", encoding="utf-8") as outf:
                outf.write(full_text)
        except Exception as e:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            continue

        # classify using fuzzy similarity and saved OCR lines
        classification = classify_document(full_text, ocr_lines=lines)

        # --- additional: extract words+boxes and save sorted text & annotated image ---
        words = extract_words_and_boxes_from_paddle(ocr_raw)
        print(f"DEBUG: Extracted {len(words)} words with bounding boxes")

        # create sorted text file under OCR_SORTED
        sorted_txt_path = os.path.join(OCR_SORTED_DIR, f"{base}_{ts}.txt")
        try:
            create_sorted_text_file_from_words(words, sorted_txt_path, tolerance_multiplier=0.6, min_tolerance_px=10, use_baseline=True)
        except Exception as e:
            print(f"DEBUG: Error creating sorted file: {e}")
            # fallback: write the naive line list
            try:
                with open(sorted_txt_path, "w", encoding="utf-8") as sf:
                    for ln in lines:
                        sf.write(ln + "\n")
            except Exception:
                pass

        # draw annotated image into OCR_IMAGE on processed_color
        texts_for_draw = [w["text"] for w in words]
        boxes_for_draw = [w["vertices"] for w in words]
        confidences = [w.get("confidence") for w in words]
        annotated_image_path = os.path.join(OCR_IMAGE_DIR, f"{base}_{ts}.png")
        try:
            draw_annotations(processed_color, texts_for_draw, boxes_for_draw, annotated_image_path, confidences=confidences)
        except Exception as e:
            print(f"DEBUG: Error drawing annotations: {e}")
            # fallback: copy processed_color
            try:
                cv2.imwrite(annotated_image_path, processed_color)
            except Exception:
                pass

        # For frontend: base64 of annotated image
        with open(annotated_image_path, "rb") as ann_f:
            ann_b64 = base64.b64encode(ann_f.read()).decode("utf-8")

        json_payload = {
            "filename": name,
            "timestamp": ts,
            "image_base64": img_b64,
            "detected_text": full_text,
            "lines": lines,
            "deskew_angle": angle,
            "classification": classification,
            "words_count": len(words),
            "boxes_count": len([w for w in words if w.get("vertices")]),
            "annotated_image_path": annotated_image_path,
            "sorted_text_path": sorted_txt_path  # include sorted path here too
        }

        try:
            with open(out_json_path, "w", encoding="utf-8") as jf:
                json.dump(json_payload, jf, indent=2, ensure_ascii=False)
        except Exception as e:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            continue

        try:
            os.remove(tmp_path)
        except Exception:
            pass

        # Result for frontend (include sorted_text_path so frontend can lazy-load)
        result_dict = {
            "inputbase64": ann_b64,  # Annotated image base64
            "document_type": classification.get("best_type"),
            "cleaned_text": [],  # Empty
            "detected_text": lines,  # List of strings
            "sorted_text_path": sorted_txt_path  # ADDED
        }

        results.append({"original_name": name, "result": result_dict})

    if not results:
        return jsonify({"error": "no valid files"}), 400

    if len(results) == 1:
        return jsonify({"result": results[0]["result"]})
    else:
        return jsonify({"results": results})


# ------------------ detect all endpoint for images_multi folder ------------------
@app.route('/detect_all', methods=['GET'])
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
            # Process
            result = process_single_image(img_bytes, filename)
            if result:
                results.append({"original_name": filename, "result": result})

    if not results:
        return jsonify({"error": "no valid files in images_multi folder"}), 400

    return jsonify({"results": results})


def process_single_image(img_bytes, filename):
    name = secure_filename(filename)
    base = os.path.splitext(name)[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_txt_path = os.path.join("Output", f"{base}_{ts}.txt")
    out_json_path = os.path.join("Output", f"{base}_{ts}.json")

    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    img = read_image_bytes(img_bytes)

    # preprocess (deskew-first + blur handling)
    processed_gray, angle, processed_color = preprocess_using_tempfile_and_user_logic(img)

    tmp_path = os.path.join("Output", f"tmp_{base}_{ts}.png")
    cv2.imwrite(tmp_path, processed_gray)

    # run paddleocr (support .predict or .ocr forms)
    try:
        if hasattr(ocr, "predict"):
            ocr_raw = ocr.predict(tmp_path)
        else:
            ocr_raw = ocr.ocr(tmp_path)
        print(f"DEBUG: OCR completed, result type: {type(ocr_raw)}")
    except Exception as e:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return None

    # extract plain lines/text
    lines = extract_texts_from_ocr_result(ocr_raw)
    full_text = "\n".join(lines)
    print(f"DEBUG: Extracted {len(lines)} text lines")

    # --- save plain detected text into Output root AND OCR_TEXT folder ---
    txt_detected_path = os.path.join(OCR_TEXT_DIR, f"{base}_{ts}.txt")
    try:
        # original file (kept for backward compatibility)
        with open(out_txt_path, "w", encoding="utf-8") as outf:
            outf.write(full_text)
        # OCR_TEXT file
        with open(txt_detected_path, "w", encoding="utf-8") as outf:
            outf.write(full_text)
    except Exception as e:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return None

    # classify using fuzzy similarity and saved OCR lines
    classification = classify_document(full_text, ocr_lines=lines)

    # --- additional: extract words+boxes and save sorted text & annotated image ---
    words = extract_words_and_boxes_from_paddle(ocr_raw)
    print(f"DEBUG: Extracted {len(words)} words with bounding boxes")

    # create sorted text file under OCR_SORTED
    sorted_txt_path = os.path.join(OCR_SORTED_DIR, f"{base}_{ts}.txt")
    try:
        create_sorted_text_file_from_words(words, sorted_txt_path, tolerance_multiplier=0.6, min_tolerance_px=10, use_baseline=True)
    except Exception as e:
        print(f"DEBUG: Error creating sorted file: {e}")
        # fallback: write the naive line list
        try:
            with open(sorted_txt_path, "w", encoding="utf-8") as sf:
                for ln in lines:
                    sf.write(ln + "\n")
        except Exception:
            pass

    # draw annotated image into OCR_IMAGE on processed_color
    texts_for_draw = [w["text"] for w in words]
    boxes_for_draw = [w["vertices"] for w in words]
    confidences = [w.get("confidence") for w in words]
    annotated_image_path = os.path.join(OCR_IMAGE_DIR, f"{base}_{ts}.png")
    try:
        draw_annotations(processed_color, texts_for_draw, boxes_for_draw, annotated_image_path, confidences=confidences)
    except Exception as e:
        print(f"DEBUG: Error drawing annotations: {e}")
        # fallback: copy processed_color
        try:
            cv2.imwrite(annotated_image_path, processed_color)
        except Exception:
            pass

    # For frontend: base64 of annotated image
    with open(annotated_image_path, "rb") as ann_f:
        ann_b64 = base64.b64encode(ann_f.read()).decode("utf-8")

    json_payload = {
        "filename": name,
        "timestamp": ts,
        "image_base64": img_b64,
        "detected_text": full_text,
        "lines": lines,
        "deskew_angle": angle,
        "classification": classification,
        "words_count": len(words),
        "boxes_count": len([w for w in words if w.get("vertices")]),
        "annotated_image_path": annotated_image_path,
        "sorted_text_path": sorted_txt_path  # include here as well
    }

    try:
        with open(out_json_path, "w", encoding="utf-8") as jf:
            json.dump(json_payload, jf, indent=2, ensure_ascii=False)
    except Exception as e:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return None

    try:
        os.remove(tmp_path)
    except Exception:
        pass

    # Result for frontend
    result_dict = {
        "inputbase64": ann_b64,  # Annotated image base64
        "document_type": classification.get("best_type"),
        "cleaned_text": [],  # Empty
        "detected_text": lines,  # List of strings
        "sorted_text_path": sorted_txt_path  # ADDED
    }

    return result_dict


# ------------------ root endpoint to serve index.html ------------------
@app.route('/')
def index():
    return render_template('index.html')


# ------------------ NEW: serve sorted text file contents (secure) ------------------
@app.route("/get_sorted_text", methods=["POST"])
def get_sorted_text():
    """Endpoint to serve sorted text file content"""
    try:
        data = request.get_json(force=True)
        file_path = data.get('file_path', '') if isinstance(data, dict) else ''
        if not file_path:
            return jsonify({"error": "Invalid file path"}), 400

        # Normalize and make absolute to perform secure check
        requested = os.path.abspath(os.path.normpath(file_path))
        allowed_dir = os.path.abspath(os.path.normpath(OCR_SORTED_DIR))  # must be inside this folder

        # Security check: ensure the path is within allowed directory
        if not requested.startswith(allowed_dir + os.sep) and requested != allowed_dir:
            return jsonify({"error": "Invalid file path"}), 400

        if not os.path.exists(requested):
            return jsonify({"error": "File not found"}), 404

        with open(requested, 'r', encoding='utf-8') as f:
            content = f.read()

        return jsonify({"sorted_text": content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
