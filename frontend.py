import os
import re
import json
import time
import base64
import tempfile
import traceback
from flask import Flask, request, jsonify, render_template
from PIL import Image

app = Flask(__name__)

# ----------------- Fuzzy matching backend (rapidfuzz preferred) -----------------
FUZZY_THRESHOLD = int(os.environ.get("FUZZY_THRESHOLD", "75"))

try:
    from rapidfuzz import fuzz as _rf_fuzz
    from rapidfuzz import utils as _rf_utils

    def similarity(a: str, b: str) -> float:
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

    print("Using rapidfuzz for fuzzy matching.")
except Exception:
    print("rapidfuzz not available. Install with: pip install rapidfuzz")
    from difflib import SequenceMatcher as _SM

    def similarity(a: str, b: str) -> float:
        try:
            return float(_SM(None, a, b).ratio() * 100.0)
        except Exception:
            return 0.0

    print("Falling back to difflib for fuzzy matching (less accurate).")

# ----------------- Optional library imports & initialization -----------------
paddle_ocr = None
try:
    from paddleocr import PaddleOCR
    _paddle_use_gpu = os.environ.get("PADDLE_USE_GPU", "0") in ("1", "true", "True", "yes")
    _paddle_lang = os.environ.get("PADDLE_LANG", "en")
    _paddle_use_angle = os.environ.get("PADDLE_USE_ANGLE", "1") in ("1", "true", "True", "yes")
    try:
        paddle_ocr = PaddleOCR(use_angle_cls=_paddle_use_angle, lang=_paddle_lang, use_gpu=_paddle_use_gpu)
        print("PaddleOCR initialized (use_angle_cls).")
    except TypeError:
        try:
            paddle_ocr = PaddleOCR(use_textline_orientation=_paddle_use_angle, lang=_paddle_lang, use_gpu=_paddle_use_gpu)
            print("PaddleOCR initialized (use_textline_orientation).")
        except TypeError:
            try:
                paddle_ocr = PaddleOCR(lang=_paddle_lang, use_gpu=_paddle_use_gpu)
                print("PaddleOCR initialized (lang/use_gpu).")
            except Exception:
                paddle_ocr = PaddleOCR()
                print("PaddleOCR initialized (minimal constructor).")
except Exception:
    paddle_ocr = None
    print("PaddleOCR not available — will fallback to Tesseract if present.")

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

cv2 = None
np = None
try:
    import cv2
    import numpy as np
    print("OpenCV available for preprocessing.")
except Exception:
    cv2 = None
    np = None
    print("OpenCV (cv2) not installed; preprocessing will be skipped.")

# ----------------- Document templates & rules (unchanged) -----------------
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
    "AADHAAR_CARD": {"min_score": 3, "checks": ["aadhaar_number"]},
    "PAN_CARD": {"min_score": 3, "checks": ["pan_number"]},
    "DRIVING_LICENSE": {"min_score": 3},
    "VOTER_ID": {"min_score": 3, "checks": ["voterid_number"]},
    "BANK_STATEMENT": {"min_score": 4, "checks": ["bank_account_number"]}
}

# ----------------- Paths (kept but no writes performed) -----------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images_multi")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputweek2")
OUTPUT_WEEK2 = os.path.join(PROJECT_ROOT, "outputweek2")
OTHERS_DIR = os.path.join(OUTPUT_WEEK2, "Others")

# keep dirs if desired but we will NOT write any files now
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_WEEK2, exist_ok=True)
os.makedirs(OTHERS_DIR, exist_ok=True)

# ----------------- Helpers -----------------
def sanitize_filename(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]', '_', name)

def bytes_to_base64(image_bytes: bytes, fmt: str = "png") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/{fmt};base64,{b64}"

# ... detection helper functions (detect_12_digit_number, detect_pan_number, etc.)
# For brevity they are the same as before — include all helper functions here.
# (I will reuse your previous helper implementations verbatim.)
# -- start helper block --
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
        m_date = re.search(r"(\b\d{1,2}[\/\-\s]\d{1,2}[\/\-\s]\d{2,4}\b|\b\d{4}[\/\-\s]\d{1,2}[\/\-\s]\d{1,2}\b)", txt)
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

def _clean_pattern_to_plaintext(pat: str) -> str:
    if not pat:
        return ""
    cleaned = re.sub(r'\\b', ' ', pat)
    cleaned = re.sub(r'[\^\$\.\*\+\?\(\)\[\]\{\}\\\|]', ' ', cleaned)
    cleaned = re.sub(r'[^A-Za-z0-9 ]+', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

# get_template_scores and classify_document copied (verbatim) from your working source.
# -- start scoring/classify block --
def get_template_scores(full_text: str):
    matches = {}
    digit12_found, digit12_value, digit12_text = detect_12_digit_number(full_text)
    pan_found, pan_value, pan_text = detect_pan_number(full_text)
    voterid_found, voterid_value, voterid_text = detect_voterid_number(full_text)
    account11_found, account11_value, account11_text = detect_bank_account_number_11(full_text)
    aadhaar_components = detect_aadhaar_components(full_text)
    lines = []
    if full_text:
        lines = [ln.strip() for ln in re.split(r'[\r\n]+', full_text) if ln.strip()]
        if len(lines) < 5:
            more = re.split(r'[.,;:]', full_text)
            lines.extend([m.strip() for m in more if m.strip()])
        seen = set()
        normalized_lines = []
        for ln in lines:
            low = re.sub(r'\s+', ' ', ln).strip()
            if low and low not in seen:
                normalized_lines.append(low)
                seen.add(low)
        lines = normalized_lines
    full_text_norm = re.sub(r'\s+', ' ', (full_text or "")).strip()
    for doc_type, spec in DOCUMENT_TEMPLATES.items():
        digit12_bonus = 0
        pan_bonus = 0
        voterid_bonus = 0
        account11_bonus = 0
        patterns = spec.get("patterns", []) if isinstance(spec, dict) else list(spec)
        digit12_bonus = int(spec.get("digit12_bonus", 0) or 0) if isinstance(spec, dict) else 0
        pan_bonus = int(spec.get("pan_bonus", 0) or 0) if isinstance(spec, dict) else 0
        voterid_bonus = int(spec.get("voterid_bonus", 0) or 0) if isinstance(spec, dict) else 0
        account11_bonus = int(spec.get("account11_bonus", 0) or 0) if isinstance(spec, dict) else 0
        score = 0
        matched = []
        for pat in patterns:
            pat = pat or ""
            try:
                if re.search(pat, full_text, flags=re.IGNORECASE):
                    score += 1
                    matched.append(pat)
                    continue
            except re.error:
                pass
            try:
                plain = _clean_pattern_to_plaintext(pat)
                if plain:
                    best_sim = 0.0
                    for ln in lines:
                        sim = similarity(plain.lower(), ln.lower())
                        if sim > best_sim:
                            best_sim = sim
                            if best_sim >= FUZZY_THRESHOLD:
                                break
                    if best_sim < FUZZY_THRESHOLD and full_text_norm:
                        sim_full = similarity(plain.lower(), full_text_norm.lower())
                        if sim_full > best_sim:
                            best_sim = sim_full
                    if best_sim >= FUZZY_THRESHOLD:
                        score += 1
                        matched.append(f"FUZZY:{plain}({int(best_sim)})")
            except Exception:
                continue
        digit12_applied = False
        pan_applied = False
        voterid_applied = False
        account11_applied = False
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
        if account11_bonus and account11_found:
            score += account11_bonus
            account11_applied = True
            matched.append("11_digit_account_number")
        matches[doc_type] = {
            "score": score,
            "matched": matched,
            "digit12_bonus_applied": digit12_applied,
            "pan_bonus_applied": pan_applied,
            "voterid_bonus_applied": voterid_applied,
            "account11_bonus_applied": account11_applied
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
        "voterid_text": voterid_text,
        "account11_found": account11_found,
        "account11_value": account11_value,
        "account11_text": account11_text,
        "aadhaar_components": aadhaar_components
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
    result["bank_account_found"] = bool(extras.get("account11_found"))
    result["bank_account_number"] = extras.get("account11_value")
    result["bank_account_text"] = extras.get("account11_text")
    aadhaar_components = extras.get("aadhaar_components", {}) or {}
    aadhaar_found = bool(aadhaar_components.get("aadhaar_found"))
    aadhaar_value = aadhaar_components.get("aadhaar_value")
    male_found = bool(aadhaar_components.get("male_found"))
    male_value = aadhaar_components.get("male_value")
    dob_found = bool(aadhaar_components.get("dob_found"))
    dob_text = aadhaar_components.get("dob_text")
    gov_found = bool(aadhaar_components.get("gov_found"))
    unique_auth_found = bool(aadhaar_components.get("unique_auth_found"))
    address_found = bool(aadhaar_components.get("address_found"))
    relation_found = bool(aadhaar_components.get("relation_found"))
    result["aadhaar_found"] = aadhaar_found
    result["aadhaar_value"] = aadhaar_value
    result["male_found"] = male_found
    result["male_value"] = male_value
    result["dob_found"] = dob_found
    result["dob_text"] = dob_text
    result["gov_found"] = gov_found
    result["unique_auth_found"] = unique_auth_found
    result["address_found"] = address_found
    result["relation_found"] = relation_found
    front_components = [dob_found, gov_found, aadhaar_found, male_found]
    back_components = [unique_auth_found, address_found, relation_found]
    front_count = sum(1 for x in front_components if bool(x))
    back_count = sum(1 for x in back_components if bool(x))
    result["aadhaar_front_count"] = front_count
    result["aadhaar_back_count"] = back_count
    front_detected = False
    back_detected = False
    both_detected = False
    if front_count >= 2:
        front_detected = True
    if back_count >= 1:
        back_detected = True
    if front_count == back_count or front_count > back_count or back_count > front_count:
        both_detected = True
    result["aadhaar_front_detected"] = front_detected
    result["aadhaar_back_detected"] = back_detected
    result["aadhaar_both_detected"] = both_detected
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
    if "bank_account_number" in checks:
        found, normalized, matched_text = detect_bank_account_number_11(full_text)
        result["bank_account_number_found"] = bool(found)
        result["bank_account_number"] = normalized
        result["bank_account_number_text"] = matched_text
    return result
# -- end scoring/classify block --

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

# ----------------- OCR backends (with normalization) -----------------
def run_paddle_and_parse(image_path: str):
    if paddle_ocr is None:
        raise RuntimeError("PaddleOCR not available.")
    try:
        try:
            ocr_res = paddle_ocr.ocr(image_path, cls=True)
        except TypeError:
            ocr_res = paddle_ocr.ocr(image_path)
    except Exception as e:
        raise RuntimeError(f"PaddleOCR failed: {e}")
    parsed_lines = []
    full_text_parts = []
    for item in ocr_res:
        try:
            if not item:
                continue
            box = None
            text = None
            conf = None
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                box = item[0]
                text_conf = item[1]
                if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 1:
                    text = str(text_conf[0])
                    if len(text_conf) > 1:
                        try:
                            conf = float(text_conf[1])
                        except Exception:
                            conf = None
                elif isinstance(text_conf, dict):
                    text = str(text_conf.get("text", ""))
                    conf_val = text_conf.get("confidence", None)
                    try:
                        conf = float(conf_val) if conf_val is not None else None
                    except Exception:
                        conf = None
                else:
                    text = str(text_conf)
            elif isinstance(item, dict):
                text = str(item.get("text", ""))
                conf = item.get("confidence")
                try:
                    conf = float(conf) if conf is not None else None
                except Exception:
                    conf = None
                box = item.get("box", None)
            else:
                text = str(item)
            text_str = (text or "").strip()
            if not text_str:
                continue
            bbox = []
            if isinstance(box, (list, tuple)):
                for pt in box:
                    try:
                        x = int(round(float(pt[0])))
                        y = int(round(float(pt[1])))
                        bbox.append([x, y])
                    except Exception:
                        pass
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

# ----------------- Utility: extract cleaned matching lines -----------------
def extract_cleaned_text(parsed_lines, matched_keywords, full_text):
    """
    Return a list of cleaned lines that match the detected patterns:
    - matched_keywords: list from classify_document (patterns and fuzzy entries)
    - parsed_lines: list of dicts {'text':..., 'confidence':...}
    """
    plains = []
    for mk in matched_keywords or []:
        if isinstance(mk, str):
            if mk.startswith("FUZZY:"):
                # format: FUZZY:plain(92)
                try:
                    inner = mk.split("FUZZY:", 1)[1]
                    plain = inner.rsplit("(", 1)[0]
                    plain = plain.strip()
                except Exception:
                    plain = mk
            else:
                plain = _clean_pattern_to_plaintext(mk)
            if plain:
                plains.append(plain)
    # finalize plains unique & lowercase
    plains = list(dict.fromkeys([p.lower() for p in plains if p and p.strip()]))

    matched_lines = []
    for item in parsed_lines or []:
        ln = (item.get("text") or "").strip()
        if not ln:
            continue
        ln_low = ln.lower()
        matched_flag = False
        # substring quick check
        for p in plains:
            if not p:
                continue
            if p in ln_low:
                matched_flag = True
                break
            # fuzzy check
            try:
                sim = similarity(p, ln_low)
                if sim >= FUZZY_THRESHOLD:
                    matched_flag = True
                    break
            except Exception:
                pass
        if matched_flag:
            matched_lines.append(ln)
    # fallback: if nothing matched, pick first few non-empty lines (clean)
    if not matched_lines:
        for item in (parsed_lines or [])[:6]:
            t = (item.get("text") or "").strip()
            if t:
                matched_lines.append(t)
    # dedupe while preserving order
    seen = set()
    cleaned = []
    for l in matched_lines:
        if l not in seen:
            cleaned.append(l)
            seen.add(l)
    return cleaned

# ----------------- Flask /detect endpoint (no writes) -----------------
@app.route("/detect", methods=["POST"])
def detect():
    try:
        results = []
        files_list = []
        if 'file' in request.files:
            files_list = request.files.getlist('file')
        elif 'files' in request.files:
            files_list = request.files.getlist('files')

        multiple_mode = bool(files_list and len(files_list) > 1)

        def _process_single_image(image_bytes, fmt, original_name):
            tmp_file_path = None
            preprocessed_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{fmt}") as tmp:
                    tmp.write(image_bytes)
                    tmp_file_path = tmp.name

                preprocessed_path = preprocess_image(tmp_file_path)

                ocr_used = None
                final_parsed = []
                final_full_text = ""

                if paddle_ocr is not None:
                    try:
                        parsed, full_text = run_paddle_and_parse(preprocessed_path)
                        if full_text and len(full_text.strip()) >= 4:
                            final_parsed = parsed
                            final_full_text = full_text
                            ocr_used = "paddleocr"
                    except Exception as e:
                        print("PaddleOCR error (using fallback):", str(e))

                if not final_full_text and pytesseract is not None:
                    try:
                        parsed, full_text = run_tesseract_and_parse(preprocessed_path)
                        if full_text and len(full_text.strip()) >= 4:
                            final_parsed = parsed
                            final_full_text = full_text
                            ocr_used = "tesseract"
                    except Exception as e:
                        print("Tesseract error:", str(e))

                # classify document
                doc_match = classify_document(final_full_text or "")

                # extract cleaned lines that match the template/patterns
                cleaned_lines = extract_cleaned_text(final_parsed, doc_match.get("matched_keywords", []), final_full_text)

                input_b64 = bytes_to_base64(image_bytes, fmt=fmt)

                response_obj = {
                    "inputbase64": input_b64,
                    "detected_text": final_parsed,      # full parsed lines (list of dicts)
                    "cleaned_text": cleaned_lines,      # array of strings (cleaned / matched)
                    "document_type": doc_match,
                    "ocr_used": ocr_used
                }

                # We explicitly DO NOT save files any more.
                return response_obj
            finally:
                try:
                    if preprocessed_path and preprocessed_path != tmp_file_path and os.path.isfile(preprocessed_path):
                        os.remove(preprocessed_path)
                except Exception:
                    pass
                try:
                    if tmp_file_path and os.path.isfile(tmp_file_path):
                        os.remove(tmp_file_path)
                except Exception:
                    pass

        # multiple uploads
        if files_list and len(files_list) > 0:
            for f in files_list:
                try:
                    image_bytes = f.read()
                    original_name = f.filename or f"upload_{int(time.time())}.png"
                    fmt = original_name.rsplit('.', 1)[-1].lower() if '.' in original_name else "png"
                    response_obj = _process_single_image(image_bytes, fmt, original_name)
                    safe_response = json.loads(json.dumps(response_obj, default=str))
                    results.append({
                        "original_name": original_name,
                        "result": safe_response
                    })
                except Exception as e:
                    tb = traceback.format_exc()
                    results.append({
                        "original_name": getattr(f, "filename", None) or "unknown",
                        "error": str(e),
                        "trace": tb
                    })
            return jsonify({"results": results}), 200

        # single-file or JSON-style
        if 'file' in request.files and not multiple_mode:
            f = request.files['file']
            image_bytes = f.read()
            original_name = f.filename or f"upload_{int(time.time())}"
            fmt = original_name.rsplit('.', 1)[-1].lower() if '.' in original_name else "png"
            response_obj = _process_single_image(image_bytes, fmt, original_name)
            safe_response = json.loads(json.dumps(response_obj, default=str))
            return jsonify({"result": safe_response}), 200

        data = request.get_json(silent=True) or {}
        if data.get("image_name"):
            image_name = os.path.basename(data["image_name"])
            if not os.path.splitext(image_name)[1]:
                image_name += ".png"
            candidate = os.path.join(IMAGES_DIR, image_name)
            if not os.path.isfile(candidate):
                return jsonify({"error": f"image_name not found: {candidate}"}), 400
            with open(candidate, "rb") as fh:
                image_bytes = fh.read()
            original_name = image_name
            fmt = original_name.rsplit(".", 1)[-1].lower()
            response_obj = _process_single_image(image_bytes, fmt, original_name)
            safe_response = json.loads(json.dumps(response_obj, default=str))
            return jsonify({"result": safe_response}), 200
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
            with open(path, "rb") as fh:
                image_bytes = fh.read()
            original_name = os.path.basename(path)
            fmt = original_name.rsplit(".", 1)[-1].lower()
            response_obj = _process_single_image(image_bytes, fmt, original_name)
            safe_response = json.loads(json.dumps(response_obj, default=str))
            return jsonify({"result": safe_response}), 200
        elif data.get("image_base64"):
            b64 = data["image_base64"]
            if b64.startswith("data:") and ";base64," in b64:
                b64 = b64.split(";base64,", 1)[1]
            image_bytes = base64.b64decode(b64)
            original_name = f"image_base64_{int(time.time())}.png"
            response_obj = _process_single_image(image_bytes, "png", original_name)
            safe_response = json.loads(json.dumps(response_obj, default=str))
            return jsonify({"result": safe_response}), 200
        else:
            return jsonify({"error": "No image provided. Send 'file' or JSON with 'image_name'/'image_path'/'image_base64'."}), 400

    except Exception as e:
        tb = traceback.format_exc()
        msg = str(e)
        return jsonify({"error": msg, "trace": tb}), 500

# ----------------- Frontend route -----------------
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
