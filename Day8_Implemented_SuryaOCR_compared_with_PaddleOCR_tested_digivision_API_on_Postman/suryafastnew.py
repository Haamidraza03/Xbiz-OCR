# app.py — Surya-aware OCR server (performance-optimized: in-memory Surya & optional concurrency)
import os
import re
import time
import json
import base64
import tempfile
import traceback
import warnings
from typing import Any, List, Dict, Tuple, Optional
from flask import Flask, request, jsonify, render_template
from PIL import Image
from io import BytesIO
from html import unescape
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore", category=DeprecationWarning)

app = Flask(__name__)

# -------------- Performance / concurrency tunables --------------
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "0"))  # 0 means auto
if MAX_WORKERS <= 0:
    try:
        import multiprocessing
        MAX_WORKERS = min(4, multiprocessing.cpu_count() or 2)
    except Exception:
        MAX_WORKERS = 2

# -------------- Optional fuzzy backend (rapidfuzz preferred) --------------
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
    from difflib import SequenceMatcher as _SM

    def similarity(a: str, b: str) -> float:
        try:
            return float(_SM(None, a, b).ratio() * 100.0)
        except Exception:
            return 0.0

    print("rapidfuzz not available; using difflib fallback.")

# -------------- OCR backends (optional) --------------
paddle_ocr = None
try:
    from paddleocr import PaddleOCR
    _paddle_lang = os.environ.get("PADDLE_LANG", "en")
    _paddle_use_angle = os.environ.get("PADDLE_USE_ANGLE", "1") in ("1", "true", "True", "yes")
    try:
        paddle_ocr = PaddleOCR(use_angle_cls=_paddle_use_angle, lang=_paddle_lang)
    except TypeError:
        try:
            paddle_ocr = PaddleOCR(use_textline_orientation=_paddle_use_angle, lang=_paddle_lang)
        except TypeError:
            paddle_ocr = PaddleOCR(lang=_paddle_lang)
    print("PaddleOCR ready.")
except Exception:
    paddle_ocr = None
    print("PaddleOCR not available (ok — fallback possible).")

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
        _ = pytesseract.get_tesseract_version()
        print("Tesseract available.")
    except Exception:
        print("pytesseract imported but binary may be missing.")
except Exception:
    pytesseract = None
    Output = None
    print("pytesseract not installed; won't use Tesseract fallback.")

cv2 = None
np = None
try:
    import cv2
    import numpy as np
    print("OpenCV available for preprocessing (in-memory).")
except Exception:
    cv2 = None
    np = None
    print("OpenCV not available; preprocessing skipped (or done via PIL).")

# -------------- Project paths --------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images_multi")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------- Surya lazy init (if you have it) --------------
_surya_initialized = False
_surya_recognition = None
_surya_detection = None
_surya_foundation = None
_surya_available = False

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

SURYA_DEVICE = os.environ.get("SURYA_DEVICE", "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
SURYA_USE_HALF = os.environ.get("SURYA_HALF", "1") in ("1", "true", "True", "yes")
SURYA_MAX_DIM = int(os.environ.get("SURYA_MAX_DIM", "1400"))   # downscale target max dim before Surya
SURYA_WARMUP = os.environ.get("SURYA_WARMUP", "1") in ("1", "true", "True", "yes")
SURYA_WARMUP_SIZE = int(os.environ.get("SURYA_WARMUP_SIZE", "256"))
# Reduce OpenMP/MKL threads to avoid oversubscription (optional)
OMP_THREADS = os.environ.get("OMP_NUM_THREADS")
if OMP_THREADS:
    try:
        import os as _os
        _os.environ["OMP_NUM_THREADS"] = str(int(OMP_THREADS))
    except Exception:
        pass

def resize_pil_for_surya(pil_img: Image.Image, max_dim: int = SURYA_MAX_DIM) -> Image.Image:
    if pil_img is None:
        return pil_img
    w, h = pil_img.size
    maxc = max(w, h)
    if maxc <= max_dim:
        return pil_img
    scale = float(max_dim) / float(maxc)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return pil_img.resize((new_w, new_h), Image.LANCZOS)

def init_surya_once():
    """
    Initialize Surya predictors and (if possible) move models to target device and optionally half precision.
    Warm up model once to reduce first-request latency.
    """
    global _surya_initialized, _surya_recognition, _surya_detection, _surya_foundation, _surya_available
    if _surya_initialized:
        return
    _surya_initialized = True
    try:
        if os.path.exists(os.path.join(PROJECT_ROOT, "surya.py")):
            print("Warning: local surya.py exists — this may shadow the package.")
        # instantiate as before
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor

        # create predictors
        _surya_foundation = FoundationPredictor()
        _surya_recognition = RecognitionPredictor(_surya_foundation)
        _surya_detection = DetectionPredictor()

        # Try to move internal torch modules to device and set eval/half
        device = SURYA_DEVICE or ("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
        if TORCH_AVAILABLE:
            try:
                # Defensive: some predictors expose `.model` or `.net` etc.
                for pred in (_surya_foundation, _surya_recognition, _surya_detection):
                    if pred is None:
                        continue
                    for attr_name in ("model", "net", "backbone", "detector", "recognizer"):
                        try:
                            if hasattr(pred, attr_name):
                                module = getattr(pred, attr_name)
                                # if it's a torch.nn.Module, move to device
                                if hasattr(module, "to") and device:
                                    try:
                                        module.to(device)
                                    except Exception:
                                        pass
                                # try half-precision if requested and on CUDA
                                if SURYA_USE_HALF and device and "cuda" in device:
                                    try:
                                        if hasattr(module, "half"):
                                            module.half()
                                    except Exception:
                                        pass
                                # set eval mode if available
                                try:
                                    if hasattr(module, "eval"):
                                        module.eval()
                                except Exception:
                                    pass
                        except Exception:
                            continue
            except Exception as e:
                print("Warning: could not move Surya internals to device:", e)

        _surya_available = True
        print(f"Surya initialized — device={SURYA_DEVICE} half={SURYA_USE_HALF} max_dim={SURYA_MAX_DIM}")

        # Warmup: run one tiny dummy image through recognition to avoid first-request overhead
        if SURYA_WARMUP:
            try:
                small = Image.new("RGB", (SURYA_WARMUP_SIZE, SURYA_WARMUP_SIZE), color=(255,255,255))
                # resize/passed in list as Surya expects list of images
                if TORCH_AVAILABLE:
                    if SURYA_USE_HALF and SURYA_DEVICE and "cuda" in SURYA_DEVICE:
                        with torch.no_grad():
                            # autocast speeds FP16 inference on supported GPUs
                            try:
                                from torch.cuda.amp import autocast
                                with autocast():
                                    _ = _surya_recognition([small], det_predictor=_surya_detection)
                            except Exception:
                                _ = _surya_recognition([small], det_predictor=_surya_detection)
                    else:
                        with torch.no_grad():
                            _ = _surya_recognition([small], det_predictor=_surya_detection)
                else:
                    # no torch: still call predictor once
                    _ = _surya_recognition([small], det_predictor=_surya_detection)
                print("Surya warmup completed.")
            except Exception as e:
                print("Surya warmup failed (non-fatal):", e)

    except Exception as e:
        _surya_available = False
        print("Surya not available or failed to init:", str(e))
        traceback.print_exc()

# -------------- Utility helpers --------------
def bytes_to_base64(image_bytes: bytes, fmt: str = "png") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/{fmt};base64,{b64}"

def _looks_like_number(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False

def normalize_polygon(raw):
    try:
        if raw is None:
            return None
        if isinstance(raw, (list, tuple)):
            if all(isinstance(pt, (list, tuple)) and len(pt) >= 2 for pt in raw):
                return [[float(pt[0]), float(pt[1])] for pt in raw]
            if all(isinstance(v, (int, float, str)) for v in raw):
                nums = [float(v) for v in raw]
                if len(nums) % 2 == 0 and len(nums) >= 6:
                    return [[nums[i], nums[i+1]] for i in range(0, len(nums), 2)]
    except Exception:
        pass
    return None

# -------------- Generic extractor (keeps text/conf/bbox when available) --------------
def extract_text_items(obj: Any) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []
    if obj is None:
        return found
    if isinstance(obj, str):
        s = obj.strip()
        if s:
            found.append({"text": s})
        return found
    if isinstance(obj, dict):
        text_keys = ("text", "transcription", "raw_text", "content", "line_text", "caption", "line")
        for k in text_keys:
            if k in obj and isinstance(obj[k], str) and obj[k].strip():
                conf = None
                for ck in ("confidence", "conf", "score", "prob"):
                    if ck in obj:
                        try:
                            conf = float(obj[ck])
                        except Exception:
                            conf = None
                        break
                bbox = None
                for polykey in ("polygon", "poly", "bbox", "box", "points", "quad", "coords"):
                    if polykey in obj and obj[polykey] is not None:
                        bbox = normalize_polygon(obj[polykey])
                        break
                found.append({"text": obj[k].strip(), "confidence": conf, "bbox": bbox})
        for v in obj.values():
            found.extend(extract_text_items(v))
        return found
    if isinstance(obj, (list, tuple)):
        for it in obj:
            found.extend(extract_text_items(it))
        return found
    for attr in ("model_dump", "dict", "to_dict"):
        if hasattr(obj, attr) and callable(getattr(obj, attr)):
            try:
                dd = getattr(obj, attr)()
                found.extend(extract_text_items(dd))
                return found
            except Exception:
                pass
    try:
        s = str(obj)
        if s and not s.startswith("<"):
            found.append({"text": s})
    except Exception:
        pass
    return found

# -------------- Surya-specific extractor --------------
_HTML_BR_RE = re.compile(r'<br\s*/?>', flags=re.IGNORECASE)
_TAG_RE = re.compile(r'<[^>]+>')

def _split_and_clean_text_field(raw_text: str) -> List[str]:
    if raw_text is None:
        return []
    s = _HTML_BR_RE.sub("\n", raw_text)
    s = _TAG_RE.sub("", s)
    s = unescape(s)
    lines = [ln.strip() for ln in re.split(r'[\r\n]+', s) if ln.strip()]
    return lines

def _extract_surya_text_from_raw(raw: Any) -> List[str]:
    out: List[str] = []
    if raw is None:
        return out
    def walk(node):
        if isinstance(node, dict):
            for key in ("prediction_details", "detection_details", "detection_details_v2", "detection"):
                if key in node:
                    walk(node[key])
            if "text_lines" in node and isinstance(node["text_lines"], (list, tuple)):
                for tl in node["text_lines"]:
                    if isinstance(tl, dict):
                        t = tl.get("text") or tl.get("line") or tl.get("content") or tl.get("raw_text")
                        if isinstance(t, str) and t.strip():
                            splits = _split_and_clean_text_field(t)
                            for s in splits:
                                out.append(s)
                    elif isinstance(tl, str):
                        splits = _split_and_clean_text_field(tl)
                        for s in splits:
                            out.append(s)
            if "lines" in node and isinstance(node["lines"], (list, tuple)):
                for li in node["lines"]:
                    if isinstance(li, dict):
                        t = li.get("text") or li.get("line")
                        if isinstance(t, str) and t.strip():
                            out.extend(_split_and_clean_text_field(t))
                    elif isinstance(li, str):
                        out.extend(_split_and_clean_text_field(li))
            for v in node.values():
                walk(v)
        elif isinstance(node, (list, tuple)):
            for it in node:
                walk(it)
    walk(raw)
    seen = set(); dedup=[]
    for s in out:
        if s not in seen:
            dedup.append(s); seen.add(s)
    return dedup

# -------------- In-memory preprocessing helper (faster, avoids disk) --------------
def pil_from_bytes_with_optional_cv2(image_bytes: bytes) -> Image.Image:
    """
    Return a PIL.Image constructed from bytes. If OpenCV is available, do the heavy preprocessing
    in-memory and return a preprocessed RGB PIL image. Otherwise simply load via PIL.
    This function is used to avoid unnecessary disk I/O when Surya is available.
    """
    if cv2 is None or np is None:
        try:
            return Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception:
            # fallback: try saving to temp file and open
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(image_bytes)
                tmp_path = tmp.name
            try:
                pil = Image.open(tmp_path).convert("RGB")
                return pil
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return Image.open(BytesIO(image_bytes)).convert("RGB")
        max_dim = max(img.shape[:2])
        if max_dim > 3000:
            scale = 3000.0 / max_dim
            img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(gray)
        denoised = cv2.bilateralFilter(cl, d=9, sigmaColor=75, sigmaSpace=75)
        th = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=15, C=9)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
        # convert back to BGR 3-channel so PIL displays correctly
        opened_color = cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)
        pil_img = Image.fromarray(opened_color[..., ::-1])  # BGR->RGB
        return pil_img.convert("RGB")
    except Exception:
        try:
            return Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception:
            # last resort use temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(image_bytes)
                tmp_path = tmp.name
            try:
                pil = Image.open(tmp_path).convert("RGB")
                return pil
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

# -------------- OCR runner wrappers (Surya now accepts PIL.Image to avoid disk) --------------
def run_surya_and_parse(image_path: Optional[str] = None, pil_img: Optional[Image.Image] = None) -> Tuple[List[Dict[str, Any]], str]:
    """
    Run Surya recognition faster:
     - accept either image_path or pil_img
     - downscale image to SURYA_MAX_DIM before sending
     - use torch.no_grad() and autocast (fp16) when available
    Returns (parsed_lines, full_text)
    """
    init_surya_once()
    if not _surya_available:
        raise RuntimeError("Surya not available.")

    # prepare PIL image (priority: pil_img, then image_path)
    pil = None
    if pil_img is not None:
        pil = pil_img.convert("RGB")
    else:
        pil = Image.open(image_path).convert("RGB")

    # Resize (fast) — large images are the main cause of slowness
    pil_for_surya = resize_pil_for_surya(pil, max_dim=SURYA_MAX_DIM)

    # Inference
    start_t = time.time()
    raw = None
    try:
        # use torch/no_grad + autocast on CUDA if available
        if TORCH_AVAILABLE:
            try:
                if SURYA_USE_HALF and SURYA_DEVICE and "cuda" in SURYA_DEVICE:
                    from torch.cuda.amp import autocast
                    with torch.no_grad():
                        with autocast():
                            preds = _surya_recognition([pil_for_surya], det_predictor=_surya_detection)
                else:
                    with torch.no_grad():
                        preds = _surya_recognition([pil_for_surya], det_predictor=_surya_detection)
            except Exception as e:
                # fallback: try without autocast/no_grad
                preds = _surya_recognition([pil_for_surya], det_predictor=_surya_detection)
        else:
            preds = _surya_recognition([pil_for_surya], det_predictor=_surya_detection)
    except Exception as e:
        # bubble up helpful message
        raise RuntimeError("Surya recognition call failed: " + str(e))

    if not preds:
        return [], ""

    pred = preds[0]
    # convert predictor output to plain python structures (defensive)
    try:
        if isinstance(pred, dict):
            raw = pred
        elif hasattr(pred, "model_dump") and callable(getattr(pred, "model_dump")):
            raw = pred.model_dump()
        elif hasattr(pred, "dict") and callable(getattr(pred, "dict")):
            raw = pred.dict()
        elif hasattr(pred, "to_dict") and callable(getattr(pred, "to_dict")):
            raw = pred.to_dict()
        elif hasattr(pred, "__dict__"):
            raw = vars(pred)
        else:
            raw = str(pred)
    except Exception:
        raw = str(pred)

    # Reuse existing extract_text_items and _extract_surya_text_from_raw functions from your file
    items = extract_text_items(raw)
    parsed_lines = []
    for it in items:
        t = (it.get("text") if isinstance(it, dict) else str(it)).strip() if it else ""
        if not t:
            continue
        parsed_lines.append({
            "text": t,
            "confidence": (float(it.get("confidence")) if (isinstance(it, dict) and it.get("confidence") is not None) else None),
            "bbox": it.get("bbox") or []
        })

    try:
        surya_texts = _extract_surya_text_from_raw(raw)
        for s in surya_texts:
            if not any((p.get("text") or "") == s for p in parsed_lines):
                parsed_lines.append({"text": s, "confidence": None, "bbox": []})
    except Exception:
        pass

    if not parsed_lines:
        if isinstance(raw, dict) and "text" in raw and isinstance(raw["text"], str) and raw["text"].strip():
            for ln in [l.strip() for l in raw["text"].splitlines() if l.strip()]:
                parsed_lines.append({"text": ln, "confidence": None, "bbox": []})
        elif isinstance(raw, str) and raw.strip():
            for ln in [l.strip() for l in raw.splitlines() if l.strip()]:
                parsed_lines.append({"text": ln, "confidence": None, "bbox": []})

    full_text = "\n".join(p["text"] for p in parsed_lines) if parsed_lines else ""
    if SURYA_DEVICE and TORCH_AVAILABLE:
        took = time.time() - start_t
        print(f"Surya inference done in {took:.3f}s (device={SURYA_DEVICE}, half={SURYA_USE_HALF})")
    return parsed_lines, full_text

def run_paddle_and_parse(image_path: str) -> Tuple[List[Dict[str, Any]], str]:
    if paddle_ocr is None:
        raise RuntimeError("PaddleOCR not available.")
    try:
        try:
            ocr_res = paddle_ocr.predict(image_path, cls=True)
        except TypeError:
            ocr_res = paddle_ocr.predict(image_path)
    except Exception as e:
        raise RuntimeError(f"PaddleOCR failed: {e}")

    parsed_lines: List[Dict[str, Any]] = []
    full_text_parts: List[str] = []
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
                        x = int(round(float(pt[0]))); y = int(round(float(pt[1])))
                        bbox.append([x, y])
                    except Exception:
                        pass
            parsed_lines.append({"text": text_str, "confidence": float(conf) if conf is not None else None, "bbox": bbox})
            full_text_parts.append(text_str)
        except Exception:
            continue
    full_text = "\n".join(full_text_parts)
    return parsed_lines, full_text

def run_tesseract_and_parse(image_path: str) -> Tuple[List[Dict[str, Any]], str]:
    if pytesseract is None:
        raise RuntimeError("pytesseract not available.")
    try:
        data = pytesseract.image_to_data(Image.open(image_path), output_type=Output.DICT, lang='eng')
    except Exception as e:
        raise RuntimeError(f"Tesseract OCR failed: {e}")
    n = len(data.get('text', []))
    lines: Dict[Tuple[int,int,int], List[Dict[str,Any]]] = {}
    for i in range(n):
        txt = (data['text'][i] or "").strip()
        if not txt:
            continue
        key = (data.get('block_num', [0])[i], data.get('par_num', [0])[i], data.get('line_num', [0])[i])
        left = int(data.get('left', [0])[i]); top = int(data.get('top', [0])[i])
        width = int(data.get('width', [0])[i]); height = int(data.get('height', [0])[i])
        conf_raw = data.get('conf', [None])[i]
        try:
            conf = float(conf_raw)
        except Exception:
            conf = None
        item = {"word": txt, "left": left, "top": top, "width": width, "height": height, "conf": conf}
        lines.setdefault(key, []).append(item)
    parsed_lines: List[Dict[str,Any]] = []
    full_text_parts: List[str] = []
    for key, words in lines.items():
        words_sorted = sorted(words, key=lambda w: w["left"])
        text_line = " ".join(w["word"] for w in words_sorted)
        full_text_parts.append(text_line)
        parsed_lines.append({"text": text_line, "confidence": None, "bbox": []})
    full_text = "\n".join(full_text_parts)
    return parsed_lines, full_text

# -------------- Document classification (kept as original) --------------
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

# detection helper functions (kept intact)
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
            "male_found": False, "male_value": None,
            "dob_found": False, "dob_text": None,
            "gov_found": False,
            "unique_auth_found": False,
            "address_found": False,
            "relation_found": False
        }
    txt = full_text
    aadhaar_found, aadhaar_value, aadhaar_text = detect_12_digit_number(txt)
    male_found = bool(re.search(r"\bMale\b", txt, flags=re.IGNORECASE) or re.search(r"\b(?:Gender|Sex)[:\s]*M\b", txt, flags=re.IGNORECASE))
    male_val = "Male" if male_found else None
    dob_found = False
    dob_text = None
    m_date = re.search(r"(\b\d{1,2}[\/\-\s]\d{1,2}[\/\-\s]\d{2,4}\b|\b\d{4}[\/\-\s]\d{1,2}[\/\-\s]\d{1,2}\b)", txt)
    if m_date:
        dob_found = True
        dob_text = m_date.group(1)
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
        digit12_bonus = int(spec.get("digit12_bonus", 0) or 0)
        pan_bonus = int(spec.get("pan_bonus", 0) or 0)
        voterid_bonus = int(spec.get("voterid_bonus", 0) or 0)
        account11_bonus = int(spec.get("account11_bonus", 0) or 0)
        score = 0
        matched = []
        patterns = spec.get("patterns", []) if isinstance(spec, dict) else []
        for pat in patterns:
            try:
                if re.search(pat, full_text, flags=re.IGNORECASE):
                    score += 1
                    matched.append(pat)
                    continue
            except re.error:
                pass
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
        if digit12_bonus and digit12_found:
            score += digit12_bonus
            matched.append("12_digit_number")
        if pan_bonus and pan_found:
            score += pan_bonus
            matched.append("pan_number")
        if voterid_bonus and voterid_found:
            score += voterid_bonus
            matched.append("voterid_number")
        if account11_bonus and account11_found:
            score += account11_bonus
            matched.append("11_digit_account_number")
        matches[doc_type] = {"score": score, "matched": matched}
    extras = {
        "digit12_found": digit12_found,
        "digit12_value": digit12_value,
        "pan_found": pan_found,
        "pan_value": pan_value,
        "voterid_found": voterid_found,
        "voterid_value": voterid_value,
        "account11_found": account11_found,
        "account11_value": account11_value,
        "aadhaar_components": aadhaar_components
    }
    return matches, extras

def classify_document(full_text: str):
    matches, extras = get_template_scores(full_text)
    if not matches:
        return {"name": "UNKNOWN", "score": 0, "matched_keywords": []}
    best_doc, best_info = max(matches.items(), key=lambda x: x[1]["score"])
    result = {"name": "UNKNOWN", "score": best_info["score"], "matched_keywords": best_info.get("matched", [])}
    doc_rule = DOCUMENT_RULES.get(best_doc, {})
    min_score = doc_rule.get("min_score", 1)
    if best_info["score"] < min_score:
        return result
    result["name"] = best_doc
    result["digit12_found"] = bool(extras.get("digit12_found"))
    result["pan_found"] = bool(extras.get("pan_found"))
    result["voterid_found"] = bool(extras.get("voterid_found"))
    aadhaar_components = extras.get("aadhaar_components", {}) or {}
    result.update({
        "aadhaar_found": aadhaar_components.get("aadhaar_found", False),
        "aadhaar_value": aadhaar_components.get("aadhaar_value")
    })
    return result

# -------------- Final cleaning: ensure we only keep textual strings --------------
_BRACKET_CHARS = re.compile(r'[\{\}\[\]\(\)]')
_COORD_ONLY_RE = re.compile(r'^[\[\]\(\)\d\.,\s\-\:;]+$')
_CONF_NUM_RE = re.compile(r'^(?:conf(?:idence)?[:\s]*)?\d{1,3}(?:\.\d+)?$')

def _alpha_ratio(s: str) -> float:
    if not s:
        return 0.0
    letters = len(re.findall(r'[A-Za-z\u0900-\u097F]', s))
    return letters / max(1, len(s))

def _looks_like_coord_or_conf(s: str) -> bool:
    ss = s.strip()
    if not ss:
        return True
    if _COORD_ONLY_RE.match(ss):
        return True
    if _CONF_NUM_RE.match(ss):
        return True
    return False

def clean_parsed_lines_to_strings(parsed_lines: List[Any]) -> List[str]:
    out: List[str] = []
    for it in parsed_lines or []:
        txt = ""
        if isinstance(it, dict):
            if isinstance(it.get("text"), str) and it.get("text").strip():
                txt = it.get("text").strip()
            else:
                for k in ("line", "content", "word", "transcription", "raw_text"):
                    v = it.get(k)
                    if isinstance(v, str) and v.strip():
                        txt = v.strip(); break
                if not txt:
                    for k, v in it.items():
                        kl = k.lower()
                        if kl in ("polygon","poly","bbox","box","points","coords","confidence","conf","score","chars","words"):
                            continue
                        if isinstance(v, str) and v.strip():
                            txt = v.strip(); break
        elif isinstance(it, str):
            txt = it.strip()
        else:
            try:
                txt = str(it).strip()
            except Exception:
                txt = ""

        if not txt:
            continue

        txts = _split_and_clean_text_field(txt) if ('<' in txt and '>' in txt) else [txt]

        for s in txts:
            s2 = s.strip()
            if _BRACKET_CHARS.search(s2) and _alpha_ratio(s2) < 0.07:
                continue
            if _BRACKET_CHARS.search(s2):
                s2 = _BRACKET_CHARS.sub(" ", s2)
                s2 = re.sub(r'\s+', ' ', s2).strip()
            if _looks_like_coord_or_conf(s2):
                continue
            s2 = re.sub(r'\s+', ' ', s2).strip()
            if not s2:
                continue
            out.append(s2)

    seen = set(); dedup=[]
    for s in out:
        if s not in seen:
            dedup.append(s); seen.add(s)
    return dedup

# -------------- Core pipeline: process image bytes, run OCR, return JSON with detected_text plain strings --------------
def process_image_bytes(image_bytes: bytes, fmt: str, original_name: str) -> Dict[str, Any]:
    tmp_file_path = None
    preprocessed_pil = None
    try:
        ocr_used = None
        final_parsed: List[Dict[str,Any]] = []
        final_full_text = ""

        # --- Prepare a PIL image in-memory (fast) to use with Surya if available ---
        try:
            preprocessed_pil = pil_from_bytes_with_optional_cv2(image_bytes)
        except Exception:
            preprocessed_pil = None

        # 1) Surya (preferred) — use in-memory PIL.Image when possible (avoids disk writes)
        try:
            init_surya_once()
            if _surya_available:
                try:
                    parsed, full_text = run_surya_and_parse(pil_img=preprocessed_pil)
                    if full_text and full_text.strip():
                        final_parsed = parsed
                        final_full_text = full_text
                        ocr_used = "surya"
                except Exception as e:
                    # if Surya fails for a particular image, continue to fallbacks
                    print("Surya failed:", e)
        except Exception:
            pass

        # If Surya did not produce anything, fall back to Paddle/Tesseract.
        # Those generally accept file paths, so only write a temp file if needed.
        if not final_full_text:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{fmt}") as tmp:
                tmp.write(image_bytes)
                tmp_file_path = tmp.name
            # 2) Paddle fallback
            if paddle_ocr is not None:
                try:
                    parsed, full_text = run_paddle_and_parse(tmp_file_path)
                    if full_text and full_text.strip():
                        final_parsed = parsed
                        final_full_text = full_text
                        ocr_used = "paddleocr"
                except Exception as e:
                    print("Paddle failed:", e)
            # 3) Tesseract fallback
            if not final_full_text and pytesseract is not None:
                try:
                    parsed, full_text = run_tesseract_and_parse(tmp_file_path)
                    if full_text and full_text.strip():
                        final_parsed = parsed
                        final_full_text = full_text
                        ocr_used = "tesseract"
                except Exception as e:
                    print("Tesseract failed:", e)

        # If parser returned only full_text, split into lines
        if not final_parsed and final_full_text:
            for ln in [l.strip() for l in final_full_text.splitlines() if l.strip()]:
                final_parsed.append({"text": ln, "confidence": None, "bbox": []})

        # Final cleaning to plain strings (this is the critical step)
        detected_strings = clean_parsed_lines_to_strings(final_parsed)

        # fallback: clean raw full_text if nothing extracted
        if not detected_strings and final_full_text:
            fallback_lines = [ln.strip() for ln in final_full_text.splitlines() if ln.strip()]
            detected_strings = clean_parsed_lines_to_strings(fallback_lines)

        input_b64 = bytes_to_base64(image_bytes, fmt=fmt)

        # Document classification uses full_text (unchanged)
        try:
            doc_match = classify_document(final_full_text or "")
            doc_name = doc_match.get("name", "UNKNOWN") if isinstance(doc_match, dict) else str(doc_match)
        except Exception:
            doc_name = "UNKNOWN"

        response_obj = {
            "inputbase64": input_b64,
            "detected_text": detected_strings,
            "cleaned_text": [],
            "document_type": doc_name,
            "ocr_used": ocr_used
        }
        return response_obj
    finally:
        try:
            if tmp_file_path and os.path.isfile(tmp_file_path):
                os.remove(tmp_file_path)
        except Exception:
            pass

# -------------- Flask endpoints (use concurrency for batch operations) --------------
def _process_single_file_tuple(args):
    image_bytes, fmt, original_name = args
    try:
        return {"original_name": original_name, "result": process_image_bytes(image_bytes, fmt, original_name)}
    except Exception as e:
        tb = traceback.format_exc()
        return {"original_name": original_name, "error": str(e), "trace": tb}

@app.route("/detect", methods=["POST"])
def detect():
    try:
        results = []
        files_list = []
        if 'file' in request.files:
            files_list = request.files.getlist('file')
        elif 'files' in request.files:
            files_list = request.files.getlist('files')

        # Multiple files path: handle with ThreadPoolExecutor to speed up batch uploads
        if files_list and len(files_list) > 1:
            # prepare args
            args = []
            for f in files_list:
                image_bytes = f.read()
                original_name = f.filename or f"upload_{int(time.time())}.png"
                fmt = original_name.rsplit('.', 1)[-1].lower() if '.' in original_name else "png"
                args.append((image_bytes, fmt, original_name))
            # run in parallel (bounded workers)
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = [ex.submit(_process_single_file_tuple, a) for a in args]
                for fut in as_completed(futures):
                    try:
                        results.append(fut.result())
                    except Exception as e:
                        results.append({"original_name": "unknown", "error": str(e), "trace": traceback.format_exc()})
            return jsonify({"results": results}), 200

        # Single-file (fast path)
        if files_list and len(files_list) == 1:
            f = files_list[0]
            image_bytes = f.read()
            original_name = f.filename or f"upload_{int(time.time())}"
            fmt = original_name.rsplit('.', 1)[-1].lower() if '.' in original_name else "png"
            response_obj = process_image_bytes(image_bytes, fmt, original_name)
            return jsonify({"result": response_obj}), 200

        # JSON-style inputs (image_name, image_path, image_base64)
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
            response_obj = process_image_bytes(image_bytes, image_name.rsplit(".",1)[-1].lower(), image_name)
            return jsonify({"result": response_obj}), 200
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
            response_obj = process_image_bytes(image_bytes, fmt, original_name)
            return jsonify({"result": response_obj}), 200
        elif data.get("image_base64"):
            b64 = data["image_base64"]
            if b64.startswith("data:") and ";base64," in b64:
                b64 = b64.split(";base64,", 1)[1]
            image_bytes = base64.b64decode(b64)
            original_name = f"image_base64_{int(time.time())}.png"
            response_obj = process_image_bytes(image_bytes, "png", original_name)
            return jsonify({"result": response_obj}), 200
        else:
            return jsonify({"error": "No image provided. Send 'file' or JSON with 'image_name'/'image_path'/'image_base64'."}), 400

    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"error": str(e), "trace": tb}), 500

@app.route("/detect_all", methods=["GET"])
def detect_all():
    try:
        allowed = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
        results = []
        files = sorted(os.listdir(IMAGES_DIR))
        tasks = []
        for fname in files:
            _, ext = os.path.splitext(fname)
            if not ext or ext.lower() not in allowed:
                continue
            candidate = os.path.join(IMAGES_DIR, fname)
            if not os.path.isfile(candidate):
                continue
            with open(candidate, "rb") as fh:
                image_bytes = fh.read()
            fmt = ext.lstrip(".").lower() or "png"
            tasks.append((image_bytes, fmt, fname))
        # process in parallel
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(_process_single_file_tuple, t) for t in tasks]
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:
                    results.append({"original_name": "unknown", "error": str(e), "trace": traceback.format_exc()})
        return jsonify({"results": results}), 200
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"error": str(e), "trace": tb}), 500

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
