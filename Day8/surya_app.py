# app.py
import os
import io
import sys
import uuid
import json
import time
import base64
import traceback
from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)

# ---------------- config ----------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------- Shadow detection (unchanged) ---------
def detect_surya_shadow():
    if "surya" in sys.modules:
        mod = sys.modules["surya"]
        path = getattr(mod, "__file__", None)
        if path:
            site_pkg_markers = ("site-packages", "dist-packages", os.sep + "lib" + os.sep)
            if not any(marker in path for marker in site_pkg_markers):
                return f"Found already-imported module 'surya' at {path}. This probably shadows the installed package. Rename or remove that file."
    here = os.getcwd()
    p1 = os.path.join(here, "surya.py")
    p2 = os.path.join(here, "surya")
    msgs = []
    if os.path.isfile(p1):
        msgs.append(f"Local file '{p1}' exists and will shadow the installed 'surya' package. Rename it.")
    if os.path.isdir(p2):
        msgs.append(f"Local directory '{p2}' exists and may shadow the installed 'surya' package. Rename or remove it.")
    return "\n".join(msgs) if msgs else None

# --------- Lazy init for surya predictors (unchanged) ---------
_SURYA = {"foundation": None, "recognition": None, "detection": None}

def init_surya_once():
    if _SURYA["foundation"] is not None:
        return _SURYA

    shadow_msg = detect_surya_shadow()
    if shadow_msg:
        raise RuntimeError(
            "Local 'surya' module detected that will shadow the real package. "
            "Please rename your local file/folder named 'surya' and retry.\n\n" + shadow_msg
        )

    try:
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor
    except ModuleNotFoundError as e:
        hints = [
            "1) pip install surya-ocr",
            "2) Install PyTorch (surya depends on torch). Example (CPU): pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision",
            "3) Ensure there is no local file/folder named 'surya' in your project directory."
        ]
        raise RuntimeError(
            "Failed to import 'surya' package or its submodules.\n"
            f"Original error: {e}\n\nFollow these steps:\n" + "\n".join(hints)
        )

    try:
        foundation = FoundationPredictor()
        recognition = RecognitionPredictor(foundation)
        detection = DetectionPredictor()
    except Exception as e:
        raise RuntimeError(f"Surya was imported but failed to initialize predictors: {e}")

    _SURYA["foundation"] = foundation
    _SURYA["recognition"] = recognition
    _SURYA["detection"] = detection
    return _SURYA

# --------- Serialization helper (keeps legacy but used only in fallback) ---------
def serialize_for_json(obj):
    # keep simple serializer (used only if needed)
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [serialize_for_json(v) for v in obj]
    try:
        import numpy as _np
        if isinstance(obj, _np.generic):
            return obj.item()
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    try:
        import torch as _torch
        if isinstance(obj, _torch.Tensor):
            try:
                return obj.detach().cpu().tolist()
            except Exception:
                return str(obj)
    except Exception:
        pass
    try:
        if hasattr(obj, "to_dict") and callable(obj.to_dict):
            return serialize_for_json(obj.to_dict())
        if hasattr(obj, "dict") and callable(obj.dict):
            return serialize_for_json(obj.dict())
    except Exception:
        pass
    if hasattr(obj, "__dict__"):
        try:
            d = {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
            return serialize_for_json(d)
        except Exception:
            pass
    try:
        return str(obj)
    except Exception:
        return None

# --------- Helpers for images / IO (unchanged) ---------
def read_file_bytes(path: str):
    with open(path, "rb") as fh:
        return fh.read()

def image_to_data_uri(image_bytes: bytes, mime_type: str = "image/png") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"

def pil_from_bytes_or_path(path=None, bytes_data=None):
    if path:
        return Image.open(path).convert("RGB")
    if bytes_data:
        return Image.open(io.BytesIO(bytes_data)).convert("RGB")
    raise ValueError("Either path or bytes_data is required")

def sanitize_filename(name: str) -> str:
    import re
    base = os.path.basename(name or "image")
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", base)
    return cleaned

def create_output_json_file(response_obj: dict, input_filename: str) -> str:
    safe_base = sanitize_filename(input_filename or "image")
    ts = int(time.time())
    shortid = uuid.uuid4().hex[:8]
    out_name = f"{safe_base}__{ts}_{shortid}.json"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(response_obj, fh, indent=2, ensure_ascii=False)
    return out_path

# --------- polygon normalization utilities ---------
def _normalize_point(obj):
    # obj could be dict with x,y or left,top or [x,y]
    if isinstance(obj, dict):
        for kx, ky in (("x","y"), ("left","top"), ("X","Y")):
            if kx in obj and ky in obj:
                try:
                    return [float(obj[kx]), float(obj[ky])]
                except Exception:
                    pass
        # maybe keys '0','1' etc
        vals = []
        for k in sorted(obj.keys()):
            try:
                vals.append(float(obj[k]))
            except Exception:
                pass
        if len(vals) >= 2:
            return [vals[0], vals[1]]
        return None
    if isinstance(obj, (list, tuple)):
        if len(obj) >= 2:
            try:
                # nested point?
                if isinstance(obj[0], (list, tuple)):
                    return [float(obj[0][0]), float(obj[0][1])]
                return [float(obj[0]), float(obj[1])]
            except Exception:
                return None
    return None

def normalize_polygon(raw):
    """
    Convert variety of polygon/bbox representations into [[x,y],...]
    Returns None if cannot normalize.
    """
    if raw is None:
        return None
    # If it's already list of points
    if isinstance(raw, (list, tuple)):
        # list of [x,y] pairs
        if all(isinstance(pt, (list, tuple)) and len(pt) >= 2 for pt in raw):
            try:
                return [[float(pt[0]), float(pt[1])] for pt in raw]
            except Exception:
                pass
        # flat list of numbers -> group by 2
        if all(isinstance(v, (int, float, str)) for v in raw):
            try:
                nums = [float(v) for v in raw]
                if len(nums) % 2 == 0 and len(nums) >= 6:
                    pts = []
                    for i in range(0, len(nums), 2):
                        pts.append([nums[i], nums[i+1]])
                    return pts
            except Exception:
                pass
        # maybe list of dicts
        pts = []
        for item in raw:
            p = _normalize_point(item)
            if p:
                pts.append(p)
        if pts:
            return pts

    # if raw is dict that includes box/bbox/pts -> try to extract
    if isinstance(raw, dict):
        for key in ("points","poly","polygon","bbox","box","quad","coords"):
            if key in raw:
                return normalize_polygon(raw[key])

        # sometimes bbox provided as { "x":.., "y":.., "w":.., "h":.. }
        if all(k in raw for k in ("x","y","w","h")):
            try:
                x = float(raw["x"]); y = float(raw["y"]); w = float(raw["w"]); h = float(raw["h"])
                return [[x,y],[x+w,y],[x+w,y+h],[x,y+h]]
            except Exception:
                pass

    # fallback: try interpreting as string of numbers
    if isinstance(raw, str):
        nums = [s for s in (raw.replace(",", " ").split()) if _looks_like_number(s)]
        try:
            nums = [float(n) for n in nums]
            if len(nums) % 2 == 0 and len(nums) >= 6:
                pts = []
                for i in range(0, len(nums), 2):
                    pts.append([nums[i], nums[i+1]])
                return pts
        except Exception:
            pass

    return None

def _looks_like_number(s):
    try:
        float(s)
        return True
    except Exception:
        return False

# --------- Main endpoint (returns only base64 + text + polygon) ---------
@app.route("/ocr", methods=["POST"])
def ocr_endpoint():
    # initialize surya predictors
    try:
        surya = init_surya_once()
        recognition = surya["recognition"]
        detection = surya["detection"]
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"error": str(e), "trace": tb}), 500

    try:
        filename = None
        image_bytes = None

        if "file" in request.files:
            f = request.files["file"]
            filename = f.filename or "uploaded_image"
            image_bytes = f.read()
        elif request.is_json:
            payload = request.get_json()
            image_path = payload.get("image_path")
            if not image_path:
                return jsonify({"error":"JSON must include 'image_path' or send multipart 'file'."}), 400
            if not os.path.isabs(image_path):
                image_path = os.path.join(os.getcwd(), image_path)
            if not os.path.isfile(image_path):
                return jsonify({"error": f"image_path not found: {image_path}"}), 400
            filename = os.path.basename(image_path)
            image_bytes = read_file_bytes(image_path)
        else:
            return jsonify({"error":"Send multipart form with 'file' or JSON with 'image_path'."}), 400

        pil_img = pil_from_bytes_or_path(bytes_data=image_bytes)

        # Run recognition with detection predictor
        predictions = recognition([pil_img], det_predictor=detection)
        if not predictions:
            return jsonify({"error":"No predictions returned by Surya."}), 500

        pred = predictions[0]

        # Try to access text lines in several common shapes
        # We first try dict-like 'text_lines', else try 'lines', else try to coerce
        raw_serialized = None
        try:
            raw_serialized = serialize_for_json(pred)
        except Exception:
            raw_serialized = None

        candidates = []
        if isinstance(raw_serialized, dict):
            # priority keys where Surya commonly stores line info
            for key in ("text_lines", "lines", "predictions", "items"):
                if key in raw_serialized and isinstance(raw_serialized[key], list):
                    candidates = raw_serialized[key]
                    break
            # fallback: if dict contains a 'text' root
            if not candidates and "text" in raw_serialized and isinstance(raw_serialized["text"], str):
                candidates = [{"text": raw_serialized["text"]}]
        else:
            # if pred is simple list-like structure
            if isinstance(pred, (list, tuple)):
                candidates = list(pred)

        detections = []
        for item in candidates:
            if not isinstance(item, (dict, list, tuple, str)):
                continue
            text = None
            polygon = None
            if isinstance(item, str):
                text = item
            elif isinstance(item, (list, tuple)):
                # if it's [text, ...] try first element
                if item:
                    text = str(item[0])
            elif isinstance(item, dict):
                # common text keys
                for k in ("text","transcription","raw_text","line_text","content"):
                    if k in item and item[k]:
                        text = str(item[k])
                        break
                # polygon keys
                for polykey in ("polygon","poly","bbox","box","points","quad","coords"):
                    if polykey in item:
                        polygon = normalize_polygon(item[polykey])
                        break
                # maybe bounding box named differently
                if polygon is None:
                    # sometimes bbox provided as [x0,y0,x1,y1] under key 'bbox'
                    if "bbox" in item and isinstance(item["bbox"], (list, tuple)) and len(item["bbox"]) == 4:
                        x0,y0,x1,y1 = item["bbox"]
                        try:
                            polygon = [[float(x0),float(y0)],[float(x1),float(y0)],[float(x1),float(y1)],[float(x0),float(y1)]]
                        except Exception:
                            polygon = None
            # if no text but item contains nested 'text' field deeper
            if not text and isinstance(item, dict):
                # search nested
                for v in item.values():
                    if isinstance(v, str) and len(v.strip())>0:
                        text = v
                        break
            if text:
                detections.append({
                    "text": text.strip(),
                    "polygon": polygon  # may be None
                })

        # Build base64
        ext = os.path.splitext(filename)[1].lower()
        mime = "image/png"
        if ext in [".jpg", ".jpeg"]:
            mime = "image/jpeg"
        elif ext == ".webp":
            mime = "image/webp"
        elif ext in [".tif", ".tiff"]:
            mime = "image/tiff"
        input_b64 = image_to_data_uri(image_bytes, mime)

        # Build minimal response
        response_obj = {
            "filename": filename,
            "input_base64": input_b64,
            "detections": detections
        }

        # Save JSON to output and return relative path
        json_path = create_output_json_file(response_obj, filename)
        rel_path = os.path.relpath(json_path, PROJECT_ROOT)
        response_obj["json_filepath"] = rel_path

        return jsonify(response_obj), 200

    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"error": str(e), "trace": tb}), 500

if __name__ == "__main__":
    # IMPORTANT: Do NOT name this file 'surya.py'
    app.run(debug=True, host="0.0.0.0", port=5000)
