# app.py
import os
import re
import json
import time
import base64
import tempfile
from flask import Flask, request, jsonify
from PIL import Image
import pytesseract
from pytesseract import Output

app = Flask(__name__)

# ----------------- Tesseract config -----------------
# If TESSERACT_CMD env var set, use it; otherwise try default Windows install path
env_cmd = os.environ.get("TESSERACT_CMD")
if env_cmd:
    pytesseract.pytesseract.tesseract_cmd = env_cmd
else:
    # default Windows path (common). Change if your tesseract.exe is elsewhere.
    default_win = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.isfile(default_win):
        pytesseract.pytesseract.tesseract_cmd = default_win

# quick check: if tesseract not found, server will still run but return an error on request
try:
    _ver = pytesseract.get_tesseract_version()
    print("Tesseract found:", _ver)
except Exception as e:
    print("Warning: pytesseract cannot find tesseract binary. "
          "Set TESSERACT_CMD env var or install Tesseract. Error:", e)


# ----------------- Document templates -----------------
DOCUMENT_TEMPLATES = {
    "PAN_CARD": [
        r"Permanent Account Number Card",
        r"permanent account",
        r"\bpan\b",
        r"INCOME TAX DEPARTMENT"
    ],
    "AADHAAR_CARD": [
        r"Unique Identification Authority",
        r"Government of India",
        r"aadhaar",
        r"Female",
        r"DOB",
        r"Male"
    ],
    "DRIVING_LICENSE": [
        r"DRIVING LICENCE",
        r"driving license",
        r"DL no",
        r"DOI",
        r"driver"
    ],
    "VOTER_ID": [
        r"ELECTION COMMISSION OF INDIA",
        r"IDENTITY CARD",
        r"Elector's Name"
    ],
    "BANK_STATEMENT":[
        r"Account Statement",
        r"Statement of Account",
        r"IFSC",
        r"Account Number",
        r"Branch",
    ]
}


# ----------------- Paths -----------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images3")  # your images folder
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)


# ----------------- Helpers -----------------
def sanitize_filename(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]', '_', name)


def bytes_to_base64(image_bytes: bytes, fmt: str = "png") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/{fmt};base64,{b64}"


def score_templates(full_text: str):
    matches = {}
    for doc_type, patterns in DOCUMENT_TEMPLATES.items():
        score = 0
        matched = []
        for pat in patterns:
            if re.search(pat, full_text, flags=re.IGNORECASE):
                score += 1
                matched.append(pat)
        matches[doc_type] = {"score": score, "matched": matched}
    best = max(matches.items(), key=lambda x: x[1]["score"])
    doc_name, info = best
    if info["score"] == 0:
        return {"name": "UNKNOWN", "score": 0, "matched_keywords": []}
    return {"name": doc_name, "score": info["score"], "matched_keywords": info["matched"]}


# ----------------- Tesseract-based OCR parse -----------------
def run_tesseract_and_parse(image_path):
    """
    Runs pytesseract.image_to_data and groups words into lines.
    Returns:
      - parsed_lines: list of {text, confidence, bbox}
      - full_text: string of all text (for template matching)
    """
    try:
        img = Image.open(image_path)
    except Exception as e:
        raise RuntimeError(f"Could not open image: {e}")

    try:
        data = pytesseract.image_to_data(img, output_type=Output.DICT, lang='eng')
    except Exception as e:
        raise RuntimeError(f"Tesseract error: {e}")

    n = len(data['text'])
    # group by (block_num, par_num, line_num)
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
        # sort words by left coordinate
        words_sorted = sorted(words, key=lambda w: w["left"])
        text_line = " ".join(w["word"] for w in words_sorted)
        full_text_parts.append(text_line)
        # compute bbox encompassing the line
        lefts = [w["left"] for w in words_sorted]
        tops = [w["top"] for w in words_sorted]
        rights = [w["left"] + w["width"] for w in words_sorted]
        bottoms = [w["top"] + w["height"] for w in words_sorted]
        bbox = [[min(lefts), min(tops)],
                [max(rights), min(tops)],
                [max(rights), max(bottoms)],
                [min(lefts), max(bottoms)]]
        # average confidence among words with numeric conf (-1 sometimes)
        confs = [w["conf"] for w in words_sorted if w["conf"] is not None and w["conf"] >= 0]
        avg_conf = (sum(confs) / len(confs)) if confs else None
        parsed_lines.append({
            "text": text_line,
            "confidence": avg_conf,
            "bbox": bbox
        })

    full_text = "\n".join(full_text_parts)
    return parsed_lines, full_text


# ----------------- Flask endpoint -----------------
@app.route("/detect", methods=["POST"])
def detect():
    """
    Accepts:
      - multipart/form-data with file field named 'file'
      - OR JSON with 'image_name' (filename inside images folder)  <-- preferred
      - OR JSON with 'image_path' (absolute or relative)
      - OR JSON with 'image_base64'
    Returns:
      JSON as {"inputbase64":..., "detected_text":[...], "document_type": {...} }
    Also saves outputs/<image_basename>.json
    """
    try:
        image_bytes = None
        fmt = "png"
        original_name = None

        # 1) file upload
        if 'file' in request.files:
            f = request.files['file']
            image_bytes = f.read()
            original_name = f.filename or f"upload_{int(time.time())}"
            if '.' in original_name:
                fmt = original_name.rsplit('.', 1)[1].lower()

        else:
            data = request.get_json(silent=True) or {}

            # preferred: image_name -> file inside images folder
            if data.get("image_name"):
                image_name = os.path.basename(data["image_name"])
                if not os.path.splitext(image_name)[1]:
                    image_name += ".png"  # try .png by default
                candidate = os.path.join(IMAGES_DIR, image_name)
                if not os.path.isfile(candidate):
                    return jsonify({"error": f"image_name not found in images: {candidate}"}), 400
                image_path = candidate
                original_name = image_name

            elif data.get("image_path"):
                path = os.path.normpath(data["image_path"])
                if not os.path.isabs(path):
                    # try relative to project root, then images folder
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
                # write to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    tmp.write(image_bytes)
                image_path = tmp.name

            else:
                return jsonify({"error": "No image found. Send 'file', or JSON with 'image_name' (preferred) or 'image_path' or 'image_base64'."}), 400

        # If image_bytes not set (we have image_path from images folder), set bytes for base64 later
        if image_bytes is None:
            with open(image_path, "rb") as fh:
                image_bytes = fh.read()
            fmt = original_name.rsplit(".", 1)[-1].lower()

        # Run Tesseract OCR and parse
        parsed, full_text = run_tesseract_and_parse(image_path)

        doc_match = score_templates(full_text)
        input_b64 = bytes_to_base64(image_bytes, fmt=fmt)

        response_obj = {
            "inputbase64": input_b64,
            "detected_text": parsed,
            "document_type": doc_match
        }

        # Save JSON only if a valid document type is detected
        if doc_match["score"] > 0:
            if not original_name:
                original_name = f"result_{int(time.time())}.json"
            base_name = sanitize_filename(os.path.splitext(original_name)[0])
            json_filename = f"{base_name}.json"
            json_path = os.path.join(OUTPUT_DIR, json_filename)
            if os.path.exists(json_path):
                json_filename = f"{base_name}_{int(time.time())}.json"
                json_path = os.path.join(OUTPUT_DIR, json_filename)
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(response_obj, jf, ensure_ascii=False, indent=2)

            rel_json = os.path.relpath(json_path, PROJECT_ROOT)
        else:
            rel_json = None

        # Clean temporary file if we created one for base64
        if data := request.get_json(silent=True) or {}:
            if data.get("image_base64"):
                try:
                    os.remove(image_path)
                except Exception:
                    pass

        return jsonify({"result": response_obj, "saved_json": rel_json}), 200

    except Exception as e:
        # helpful error message about tesseract
        msg = str(e)
        if "tesseract" in msg.lower():
            msg += (" -- make sure Tesseract is installed and the TESSERACT_CMD env var "
                    "points to the tesseract.exe on Windows, or edit the path in app.py")
        return jsonify({"error": msg}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)