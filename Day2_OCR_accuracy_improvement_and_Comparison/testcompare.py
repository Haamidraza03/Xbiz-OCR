# test.py
import requests
import json
from pathlib import Path

SERVER = "http://127.0.0.1:5000/upload"
# update this to your test image path
IMG_PATH = Path("images2/voter.png")
LOCAL_COMPARISON_OUT = Path("comparison.json")

if not IMG_PATH.exists():
    raise SystemExit(f"Image not found: {IMG_PATH}")

with open(IMG_PATH, "rb") as f:
    files = {"file": (IMG_PATH.name, f, "application/octet-stream")}
    print(f"Uploading {IMG_PATH} to {SERVER} ...")
    r = requests.post(SERVER, files=files)

print("Status code:", r.status_code)
try:
    data = r.json()
except Exception as e:
    print("Failed to decode JSON response:", e)
    print("Response text:", r.text)
    raise SystemExit(1)

# Extract the exact JSON fields the user requested
comparison_json = {
    "Input_Image_Base64": data.get("Input_Image_Base64"),
    "Tesseract_ocr_response": data.get("Tesseract_ocr_response"),
    "EasyOCR_ocr_response": data.get("EasyOCR_ocr_response"),
    "PaddleOCR_ocr_response": data.get("PaddleOCR_ocr_response"),
}

# Write local comparison JSON
with open(LOCAL_COMPARISON_OUT, "w", encoding="utf-8") as fh:
    json.dump(comparison_json, fh, ensure_ascii=False, indent=2, default=str)

print(f"Wrote local comparison JSON to {LOCAL_COMPARISON_OUT}")

# Optionally show short preview
preview = json.dumps({k: (v if isinstance(v, str) else (v.get('duration_sec') if isinstance(v, dict) else '...')) for k,v in comparison_json.items()}, indent=2)
print("Preview (durations):")
print(preview)
