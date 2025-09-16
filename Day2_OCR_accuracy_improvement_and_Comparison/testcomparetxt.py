# test.py
import requests
import base64
import json
from pathlib import Path

SERVER = "http://127.0.0.1:5000/upload"
# update this to your test image path
IMG_PATH = Path("images2/voter.png")
LOCAL_COMPARISON_OUT = Path("comparison.txt")

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

# Create a friendly text summary locally (comparison.txt)
lines = []
lines.append(f"Server response status: {r.status_code}")
lines.append("")
meta = data.get("metadata", {})
lines.append("METADATA:")
lines.append(json.dumps(meta, indent=2))
lines.append("")

# Tesseract
t = data.get("Tesseract_ocr_response", {})
lines.append("=== TESSERACT ===")
lines.append(f"duration_sec: {t.get('duration_sec')}")
lines.append("full_text:")
lines.append(t.get('full_text', '').strip() or "<no text>")
lines.append("")

# EasyOCR
e = data.get("EasyOCR_ocr_response", {})
lines.append("=== EASYOCR ===")
lines.append(f"duration_sec: {e.get('duration_sec')}")
lines.append("full_text:")
lines.append(e.get('full_text', '').strip() or "<no text>")
lines.append("")

# PaddleOCR
p = data.get("PaddleOCR_ocr_response", {})
lines.append("=== PADDLEOCR ===")
lines.append(f"duration_sec: {p.get('duration_sec')}")
lines.append("full_text:")
lines.append(p.get('full_text', '').strip() or "<no text>")
lines.append("")

LOCAL_COMPARISON_OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote local comparison summary to {LOCAL_COMPARISON_OUT}")

# Optionally print small preview
print("Preview (first 200 chars):")
print("\n".join(lines)[:200])
