# test_detect.py
import os
import json
import time
import requests

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images3")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Put the PNG filename (existing in images folder) here:
IMAGE_NAME = "samp1.png"   # <- change to the PNG filename in images/

API_URL = "http://127.0.0.1:5000/detect"

def send_image_name(image_name):
    payload = {"image_name": image_name}
    r = requests.post(API_URL, json=payload)
    print("Status:", r.status_code)
    try:
        resp = r.json()
    except Exception:
        print("Server returned non-JSON:")
        print(r.text)
        return

    if r.status_code != 200:
        print("Error from server:", resp)
        return

    result = resp.get("result")
    saved_json_rel = resp.get("saved_json")
    print("Document type detected:", result.get("document_type"))
    texts = result.get("detected_text", [])
    print("First detected lines (up to 8):")
    for i, t in enumerate(texts[:8], 1):
        print(f" {i}. {t.get('text')} (conf={t.get('confidence')})")

    if saved_json_rel:
        saved_path = os.path.join(PROJECT_ROOT, saved_json_rel)
        if os.path.isfile(saved_path):
            print("Server saved JSON at:", saved_json_rel)
            with open(saved_path, "r", encoding="utf-8") as jf:
                data = json.load(jf)
            print("Saved JSON keys:", list(data.keys()))
        else:
            print("Server reported saved_json but file not found at:", saved_path)


if __name__ == "__main__":
    image_path = os.path.join(IMAGES_DIR, IMAGE_NAME)
    if not os.path.isfile(image_path):
        print("Image not found:", image_path)
        print("Place the PNG image inside the 'images' folder and update IMAGE_NAME.")
    else:
        send_image_name(IMAGE_NAME)
