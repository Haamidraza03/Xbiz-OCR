# test.py
import requests
import json
import sys
import os

API_URL = os.environ.get("SURYA_API_URL", "http://127.0.0.1:5000/ocr")

def test_with_path(image_path):
    payload = {"image_path": image_path}
    resp = requests.post(API_URL, json=payload)
    print("Status:", resp.status_code)
    try:
        print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
    except Exception:
        print(resp.text)

def test_with_upload(image_path):
    with open(image_path, "rb") as fh:
        files = {"file": (os.path.basename(image_path), fh, "application/octet-stream")}
        resp = requests.post(API_URL, files=files)
    print("Status:", resp.status_code)
    try:
        print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
    except Exception:
        print(resp.text)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test.py [path|upload] /path/to/image.jpg")
        sys.exit(1)

    mode = sys.argv[1].lower()
    image_path = sys.argv[2]
    if not os.path.isfile(image_path):
        print("Image not found:", image_path)
        sys.exit(2)

    if mode == "path":
        test_with_path(image_path)
    elif mode == "upload":
        test_with_upload(image_path)
    else:
        print("Unknown mode. Use 'path' or 'upload'.")
