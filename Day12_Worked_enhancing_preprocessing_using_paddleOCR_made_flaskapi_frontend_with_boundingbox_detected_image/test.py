# test.py
import sys
import requests
import os

DEFAULT = "images_multi/voter.png"
URL = "http://127.0.0.1:5000/upload"

def upload(path, url=URL):
    if not os.path.exists(path):
        print("Image not found:", path); return
    with open(path, "rb") as f:
        files = {"file": (os.path.basename(path), f, "application/octet-stream")}
        try:
            r = requests.post(url, files=files, timeout=120)
        except Exception as e:
            print("HTTP request failed:", e)
            return
    print("HTTP", r.status_code)
    try:
        j = r.json()
        print("JSON response:", j)
        out = j.get("output_txt")
        if out:
            if os.path.exists(out):
                print("\n--- OCR OUTPUT (file) ---\n")
                print(open(out, encoding="utf-8").read())
            else:
                # server may be running in different working dir — just show reported path
                print("Server reported output_txt path (may be on server machine):", out)
    except Exception as e:
        print("Non-JSON response or error:", e)
        print(r.text)

if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    upload(img)
