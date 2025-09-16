import os
import json
import time
import mimetypes
import requests

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images_multi")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputweek2")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# --- Configuration ---
API_URL = "http://127.0.0.1:5000/detect"
# Mode: choose one of 'image_name', 'single_file', 'multi_files'
MODE = 'multi_files'  # change as needed

# For image_name mode: filename (must exist in IMAGES_DIR)
IMAGE_NAME = ""

# For single_file mode: filename to upload from IMAGES_DIR
SINGLE_UPLOAD_NAME = ""

# For multi_files mode: list filenames to upload from IMAGES_DIR.
# If empty, all PNG/JPG/JPEG files in IMAGES_DIR will be used.
MULTI_UPLOAD_NAMES = [
    #"aadhar6.png",
    #"aadhar7.png",
]

# ----------------------

def print_saved_json_info(rel_path):
    if not rel_path:
        print("No saved_json reported by server for this item.")
        return
    saved_path = os.path.join(PROJECT_ROOT, rel_path)
    print("Saved JSON (relative):", rel_path)
    if os.path.isfile(saved_path):
        try:
            with open(saved_path, 'r', encoding='utf-8') as jf:
                data = json.load(jf)
            print("  Keys in saved JSON:", list(data.keys()))
        except Exception as e:
            print("  Failed to read saved JSON:", e)
    else:
        print("  File not found on disk:", saved_path)


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

    print_saved_json_info(saved_json_rel)


def send_single_file(filename):
    path = os.path.join(IMAGES_DIR, filename)
    if not os.path.isfile(path):
        print("Image not found:", path)
        return
    mime, _ = mimetypes.guess_type(path)
    mime = mime or 'application/octet-stream'
    with open(path, 'rb') as fh:
        files = {'file': (filename, fh, mime)}
        r = requests.post(API_URL, files=files)
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

    print_saved_json_info(saved_json_rel)


def send_multi_files(filenames):
    # prepare the list of files; requests accepts a list of tuples for repeated fields
    file_tuples = []
    file_handles = []
    try:
        for fname in filenames:
            path = os.path.join(IMAGES_DIR, fname)
            if not os.path.isfile(path):
                print("Skipping missing file:", path)
                continue
            mime, _ = mimetypes.guess_type(path)
            mime = mime or 'application/octet-stream'
            fh = open(path, 'rb')
            file_handles.append(fh)
            # use the 'file' field multiple times (server accepts repeated 'file')
            file_tuples.append(('file', (fname, fh, mime)))

        if not file_tuples:
            print('No valid files to upload.')
            return

        r = requests.post(API_URL, files=file_tuples)
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

        # Server returns {'results': [...], 'saved_jsons': [...]}
        results = resp.get('results') or resp.get('result')
        saved = resp.get('saved_jsons') or resp.get('saved_json')

        if isinstance(results, list):
            print(f"Received {len(results)} item(s) from server.")
            for idx, item in enumerate(results, 1):
                name = item.get('original_name') or item.get('original') or f'item_{idx}'
                print('\n---')
                print(f"Item {idx}: {name}")
                if 'error' in item:
                    print('  Error:', item.get('error'))
                    continue
                res = item.get('result') or item.get('result', {})
                doc = res.get('document_type') if isinstance(res, dict) else None
                print('  Document type:', doc)
                texts = res.get('detected_text', []) if isinstance(res, dict) else []
                for i, t in enumerate(texts[:6], 1):
                    print(f"   {i}. {t.get('text')} (conf={t.get('confidence')})")
                saved_rel = item.get('saved_json')
                if saved_rel:
                    print_saved_json_info(saved_rel)
        else:
            # fallback single-style response
            print('Server returned single response:')
            print(json.dumps(resp, indent=2))

        # Also print global saved_jsons if provided
        if saved:
            print('\nSaved JSON paths reported:')
            for p in saved:
                print(' -', p)
    finally:
        for fh in file_handles:
            try:
                fh.close()
            except Exception:
                pass


if __name__ == '__main__':
    # Ensure images dir exists
    if not os.path.isdir(IMAGES_DIR):
        print('Images directory not found:', IMAGES_DIR)
        print('Create the directory and add test images before running.')
        exit(1)

    if MODE == 'image_name':
        print('Running in image_name mode...')
        send_image_name(IMAGE_NAME)
    elif MODE == 'single_file':
        print('Running in single_file mode...')
        send_single_file(SINGLE_UPLOAD_NAME)
    elif MODE == 'multi_files':
        print('Running in multi_files mode...')
        if not MULTI_UPLOAD_NAMES:
            # auto-discover common image types
            all_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print('Auto-discovered files:', all_files)
            MULTI_UPLOAD_NAMES = all_files
        send_multi_files(MULTI_UPLOAD_NAMES)
    else:
        print('Unknown MODE:', MODE)
        print("Set MODE to one of: 'image_name', 'single_file', 'multi_files'")
