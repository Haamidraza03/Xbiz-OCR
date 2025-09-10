"""
Document Segmentation for Fintech Documents
-------------------------------------------

This script implements document segmentation (text blocks, tables, photos) and OCR extraction
for fintech documents such as Aadhaar, PAN, bank statements, and driver's licenses.

Features
- Modular functions: preprocessing, text block detection, table detection, photo detection,
  OCR extraction (text and table data), and utilities for saving outputs.
- CLI: point the script at an input folder (images or PDFs) and an output folder.
- Produces: cropped images per segment, OCR text files, CSVs for detected tables, and a JSON
  manifest describing all segments.

Dependencies
- Python 3.8+
- OpenCV (opencv-python)
- numpy
- Pillow (PIL)
- pytesseract (Tesseract OCR must be installed separately and TESSERACT_CMD set if needed)
- pandas
- pdf2image (optional, to read PDFs) -> requires poppler on your system
- imutils (optional but convenient)

Install example:
    pip install opencv-python numpy pillow pytesseract pandas pdf2image imutils

If you get poor OCR results, make sure Tesseract is installed and available on PATH.
On Windows, set pytesseract.pytesseract.tesseract_cmd to the tesseract.exe path.

Usage:
    python document_segmentation_fintech.py --input examples/ --output results/ --min-table-area 5000

Notes and limitations
- This is a rule-based, traditional CV approach (morphology, contours, Hough-like line
  detection). It works well for structured scanned documents and photos but may need tuning
  for noisy camera-captured images or unusual layouts. For production, consider integrating
  deep-learning detectors (EAST, LayoutLM, Detectron2-based models) for more robust results.

"""

import os
import cv2
import sys
import json
import argparse
import numpy as np
from PIL import Image
import pytesseract
import pandas as pd
from pathlib import Path
import imutils
from pdf2image import convert_from_path

# Optional: If Tesseract not on PATH, set this to your tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ----------------------------- Utility functions -----------------------------

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_images_from_path(path):
    """Yield (filename, BGR image) pairs for images or PDFs.

    Accepts common image formats and PDF files (will convert each page to an image).
    """
    p = Path(path)
    if p.is_dir():
        for f in sorted(p.iterdir()):
            if f.suffix.lower() in ('.pdf',):
                pages = convert_from_path(str(f))
                for i, page in enumerate(pages):
                    img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
                    name = f.stem + f'_page{i+1}.png'
                    yield name, img
            elif f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'):
                img = cv2.imread(str(f))
                if img is not None:
                    yield f.name, img
    elif p.is_file():
        if p.suffix.lower() in ('.pdf',):
            pages = convert_from_path(str(p))
            for i, page in enumerate(pages):
                img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
                name = p.stem + f'_page{i+1}.png'
                yield name, img
        else:
            img = cv2.imread(str(p))
            yield p.name, img

# ----------------------------- Preprocessing ---------------------------------

def preprocess_for_layout(img):
    """Return a gray, denoised, and contrast-stretched image for layout analysis.

    Keep the original BGR image for final crops; this function returns a working copy.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Contrast stretch / histogram equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    # Mild blur to remove speckle noise
    gray = cv2.medianBlur(gray, 3)
    return gray

# ----------------------------- Text block detection --------------------------

def detect_text_blocks(img, debug=False, min_area=500, max_area=None):
    """Detect rectangular text regions using morphological operations.

    Returns list of bboxes (x, y, w, h) sorted top-to-bottom.
    """
    proc = preprocess_for_layout(img)
    # Binary inverse so text is white on black
    _, bw = cv2.threshold(proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = 255 - bw

    # Use dilation with a horizontal kernel to merge letters into lines/blocks
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 3))
    dilated = cv2.dilate(bw, horiz_kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    h_img, w_img = bw.shape
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w*h
        if area < min_area:
            continue
        if max_area and area > max_area:
            continue
        # Filter very tall narrow or tiny boxes
        if h < 10 or w < 10:
            continue
        # Normalize by image size to avoid weird outs
        if w > 0.98*w_img and h > 0.98*h_img:
            continue
        boxes.append((x, y, w, h))

    # Merge overlapping boxes (optional — helps group paragraphs)
    boxes = merge_overlapping_boxes(boxes)
    # Sort top-to-bottom
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    if debug:
        vis = img.copy()
        for (x,y,w,h) in boxes:
            cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.imshow('text_blocks', imutils.resize(vis, width=1000))
        cv2.waitKey(0)
    return boxes


def merge_overlapping_boxes(boxes, iou_thresh=0.1):
    """Simple non-max merge: if boxes overlap significantly, union them into one box.
    Works on lists of (x,y,w,h).
    """
    if not boxes:
        return []
    rects = [(*b[:2], b[0]+b[2], b[1]+b[3]) for b in boxes]  # x1,y1,x2,y2
    used = [False]*len(rects)
    merged = []
    for i, r in enumerate(rects):
        if used[i]:
            continue
        x1,y1,x2,y2 = r
        used[i] = True
        for j in range(i+1, len(rects)):
            if used[j]:
                continue
            xx1,yy1,xx2,yy2 = rects[j]
            # intersection
            ix1 = max(x1, xx1)
            iy1 = max(y1, yy1)
            ix2 = min(x2, xx2)
            iy2 = min(y2, yy2)
            iw = max(0, ix2-ix1)
            ih = max(0, iy2-iy1)
            inter = iw*ih
            area_i = (x2-x1)*(y2-y1)
            area_j = (xx2-xx1)*(yy2-yy1)
            union = area_i + area_j - inter
            iou = inter/union if union>0 else 0
            if iou > iou_thresh:
                # union
                x1 = min(x1, xx1)
                y1 = min(y1, yy1)
                x2 = max(x2, xx2)
                y2 = max(y2, yy2)
                used[j] = True
        merged.append((x1, y1, x2-x1, y2-y1))
    return merged

# ----------------------------- Table detection ------------------------------

def detect_tables(img, debug=False, min_table_area=2000):
    """Detect table-like regions by extracting horizontal & vertical lines and finding
    intersection patterns.

    Returns list of bboxes (x,y,w,h).
    """
    gray = preprocess_for_layout(img)
    # Binary
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw_inv = 255 - bw

    # Detect horizontal lines
    scale = max(20, img.shape[1]//50)  # kernel length relative to width
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (scale, 1))
    horizontal = cv2.erode(bw_inv, horiz_kernel, iterations=1)
    horizontal = cv2.dilate(horizontal, horiz_kernel, iterations=1)

    # Detect vertical lines
    scale_v = max(10, img.shape[0]//50)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, scale_v))
    vertical = cv2.erode(bw_inv, vert_kernel, iterations=1)
    vertical = cv2.dilate(vertical, vert_kernel, iterations=1)

    # Combine
    mask = cv2.bitwise_and(horizontal, vertical)

    # Find contours on mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        area = w*h
        if area < min_table_area:
            continue
        boxes.append((x,y,w,h))

    # Sort and return
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    if debug:
        vis = img.copy()
        for (x,y,w,h) in boxes:
            cv2.rectangle(vis, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.imshow('tables', imutils.resize(vis, width=1000))
        cv2.waitKey(0)
    return boxes

# ----------------------------- Photo detection -------------------------------

def detect_photos(img, candidate_boxes=None, ocr_word_thresh=10):
    """Given an image and some candidate boxes, classify which are photos (non-text)

    Heuristics used:
    - Low OCR word count relative to area
    - Higher local variance / colorfulness
    """
    # If candidate_boxes is None, fall back to detecting large regions not detected as text
    boxes = candidate_boxes or []
    photo_boxes = []
    h_img, w_img = img.shape[:2]
    for (x,y,w,h) in boxes:
        roi = img[y:y+h, x:x+w]
        # Quick OCR word count
        text = ocr_text_from_image(roi, digits_only=False)
        words = [t for t in text.split() if len(t.strip())>0]
        # Compute color variance / entropy
        variance = float(np.var(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)))
        # colorfulness metric (Hasler and Suesstrunk)
        (B,G,R) = cv2.split(roi.astype('float'))
        rg = np.absolute(R-G)
        yb = np.absolute(0.5*(R+G)-B)
        stdRoot = np.sqrt((np.var(rg)) + (np.var(yb)))
        meanRoot = np.sqrt((np.mean(rg)) + (np.mean(yb)))
        colorfulness = stdRoot + (0.3 * meanRoot)

        # Heuristic thresholds (tunable)
        if len(words) < ocr_word_thresh and (variance > 200 or colorfulness > 10):
            photo_boxes.append((x,y,w,h))
    return photo_boxes

# ----------------------------- OCR functions ---------------------------------

def ocr_text_from_image(roi, psm=3, oem=1, lang='eng', digits_only=False):
    """Return OCR'd text from a cropped ROI (BGR image).

    Use Tesseract through pytesseract.image_to_string. Preprocess ROI for better accuracy.
    """
    if roi is None or roi.size == 0:
        return ''
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Resize if small
    h, w = gray.shape
    if h < 50 or w < 50:
        gray = cv2.resize(gray, (max(100, w*2), max(100, h*2)), interpolation=cv2.INTER_CUBIC)

    # Threshold for clearer text
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 15, 9)
    config = f'--psm {psm} --oem {oem}'
    if digits_only:
        config += ' -c tessedit_char_whitelist=0123456789'
    try:
        text = pytesseract.image_to_string(gray, config=config, lang=lang)
    except Exception as e:
        print('pytesseract error:', e)
        text = ''
    return text.strip()


def ocr_table_to_dataframe(roi):
    """Try to extract table cells using Tesseract's TSV output and return a pandas DataFrame.

    This is a heuristic approach — real table recognition often needs specialized tools.
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    try:
        data = pytesseract.image_to_data(bw, output_type=pytesseract.Output.DATAFRAME)
    except Exception as e:
        print('pytesseract table extraction error:', e)
        return pd.DataFrame()

    # Keep only rows with some text
    data = data.dropna(subset=['text'])
    if data.empty:
        return pd.DataFrame()

    # Group by block_num/line_num then aggregate left-to-right by left coordinate
    rows = []
    for _, block in data.groupby(['block_num', 'par_num', 'line_num']):
        sorted_block = block.sort_values('left')
        row_text = ' | '.join(sorted_block['text'].astype(str).tolist())
        rows.append(row_text)
    df = pd.DataFrame({'row': rows})
    return df

# ----------------------------- High-level pipeline --------------------------

def segment_document(img, filename, output_dir, params):
    """Detect and segment text blocks, tables, and photos in the given image.

    Returns a manifest dict describing segments and saves crops + OCR outputs to output_dir.
    """
    manifest = {
        'filename': filename,
        'segments': []
    }

    # Detect tables first (they often contain text but are structured)
    table_boxes = detect_tables(img, min_table_area=params.get('min_table_area', 2000))

    # Detect text blocks
    # Use smaller min_area for mobile-captured documents
    text_boxes = detect_text_blocks(img, min_area=params.get('min_text_area', 300))

    # Remove text boxes that overlap heavily with detected tables (tables handled separately)
    text_boxes_filtered = []
    for tb in text_boxes:
        if not any(iou_box(tb, tb2) > 0.3 for tb2 in table_boxes):
            text_boxes_filtered.append(tb)

    # Detect photos among remaining candidates (or large boxes)
    # We'll combine big textless areas as photo candidates
    candidate_photo_boxes = []
    # Use contours on inverted threshold to find non-text big regions
    gray = preprocess_for_layout(img)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw_inv = 255 - bw
    contours, _ = cv2.findContours(bw_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w*h < params.get('min_photo_area', 1000):
            continue
        candidate_photo_boxes.append((x,y,w,h))

    photo_boxes = detect_photos(img, candidate_boxes=candidate_photo_boxes,
                                ocr_word_thresh=params.get('photo_ocr_word_thresh', 5))

    # Save and OCR segments
    seg_id = 0
    base_out = Path(output_dir)
    ensure_dir(base_out)
    txt_out = base_out / 'ocr_texts'
    ensure_dir(txt_out)
    crops_out = base_out / 'crops'
    ensure_dir(crops_out)
    tables_out = base_out / 'tables'
    ensure_dir(tables_out)

    # Save tables
    for (x,y,w,h) in table_boxes:
        seg_id += 1
        crop = img[y:y+h, x:x+w]
        fname = f'{Path(filename).stem}_segment{seg_id}_table.png'
        cv2.imwrite(str(crops_out / fname), crop)
        df = ocr_table_to_dataframe(crop)
        table_csv = tables_out / f'{Path(filename).stem}_segment{seg_id}_table.csv'
        try:
            if not df.empty:
                df.to_csv(table_csv, index=False)
        except Exception:
            pass
        ocr_text = '\n'.join(df['row'].tolist()) if not df.empty else ocr_text_from_image(crop)
        (txt_out / f'{Path(filename).stem}_segment{seg_id}_table.txt').write_text(ocr_text, encoding='utf-8')
        manifest['segments'].append({'id': seg_id, 'type': 'table', 'bbox': [x,y,w,h],
                                     'crop': str(crops_out / fname), 'ocr_text_file': str(txt_out / f'{Path(filename).stem}_segment{seg_id}_table.txt')})

    # Save text blocks
    for (x,y,w,h) in text_boxes_filtered:
        seg_id += 1
        crop = img[y:y+h, x:x+w]
        fname = f'{Path(filename).stem}_segment{seg_id}_text.png'
        cv2.imwrite(str(crops_out / fname), crop)
        ocr_text = ocr_text_from_image(crop)
        (txt_out / f'{Path(filename).stem}_segment{seg_id}_text.txt').write_text(ocr_text, encoding='utf-8')
        manifest['segments'].append({'id': seg_id, 'type': 'text_block', 'bbox': [x,y,w,h],
                                     'crop': str(crops_out / fname), 'ocr_text_file': str(txt_out / f'{Path(filename).stem}_segment{seg_id}_text.txt')})

    # Save photos
    for (x,y,w,h) in photo_boxes:
        seg_id += 1
        crop = img[y:y+h, x:x+w]
        fname = f'{Path(filename).stem}_segment{seg_id}_photo.png'
        cv2.imwrite(str(crops_out / fname), crop)
        # photos: no OCR, but we may still run a light OCR to capture text printed on photo
        ocr_text = ocr_text_from_image(crop)
        (txt_out / f'{Path(filename).stem}_segment{seg_id}_photo.txt').write_text(ocr_text, encoding='utf-8')
        manifest['segments'].append({'id': seg_id, 'type': 'photo', 'bbox': [x,y,w,h],
                                     'crop': str(crops_out / fname), 'ocr_text_file': str(txt_out / f'{Path(filename).stem}_segment{seg_id}_photo.txt')})

    # Write manifest
    (base_out / f'{Path(filename).stem}_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest

# ----------------------------- Helpers --------------------------------------

def iou_box(a, b):
    """Intersection-over-union for two boxes a and b in (x,y,w,h) format."""
    x1,y1,w1,h1 = a
    x2,y2,w2,h2 = b
    xa1, ya1, xa2, ya2 = x1, y1, x1+w1, y1+h1
    xb1, yb1, xb2, yb2 = x2, y2, x2+w2, y2+h2
    ix1 = max(xa1, xb1)
    iy1 = max(ya1, yb1)
    ix2 = min(xa2, xb2)
    iy2 = min(ya2, yb2)
    iw = max(0, ix2-ix1)
    ih = max(0, iy2-iy1)
    inter = iw*ih
    union = (w1*h1) + (w2*h2) - inter
    return inter/union if union>0 else 0

# ----------------------------- CLI / main -----------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Document segmentation for fintech docs')
    parser.add_argument('--input', '-i', required=True, help='Input image file or folder (images or PDFs)')
    parser.add_argument('--output', '-o', required=True, help='Output folder to save segments and OCR results')
    parser.add_argument('--min_table_area', type=int, default=2000, help='Minimum area to consider a table')
    parser.add_argument('--min_text_area', type=int, default=300, help='Minimum area to consider a text block')
    parser.add_argument('--min_photo_area', type=int, default=1000, help='Minimum area to consider a photo candidate')
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.input
    out_dir = args.output
    ensure_dir(out_dir)

    params = {
        'min_table_area': args.min_table_area,
        'min_text_area': args.min_text_area,
        'min_photo_area': args.min_photo_area,
    }

    manifests = []
    for fname, img in load_images_from_path(input_path):
        if img is None:
            print('Failed to read', fname)
            continue
        print('Processing', fname)
        manifest = segment_document(img, fname, out_dir, params)
        manifests.append(manifest)

    # Save combined manifest
    (Path(out_dir) / 'combined_manifest.json').write_text(json.dumps(manifests, indent=2), encoding='utf-8')
    print('Done. Results saved to', out_dir)


if __name__ == '__main__':
    main()
