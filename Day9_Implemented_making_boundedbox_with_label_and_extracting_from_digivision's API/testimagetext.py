import re
import cv2
import numpy as np
import inspect
from paddleocr import PaddleOCR

# ---------- init PaddleOCR safely (works across versions) ----------
def init_paddleocr_safe(lang="en"):
    sig = inspect.signature(PaddleOCR.__init__)
    params = sig.parameters
    kwargs = {}
    if "lang" in params:
        kwargs["lang"] = lang
    if "use_textline_orientation" in params:
        kwargs["use_textline_orientation"] = True
    elif "use_angle_cls" in params:
        kwargs["use_angle_cls"] = True
    return PaddleOCR(**kwargs)

ocr = init_paddleocr_safe(lang="en")

# ---------- polygon parsing utilities ----------
_num_re = re.compile(r'-?\d+')

def parse_poly_string(s):
    if not isinstance(s, str):
        return []
    groups = re.findall(r'\[[^\]]+\]', s)
    pts = []
    for g in groups:
        nums = _num_re.findall(g)
        if len(nums) >= 2:
            try:
                x = int(nums[0]); y = int(nums[1])
                pts.append([x, y])
            except Exception:
                continue
    return pts

def split_polys_from_big_string(s):
    if not isinstance(s, str):
        return []
    blocks = re.split(r'\]\s*\n\s*\n\s*\[', s)
    polygons = []
    for blk in blocks:
        pts = parse_poly_string(blk)
        if pts:
            polygons.append(pts)
    if polygons:
        return polygons
    groups = re.findall(r'\[[^\]]+\]', s)
    if groups and len(groups) % 4 == 0:
        for i in range(0, len(groups), 4):
            chunk = ''.join(groups[i:i+4])
            pts = parse_poly_string(chunk)
            if pts:
                polygons.append(pts)
    if polygons:
        return polygons
    pts = parse_poly_string(s)
    return [pts] if pts else []

def normalize_box_element(el):
    if isinstance(el, str):
        many = split_polys_from_big_string(el)
        if many:
            return many
        pts = parse_poly_string(el)
        return [pts] if pts else []
    if isinstance(el, (list, tuple, np.ndarray)):
        if len(el) == 4 and all(isinstance(x, (int, float, np.integer, np.floating)) for x in el):
            x1, y1, x2, y2 = map(int, map(round, el))
            rect = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            return [rect]
        if len(el) > 0 and isinstance(el[0], (list, tuple, np.ndarray)):
            pts = []
            for p in el:
                try:
                    x = int(round(float(p[0]))); y = int(round(float(p[1])))
                    pts.append([x, y])
                except Exception:
                    pass
            return [pts] if pts else []
    return []

# ---------- helper to extract texts+boxes from PaddleOCR result ----------
def extract_texts_and_boxes(ocr_result):
    texts = []
    boxes = []

    if isinstance(ocr_result, list) and len(ocr_result) > 0 and isinstance(ocr_result[0], dict):
        raw = ocr_result[0]
        texts = raw.get("rec_texts") or raw.get("texts") or raw.get("transcriptions") or []
        candidate_keys = ["rec_polys", "dt_polys", "rec_boxes", "dt_boxes", "boxes", "polygons", "points"]
        for key in candidate_keys:
            val = raw.get(key)
            if val is None:
                continue
            if isinstance(val, str):
                parsed = split_polys_from_big_string(val)
                if parsed and len(parsed) >= len(texts):
                    boxes = parsed[:len(texts)]
                    break
            if isinstance(val, (list, tuple)):
                normalized = []
                for el in val:
                    normalized_el = normalize_box_element(el)
                    if normalized_el:
                        normalized.append(normalized_el[0])
                    else:
                        normalized.append([])
                if texts and len(normalized) == len(texts):
                    boxes = normalized
                    break
                if not texts and len(normalized) > 0:
                    boxes = normalized
                    break
        if not boxes and "rec_boxes" in raw and isinstance(raw["rec_boxes"], str):
            parsed = split_polys_from_big_string(raw["rec_boxes"])
            if parsed and texts and len(parsed) >= len(texts):
                boxes = parsed[:len(texts)]

    # fallback to older format
    if (not texts or not boxes) and isinstance(ocr_result, list) and len(ocr_result) > 0:
        first = ocr_result[0]
        if isinstance(first, list) and len(first) > 0 and isinstance(first[0], (list, tuple)):
            lines = first
            for line in lines:
                try:
                    bbox_raw = line[0]
                    text_part = line[1]
                    if isinstance(text_part, (list, tuple)):
                        text = text_part[0]
                    else:
                        text = text_part
                    texts.append(text)
                    pts = []
                    for p in bbox_raw:
                        try:
                            x = int(round(float(p[0]))); y = int(round(float(p[1])))
                            pts.append([x, y])
                        except Exception:
                            pass
                    boxes.append(pts)
                except Exception:
                    continue

    return texts, boxes

# ---------- drawing ----------
def draw_annotations(img, texts, boxes, out_path="annotated.png"):
    img_draw = img.copy()
    h, w = img.shape[:2]
    any_drawn = False

    for i, text in enumerate(texts):
        bbox = boxes[i] if i < len(boxes) else None
        if not bbox or len(bbox) < 2:
            continue
        pts = np.array(bbox, dtype=np.int32)

        # draw polygon (use the standard form accepted by OpenCV)
        # try:
            # cv2.polylines(img_draw, [pts], isClosed=True, color=(0,200,0), thickness=1, lineType=cv2.LINE_AA)
        # except Exception:
            # fallback: draw bounding rect only
            # pass

        # bounding rect fallback
        x, y, bw, bh = cv2.boundingRect(pts)
        cv2.rectangle(img_draw, (x, y), (x + bw, y + bh), (0,200,0), thickness=1, lineType=cv2.LINE_AA)

        # label: filled green rectangle + black text
        font_scale = max(0.1, min(0.3, w / 900.0))
        # ensure integer thickness for text drawing functions
        text_thickness = max(1, int(round(font_scale * 2)))
        (w_text, h_text), baseline = cv2.getTextSize(str(text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
        pad = 1

        rect_x1 = max(0, x)
        rect_y2 = max(0, y)
        rect_y1 = max(0, rect_y2 - (h_text + pad))
        rect_x2 = min(w - 1, x + w_text + pad)

        # filled label background (green)
        cv2.rectangle(img_draw, (rect_x1, rect_y1), (rect_x2, rect_y2), (0,255,0), thickness=-1)
        # black text on top
        cv2.putText(img_draw, str(text), (rect_x1 + 3, rect_y2 - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), text_thickness, cv2.LINE_AA)

        any_drawn = True

    if not any_drawn:
        cv2.putText(img_draw, "NO DETECTIONS", (20, max(50, h//2)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3, cv2.LINE_AA)

    cv2.imwrite(out_path, img_draw)
    return out_path

# ---------- main usage ----------
if __name__ == "__main__":
    # change this path to your image
    img_path = "images_multi/adharsamp.png"
    img = cv2.imread(img_path)
    if img is None:
        raise SystemExit(f"Can't read image: {img_path}")

    # Run OCR: prefer `predict` if available (newer PaddleOCR versions) to avoid deprecation warning.
    if hasattr(ocr, "predict"):
        result = ocr.predict(img_path)
    else:
        result = ocr.ocr(img_path)

    texts, boxes = extract_texts_and_boxes(result)

    # ensure boxes list length matches texts by padding with empty lists
    while len(boxes) < len(texts):
        boxes.append([])

    annotated = draw_annotations(img, texts, boxes, out_path="annotated.png")
    print("Saved annotated image to", annotated)
