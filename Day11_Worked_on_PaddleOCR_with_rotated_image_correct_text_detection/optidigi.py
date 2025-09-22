import re, os, json, base64, math, statistics
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
import cv2
import requests

API_URL = "https://bankdevapi.digivision.ai/digivision/ai/rawtext-extraction"
_num_re = re.compile(r"-?\d+")

# ----------------- I/O helpers -----------------
def encode_image_file_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def prepare_payload(txn_id: str, document_name: str, doc_type: str, case_no: str, document_blob_b64: str) -> Dict[str, Any]:
    return {"txnId": txn_id, "docType": doc_type, "source": "OCR_RAW", "documentName": document_name, "caseNo": case_no, "documentBlob": document_blob_b64}

def post_to_api(payload: Dict[str, Any], url: str = API_URL, timeout: int = 60) -> Dict[str, Any]:
    resp = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

def save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return path

# ----------------- polygon parsing -----------------
def _num_list_from_str(s: str) -> List[int]:
    return [int(n) for n in _num_re.findall(s)]

def parse_poly_string(s: Any) -> List[List[int]]:
    if not isinstance(s, str):
        return []
    groups = re.findall(r"\[[^\]]+\]", s)
    pts = []
    for g in groups:
        nums = _num_re.findall(g)
        if len(nums) >= 2:
            try:
                pts.append([int(nums[0]), int(nums[1])])
            except Exception:
                pass
    if pts:
        return pts
    # last resort: look for any two numbers
    nums = _num_list_from_str(s)
    return [[nums[i], nums[i+1]] for i in range(0, len(nums)-1, 2)] if len(nums) >= 2 else []

def split_polys_from_big_string(s: Any) -> List[List[List[int]]]:
    if not isinstance(s, str):
        return []
    blocks = re.split(r"\]\s*\n\s*\n\s*\[", s)
    polygons = [parse_poly_string(b) for b in blocks]
    polygons = [p for p in polygons if p]
    if polygons:
        return polygons
    groups = re.findall(r"\[[^\]]+\]", s)
    if groups and len(groups) % 4 == 0:
        for i in range(0, len(groups), 4):
            pts = parse_poly_string(''.join(groups[i:i+4]))
            if pts:
                polygons.append(pts)
    return polygons

def normalize_box_element(el: Any) -> List[List[List[int]]]:
    if isinstance(el, str):
        many = split_polys_from_big_string(el)
        if many:
            return many
        pts = parse_poly_string(el)
        return [pts] if pts else []
    if isinstance(el, (list, tuple, np.ndarray)):
        # rectangle array [x1,y1,x2,y2]
        if len(el) == 4 and all(isinstance(x, (int, float, np.integer, np.floating)) for x in el):
            x1, y1, x2, y2 = map(int, map(round, el))
            return [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]
        # list of points
        if len(el) and isinstance(el[0], (list, tuple, np.ndarray)):
            pts = []
            for p in el:
                try:
                    pts.append([int(round(float(p[0]))), int(round(float(p[1])) )])
                except Exception:
                    pass
            return [pts] if pts else []
    return []

# ----------------- legacy paddle-like extractor -----------------
def extract_texts_and_boxes(ocr_result: Any) -> Tuple[List[str], List[List[List[int]]]]:
    texts, boxes = [], []
    if isinstance(ocr_result, list) and ocr_result and isinstance(ocr_result[0], dict):
        raw = ocr_result[0]
        texts = raw.get("rec_texts") or raw.get("texts") or raw.get("transcriptions") or []
        for key in ["rec_polys","dt_polys","rec_boxes","dt_boxes","boxes","polygons","points"]:
            val = raw.get(key)
            if val is None: continue
            if isinstance(val, str):
                parsed = split_polys_from_big_string(val)
                if parsed and len(parsed) >= len(texts):
                    boxes = parsed[:len(texts)]; break
            elif isinstance(val, (list, tuple)):
                normalized = [normalize_box_element(el)[0] if normalize_box_element(el) else [] for el in val]
                if texts and len(normalized) == len(texts):
                    boxes = normalized; break
                if not texts and normalized:
                    boxes = normalized; break
    # fallback older format
    if (not texts or not boxes) and isinstance(ocr_result, list) and ocr_result and isinstance(ocr_result[0], list):
        for line in ocr_result[0]:
            try:
                bbox_raw, text_part = line[0], line[1]
                text = text_part[0] if isinstance(text_part, (list, tuple)) else text_part
                texts.append(text)
                pts = [[int(round(float(p[0]))), int(round(float(p[1])))] for p in bbox_raw]
                boxes.append(pts)
            except Exception:
                pass
    return texts, boxes

# ----------------- fullTextAnnotation helpers -----------------

def find_full_text_annotation(resp_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(resp_json, dict):
        return None
    # common nested path
    r = resp_json.get("results")
    if isinstance(r, list) and r and isinstance(r[0], dict):
        data = r[0].get("Data") or r[0].get("data")
        if isinstance(data, dict):
            responses = data.get("responses")
            if isinstance(responses, list) and responses and isinstance(responses[0], dict):
                if responses[0].get("fullTextAnnotation"): return responses[0]["fullTextAnnotation"]
    if "fullTextAnnotation" in resp_json and isinstance(resp_json["fullTextAnnotation"], dict):
        return resp_json["fullTextAnnotation"]
    data = resp_json.get("data")
    if isinstance(data, dict) and data.get("fullTextAnnotation"): return data.get("fullTextAnnotation")
    # recursive search
    def _search(o: Any) -> Optional[Dict[str, Any]]:
        if isinstance(o, dict):
            if ("pages" in o and isinstance(o.get("pages"), list)) or "text" in o:
                return o
            for v in o.values():
                f = _search(v)
                if f: return f
        elif isinstance(o, list):
            for it in o:
                f = _search(it)
                if f: return f
        return None
    return _search(resp_json)


def reconstruct_text_from_paragraph(para: Dict[str, Any]) -> str:
    t = para.get("text") or para.get("description")
    if t: return t
    pieces = []
    for w in para.get("words", []):
        wt = w.get("text") or w.get("description")
        if wt: pieces.append(wt); continue
        syms = w.get("symbols", [])
        if syms: pieces.append("".join([s.get("text", "") for s in syms]))
    return " ".join(pieces).strip()


def get_vertices_from_bounding_box(bbox: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(bbox, dict): return []
    return bbox.get("vertices") or bbox.get("normalizedVertices") or []


def _normalize_vertices(vertices: Any) -> List[List[int]]:
    out = []
    if not vertices: return out
    if isinstance(vertices, list):
        for v in vertices:
            if isinstance(v, dict):
                x = v.get("x") or v.get("X") or v.get("cx")
                y = v.get("y") or v.get("Y") or v.get("cy")
                try: out.append([int(round(float(x))), int(round(float(y)))]);
                except Exception: pass
            elif isinstance(v, (list, tuple)) and len(v) >= 2:
                try: out.append([int(round(float(v[0]))), int(round(float(v[1])))])
                except Exception: pass
    return out


def text_and_boxes_from_fulltext(full_text_annotation: Dict[str, Any]) -> Tuple[List[str], List[List[List[int]]]]:
    texts, boxes = [], []
    pages = full_text_annotation.get("pages") or []
    if not pages:
        paras = full_text_annotation.get("paragraphs") or []
        for p in paras:
            texts.append(reconstruct_text_from_paragraph(p))
            boxes.append(_normalize_vertices(get_vertices_from_bounding_box(p.get("boundingBox") or p.get("bounding_box") or {})))
        return texts, boxes
    for page in pages:
        for block in page.get("blocks", []) or []:
            paras = block.get("paragraphs") or []
            if paras:
                for p in paras:
                    texts.append(reconstruct_text_from_paragraph(p))
                    boxes.append(_normalize_vertices(get_vertices_from_bounding_box(p.get("boundingBox") or p.get("bounding_box") or {})))
                continue
            lines = block.get("lines") or block.get("words") or []
            added = False
            for line in lines:
                if isinstance(line, dict):
                    texts.append(line.get("text") or line.get("description") or reconstruct_text_from_paragraph(line))
                    boxes.append(_normalize_vertices(get_vertices_from_bounding_box(line.get("boundingBox") or line.get("bounding_box") or {})))
                    added = True
            if added: continue
            bt = block.get("text") or block.get("description")
            if bt:
                texts.append(bt)
                boxes.append(_normalize_vertices(get_vertices_from_bounding_box(block.get("boundingBox") or block.get("bounding_box") or {})))
    return texts, boxes

# ----------------- paragraph & word extraction -----------------

def extract_and_save_paragraphs(full_text_annotation: Dict[str, Any], paragraph_dir: str, ocr_text_dir: str, base_filename: str):
    os.makedirs(paragraph_dir, exist_ok=True); os.makedirs(ocr_text_dir, exist_ok=True)
    all_paragraphs = []
    for p in (full_text_annotation.get("pages") or []):
        for block in p.get("blocks", []) or []:
            for para in block.get("paragraphs", []) or []:
                all_paragraphs.append(para)
    json_path = os.path.join(paragraph_dir, f"{base_filename}_paragraphs.json")
    save_json(all_paragraphs, json_path)
    simplified = []
    for para in all_paragraphs:
        t = reconstruct_text_from_paragraph(para)
        verts = _normalize_vertices(get_vertices_from_bounding_box(para.get("boundingBox") or para.get("bounding_box") or {}))
        simplified.append({"text": t, "bounding_box": verts})
    simplified_path = os.path.join(paragraph_dir, f"{base_filename}_simplified_paragraphs.json")
    save_json(simplified, simplified_path)
    txt_path = os.path.join(ocr_text_dir, f"{base_filename}_text.txt")
    full_text = full_text_annotation.get("text", "")
    with open(txt_path, "w", encoding="utf-8") as fh:
        if full_text: fh.write(full_text)
        else:
            for p in simplified: fh.write(p["text"] + "\n\n")
    return json_path, simplified_path, txt_path


def extract_words_from_fulltext(full_text_annotation: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    pages = full_text_annotation.get("pages") or []
    for page_idx, page in enumerate(pages):
        for block in page.get("blocks", []) or []:
            paras = block.get("paragraphs", []) or []
            if paras:
                for para in paras:
                    wlist = para.get("words", []) or []
                    if wlist:
                        for w in wlist:
                            text = w.get("text") or w.get("description") or "".join([s.get("text", "") for s in (w.get("symbols") or [])])
                            verts = _normalize_vertices((w.get("boundingBox") or w.get("bounding_box") or {}).get("vertices") or (w.get("boundingBox") or w.get("bounding_box") or {}).get("normalizedVertices") or [])
                            if text and verts: out.append({"text": str(text).strip(), "vertices": verts, "page": page_idx})
                        continue
                    ptext = para.get("text") or para.get("description")
                    verts = _normalize_vertices(get_vertices_from_bounding_box(para.get("boundingBox") or para.get("bounding_box") or {}))
                    if ptext and verts: out.append({"text": str(ptext).strip(), "vertices": verts, "page": page_idx})
                continue
            lines = block.get("lines", []) or block.get("words", []) or []
            for line in lines:
                if isinstance(line, dict):
                    wlist = line.get("words", []) or []
                    if wlist:
                        for w in wlist:
                            text = w.get("text") or w.get("description") or "".join([s.get("text", "") for s in (w.get("symbols") or [])])
                            verts = _normalize_vertices((w.get("boundingBox") or w.get("bounding_box") or {}).get("vertices") or (w.get("boundingBox") or w.get("bounding_box") or {}).get("normalizedVertices") or [])
                            if text and verts: out.append({"text": str(text).strip(), "vertices": verts, "page": page_idx})
                        continue
                    ltext = line.get("text") or line.get("description")
                    verts = _normalize_vertices(get_vertices_from_bounding_box(line.get("boundingBox") or line.get("bounding_box") or {}))
                    if ltext and verts: out.append({"text": str(ltext).strip(), "vertices": verts, "page": page_idx})
    # final fallback top-level paragraphs
    if not out:
        for para in full_text_annotation.get("paragraphs", []) or []:
            wlist = para.get("words", []) or []
            if wlist:
                for w in wlist:
                    text = w.get("text") or w.get("description")
                    verts = _normalize_vertices(get_vertices_from_bounding_box(w.get("boundingBox") or w.get("bounding_box") or {}))
                    if text and verts: out.append({"text": str(text).strip(), "vertices": verts, "page": 0})
            else:
                ptext = para.get("text") or para.get("description")
                verts = _normalize_vertices(get_vertices_from_bounding_box(para.get("boundingBox") or para.get("bounding_box") or {}))
                if ptext and verts: out.append({"text": str(ptext).strip(), "vertices": verts, "page": 0})
    return out

# ----------------- grouping / sorting -----------------

def _centroid(vertices: List[Tuple[float, float]]) -> Tuple[float, float]:
    xs = [v[0] for v in vertices]; ys = [v[1] for v in vertices]
    return float(sum(xs))/len(xs), float(sum(ys))/len(ys)

def _fit_baseline_and_sort(items: List[Dict[str, Any]]) -> List[Tuple[float, str]]:
    pts = [it.get("centroid", (it.get("x_left", 0.0), it.get("y_center", 0.0))) for it in items]
    xs = np.array([p[0] for p in pts], dtype=float); ys = np.array([p[1] for p in pts], dtype=float)
    if len(xs) < 2 or np.allclose(xs.std(), 0.0):
        return sorted([(it["x_left"], it["text"]) for it in items], key=lambda x: x[0])
    try: m, c = np.polyfit(xs, ys, 1)
    except Exception: m, c = 0.0, float(np.mean(ys))
    dir_unit = np.array([1.0, m], dtype=float); dir_unit /= np.linalg.norm(dir_unit) if np.linalg.norm(dir_unit) != 0 else 1.0
    projs = [(x*dir_unit[0] + y*dir_unit[1], it["text"]) for it, (x,y) in zip(items, pts)]
    projs.sort(key=lambda t: t[0]); return projs


def group_words_to_lines(words: List[Dict[str, Any]], tolerance_multiplier: float = 0.6, min_tolerance_px: int = 10, use_baseline: bool = True) -> Dict[Tuple[int,int], List[Tuple[float,str]]]:
    items = []
    for w in words:
        verts = w.get("vertices") or []
        if not verts: continue
        xs = [float(p[0]) for p in verts]; ys = [float(p[1]) for p in verts]
        x_left, y_top, y_bottom = min(xs), min(ys), max(ys)
        height = max(1.0, y_bottom - y_top)
        items.append({"text": w.get("text",""), "x_left": float(x_left), "y_center": (y_top+y_bottom)/2.0, "y_top": y_top, "y_bottom": y_bottom, "height": float(height), "centroid": _centroid(verts)})
    if not items: return {}
    heights = [it["height"] for it in items if it["height"]>0]
    median_h = statistics.median(heights) if heights else 12.0
    tol = max(min_tolerance_px, int(round(median_h * float(tolerance_multiplier))))
    items.sort(key=lambda e: e["y_center"])
    lines = []
    for it in items:
        placed = False
        for line in lines:
            if abs(it["y_center"] - line["center"]) <= tol:
                line["items"].append(it);
                line["center"] = sum(x["y_center"] for x in line["items"]) / len(line["items"])
                line["y_min"] = min(line["y_min"], it["y_top"]); line["y_max"] = max(line["y_max"], it["y_bottom"])
                placed = True; break
        if not placed:
            lines.append({"center": it["y_center"], "y_min": it["y_top"], "y_max": it["y_bottom"], "items": [it]})
    grouped = {}
    for line in lines:
        li = line["items"]
        if use_baseline and li:
            grouped[(int(round(line["y_min"])), int(round(line["y_max"])))] = _fit_baseline_and_sort(li)
        else:
            grouped[(int(round(line["y_min"])), int(round(line["y_max"])))] = sorted([(it["x_left"], it["text"]) for it in li], key=lambda t: t[0])
    return grouped


def create_sorted_text_file_from_words(words: List[Dict[str, Any]], output_path: str, tolerance_multiplier: float = 0.6, min_tolerance_px: int = 10, use_baseline: bool = True) -> str:
    grouped = group_words_to_lines(words, tolerance_multiplier=tolerance_multiplier, min_tolerance_px=min_tolerance_px, use_baseline=use_baseline)
    with open(output_path, 'w', encoding='utf-8') as fh:
        for y_range, elems in sorted(grouped.items(), key=lambda x: x[0][0]):
            fh.write(" ".join(t for _, t in elems).strip() + "\n")
    return output_path

# ----------------- drawing -----------------

def draw_annotations(img, texts, boxes, out_path):
    img_draw = img.copy(); h, w = img.shape[:2]; any_drawn = False
    for i, text in enumerate(texts):
        bbox = boxes[i] if i < len(boxes) else None
        if not bbox or len(bbox) < 2: continue
        pts = np.array(bbox, dtype=np.int32)
        x, y, bw, bh = cv2.boundingRect(pts)
        cv2.rectangle(img_draw, (x, y), (x+bw, y+bh), (0,200,0), 1, cv2.LINE_AA)
        font_scale = max(0.2, min(0.35, w / 900.0)); text_thickness = max(1, int(round(font_scale*2)))
        (w_text, h_text), _ = cv2.getTextSize(str(text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
        pad = 1
        rect_x1 = max(0, x); rect_y2 = max(0, y); rect_y1 = max(0, rect_y2 - (h_text + pad)); rect_x2 = min(w-1, x + w_text + pad)
        cv2.rectangle(img_draw, (rect_x1, rect_y1), (rect_x2, rect_y2), (0,255,0), -1)
        cv2.putText(img_draw, str(text), (rect_x1+3, rect_y2-3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), text_thickness, cv2.LINE_AA)
        any_drawn = True
    if not any_drawn:
        cv2.putText(img_draw, "NO DETECTIONS", (20, max(50, h//2)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3, cv2.LINE_AA)
    cv2.imwrite(out_path, img_draw)
    return out_path

# ----------------- misc -----------------

def save_raw_response(response: Dict[str, Any], output_dir: str, base_filename: str):
    os.makedirs(output_dir, exist_ok=True)
    return save_json(response, os.path.join(output_dir, f"{base_filename}_raw_response.json"))

def process_sorted_text_from_response(response_json, output_dir, base_filename, tolerance_multiplier: float = 0.6, min_tolerance_px: int = 10, use_baseline: bool = True):
    fta = find_full_text_annotation(response_json)
    if not fta:
        print("Could not find 'fullTextAnnotation' in response.")
        return None
    words = extract_words_from_fulltext(fta)
    sorted_text_path = os.path.join(output_dir, f"{base_filename}_sorted.txt")
    create_sorted_text_file_from_words(words, sorted_text_path, tolerance_multiplier=tolerance_multiplier, min_tolerance_px=min_tolerance_px, use_baseline=use_baseline)
    return sorted_text_path

# ----------------- main -----------------
if __name__ == "__main__":
    img_path = "images_multi/dhapubal.png"
    if not os.path.exists(img_path): raise SystemExit(f"Can't read image: {img_path}")
    img = cv2.imread(img_path)
    if img is None: raise SystemExit(f"Can't read image: {img_path}")

    paragraph_dir = "paragraph"; ocr_text_dir = "OCR_TEXT"; ocr_sorted_dir = "OCR_SORTED"; ocr_image_dir = "OCR_IMAGE"
    os.makedirs(paragraph_dir, exist_ok=True); os.makedirs(ocr_text_dir, exist_ok=True); os.makedirs(ocr_sorted_dir, exist_ok=True); os.makedirs(ocr_image_dir, exist_ok=True)

    b64 = encode_image_file_to_base64(img_path)
    payload = prepare_payload(txn_id="TXN0001", document_name=os.path.basename(img_path), doc_type=os.path.splitext(img_path)[1] or ".PNG", case_no="case001", document_blob_b64=b64)

    try:
        resp = post_to_api(payload)
    except Exception as e:
        raise SystemExit(f"Digivision API call failed: {e}")

    base_filename = os.path.splitext(os.path.basename(img_path))[0]
    raw_response_path = save_raw_response(resp, paragraph_dir, base_filename)
    print(f"Saved raw API response to {raw_response_path}")

    fta = find_full_text_annotation(resp)
    if not fta:
        print("Could not find 'fullTextAnnotation' in response. Response keys:", list(resp.keys()))
        raise SystemExit("No OCR data found in Digivision response.")

    json_path, simplified_path, txt_path = extract_and_save_paragraphs(fta, paragraph_dir, ocr_text_dir, base_filename)
    print(f"Saved complete paragraph data to {json_path}")
    print(f"Saved simplified paragraph data to {simplified_path}")
    print(f"Saved extracted text to {txt_path}")

    TOLERANCE_MULTIPLIER = 0.6; MIN_TOL_PX = 10; USE_BASELINE = True
    sorted_text_path = process_sorted_text_from_response(resp, ocr_sorted_dir, base_filename, tolerance_multiplier=TOLERANCE_MULTIPLIER, min_tolerance_px=MIN_TOL_PX, use_baseline=USE_BASELINE)
    if sorted_text_path: print(f"Saved sorted text to {sorted_text_path}")

    texts_par, boxes_par = text_and_boxes_from_fulltext(fta)
    words = extract_words_from_fulltext(fta)
    texts_for_draw = [w["text"] for w in words]
    boxes_for_draw = [w["vertices"] for w in words]
    while len(boxes_for_draw) < len(texts_for_draw): boxes_for_draw.append([])
    annotated_image_path = os.path.join(ocr_image_dir, f"{base_filename}_ocr.png")
    annotated = draw_annotations(img, texts_for_draw, boxes_for_draw, out_path=annotated_image_path)
    print(f"Saved annotated image to {annotated}")
    print("Done.")
