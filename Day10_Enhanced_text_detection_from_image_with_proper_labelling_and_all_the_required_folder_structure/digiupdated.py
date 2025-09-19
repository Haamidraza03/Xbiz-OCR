import re
import cv2
import numpy as np
import requests
import json
import base64
import os
import sys
import statistics
import math
from typing import List, Tuple, Any, Dict, Optional
from collections import defaultdict

# ---------- Digivision endpoint ----------
API_URL = "https://bankdevapi.digivision.ai/digivision/ai/rawtext-extraction"


def encode_image_file_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def prepare_payload(txn_id: str, document_name: str, doc_type: str, case_no: str, document_blob_b64: str) -> Dict[str, Any]:
    return {
        "txnId": txn_id,
        "docType": doc_type,
        "source": "OCR_RAW",
        "documentName": document_name,
        "caseNo": case_no,
        "documentBlob": document_blob_b64,
    }


def post_to_api(payload: Dict[str, Any], url: str = API_URL, timeout: int = 60) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


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
                x = int(nums[0])
                y = int(nums[1])
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
                    x = int(round(float(p[0])))
                    y = int(round(float(p[1])))
                    pts.append([x, y])
                except Exception:
                    pass
            return [pts] if pts else []
    return []


# ---------- helper to extract texts+boxes from legacy paddle-like results ----------
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
                            x = int(round(float(p[0])))
                            y = int(round(float(p[1])))
                            pts.append([x, y])
                        except Exception:
                            pass
                    boxes.append(pts)
                except Exception:
                    continue

    return texts, boxes


# ---------- helper utilities to work with Digivision fullTextAnnotation ----------
def find_full_text_annotation(resp_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(resp_json, dict):
        return None

    # Check for the structure in your example response
    if "results" in resp_json and isinstance(resp_json["results"], list) and len(resp_json["results"]) > 0:
        first_result = resp_json["results"][0]
        if isinstance(first_result, dict) and "Data" in first_result:
            data = first_result["Data"]
            if isinstance(data, dict) and "responses" in data and isinstance(data["responses"], list) and len(data["responses"]) > 0:
                first_response = data["responses"][0]
                if isinstance(first_response, dict) and "fullTextAnnotation" in first_response:
                    return first_response["fullTextAnnotation"]

    # Fallback to original search method
    if "fullTextAnnotation" in resp_json and isinstance(resp_json["fullTextAnnotation"], dict):
        return resp_json["fullTextAnnotation"]

    data = resp_json.get("data")
    if isinstance(data, dict) and "fullTextAnnotation" in data:
        return data["fullTextAnnotation"]

    # search recursively
    def _search(obj: Any) -> Optional[Dict[str, Any]]:
        if isinstance(obj, dict):
            if ("pages" in obj and isinstance(obj.get("pages"), list)) or "text" in obj:
                return obj
            for v in obj.values():
                found = _search(v)
                if found:
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = _search(item)
                if found:
                    return found
        return None

    return _search(resp_json)


def reconstruct_text_from_paragraph(paragraph: Dict[str, Any]) -> str:
    text = paragraph.get("text") or paragraph.get("description")
    if text:
        return text
    pieces: List[str] = []
    for word in paragraph.get("words", []):
        wtext = word.get("text") or word.get("description")
        if wtext:
            pieces.append(wtext)
            continue
        symbols = word.get("symbols", [])
        sym_texts = [s.get("text", "") for s in symbols if s.get("text") is not None]
        if sym_texts:
            pieces.append("".join(sym_texts))
    return " ".join(pieces).strip()


def get_vertices_from_bounding_box(bbox: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(bbox, dict):
        return None
    return bbox.get("vertices") or bbox.get("normalizedVertices")


def _normalize_vertices(vertices: Any) -> List[List[int]]:
    out: List[List[int]] = []
    if not vertices:
        return out
    if isinstance(vertices, list):
        for v in vertices:
            if isinstance(v, dict):
                x = v.get("x") or v.get("X") or v.get("cx")
                y = v.get("y") or v.get("Y") or v.get("cy")
                try:
                    out.append([int(round(float(x))), int(round(float(y)))])
                except Exception:
                    continue
            elif isinstance(v, (list, tuple)) and len(v) >= 2:
                try:
                    out.append([int(round(float(v[0]))), int(round(float(v[1])) )])
                except Exception:
                    continue
    return out


def text_and_boxes_from_fulltext(full_text_annotation: Dict[str, Any]) -> Tuple[List[str], List[List[List[int]]]]:
    """
    Produce one entry per paragraph/line (not per block). This yields smaller rectangles
    for each line/paragraph instead of one large block rectangle.
    """
    texts: List[str] = []
    boxes: List[List[List[int]]] = []

    pages = full_text_annotation.get("pages") or []
    if not pages:
        paragraphs = full_text_annotation.get("paragraphs") or []
        if paragraphs:
            for para in paragraphs:
                text = reconstruct_text_from_paragraph(para)
                texts.append(text)
                bbox = para.get("boundingBox") or para.get("bounding_box") or {}
                vertices = get_vertices_from_bounding_box(bbox) or []
                boxes.append(_normalize_vertices(vertices))
        return texts, boxes

    for page in pages:
        blocks = page.get("blocks", []) or []
        for block in blocks:
            # Prefer paragraph-level granularity: one rectangle per paragraph
            block_paras = block.get("paragraphs", []) or []
            if block_paras:
                for para in block_paras:
                    p_text = reconstruct_text_from_paragraph(para)
                    texts.append(p_text)
                    bbox = para.get("boundingBox") or para.get("bounding_box") or {}
                    vertices = get_vertices_from_bounding_box(bbox) or []
                    boxes.append(_normalize_vertices(vertices))
                continue

            # Fallback: if no paragraphs, try lines/words inside block
            line_candidates = block.get("lines") or block.get("words") or []
            added_any = False
            for line in line_candidates:
                if isinstance(line, dict):
                    line_text = line.get("text") or line.get("description") or reconstruct_text_from_paragraph(line)
                    texts.append(line_text)
                    bbox = line.get("boundingBox") or line.get("bounding_box") or {}
                    vertices = get_vertices_from_bounding_box(bbox) or []
                    boxes.append(_normalize_vertices(vertices))
                    added_any = True
            if added_any:
                continue

            # Last fallback: treat whole block as a single line (if it has its own bounding box)
            block_text = block.get("text") or block.get("description") or ""
            if block_text:
                texts.append(block_text)
                bbox = block.get("boundingBox") or block.get("bounding_box") or {}
                vertices = get_vertices_from_bounding_box(bbox) or []
                boxes.append(_normalize_vertices(vertices))

    return texts, boxes


# ---------- Extract and save all paragraphs from fullTextAnnotation ----------
def extract_and_save_paragraphs(full_text_annotation: Dict[str, Any], paragraph_dir: str, ocr_text_dir: str, base_filename: str):
    """
    Extract all paragraphs from the fullTextAnnotation and save them in a structured format
    """
    os.makedirs(paragraph_dir, exist_ok=True)
    os.makedirs(ocr_text_dir, exist_ok=True)

    # Extract all paragraphs exactly as they appear in the API response
    all_paragraphs = []

    pages = full_text_annotation.get("pages", [])
    for page_idx, page in enumerate(pages):
        blocks = page.get("blocks", [])
        for block_idx, block in enumerate(blocks):
            block_paras = block.get("paragraphs", [])
            for para_idx, para in enumerate(block_paras):
                # Add the complete paragraph object as-is from the API response
                all_paragraphs.append(para)

    # Save as JSON with the exact structure from the API (in paragraph folder)
    json_path = os.path.join(paragraph_dir, f"{base_filename}_paragraphs.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_paragraphs, f, indent=2, ensure_ascii=False)

    # Also save a simplified version with just text and bounding boxes (in paragraph folder)
    simplified_paragraphs = []
    for para in all_paragraphs:
        para_text = reconstruct_text_from_paragraph(para)

        # Get bounding box
        bbox = para.get("boundingBox") or para.get("bounding_box") or {}
        vertices = get_vertices_from_bounding_box(bbox) or []
        normalized_vertices = _normalize_vertices(vertices)

        simplified_paragraphs.append({
            "text": para_text,
            "bounding_box": normalized_vertices
        })

    simplified_path = os.path.join(paragraph_dir, f"{base_filename}_simplified_paragraphs.json")
    with open(simplified_path, 'w', encoding='utf-8') as f:
        json.dump(simplified_paragraphs, f, indent=2, ensure_ascii=False)

    # Save as text file with original formatting (in OCR_TEXT folder)
    txt_path = os.path.join(ocr_text_dir, f"{base_filename}_text.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        # Use the full text from the annotation if available
        full_text = full_text_annotation.get("text", "")
        if full_text:
            f.write(full_text)
        else:
            # Fallback: reconstruct text from paragraphs
            for para in simplified_paragraphs:
                f.write(para["text"] + "\n\n")

    return json_path, simplified_path, txt_path


# ---------- WORD-LEVEL extraction (keeps rest of pipeline identical) ----------
def extract_words_from_fulltext(full_text_annotation: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract word-level tokens from fullTextAnnotation.
    Each returned item: { "text": str, "vertices": [[x,y],...], "page": int }
    This function does NOT save any extra JSON file.
    """
    words_out: List[Dict[str, Any]] = []
    pages = full_text_annotation.get("pages", []) or []
    for page_idx, page in enumerate(pages):
        blocks = page.get("blocks", []) or []
        for block in blocks:
            # prefer paragraphs -> words
            paras = block.get("paragraphs", []) or []
            if paras:
                for para in paras:
                    wlist = para.get("words", []) or []
                    if wlist:
                        for w in wlist:
                            # text might be in w['text'] or as symbols
                            text = w.get("text") or w.get("description")
                            if not text:
                                symbols = w.get("symbols", []) or []
                                text = "".join([s.get("text", "") for s in symbols])
                            bbox = w.get("boundingBox") or w.get("bounding_box") or {}
                            verts = bbox.get("vertices") or bbox.get("normalizedVertices") or []
                            verts_norm = _normalize_vertices(verts)
                            if text and verts_norm:
                                words_out.append({
                                    "text": str(text).strip(),
                                    "vertices": verts_norm,
                                    "page": page_idx
                                })
                        continue
                    # fallback: paragraph-level text (treat as one token)
                    ptext = para.get("text") or para.get("description")
                    bbox = para.get("boundingBox") or para.get("bounding_box") or {}
                    verts = bbox.get("vertices") or bbox.get("normalizedVertices") or []
                    verts_norm = _normalize_vertices(verts)
                    if ptext and verts_norm:
                        words_out.append({
                            "text": str(ptext).strip(),
                            "vertices": verts_norm,
                            "page": page_idx
                        })
                continue

            # fallback: block-level lines/words
            lines = block.get("lines", []) or block.get("words", []) or []
            if lines:
                for line in lines:
                    if isinstance(line, dict):
                        wlist = line.get("words", []) or []
                        if wlist:
                            for w in wlist:
                                text = w.get("text") or w.get("description")
                                if not text:
                                    symbols = w.get("symbols", []) or []
                                    text = "".join([s.get("text", "") for s in symbols])
                                bbox = w.get("boundingBox") or w.get("bounding_box") or {}
                                verts = bbox.get("vertices") or bbox.get("normalizedVertices") or []
                                verts_norm = _normalize_vertices(verts)
                                if text and verts_norm:
                                    words_out.append({
                                        "text": str(text).strip(),
                                        "vertices": verts_norm,
                                        "page": page_idx
                                    })
                            continue
                        # treat line node as single token
                        ltext = line.get("text") or line.get("description")
                        bbox = line.get("boundingBox") or line.get("bounding_box") or {}
                        verts = bbox.get("vertices") or bbox.get("normalizedVertices") or []
                        verts_norm = _normalize_vertices(verts)
                        if ltext and verts_norm:
                            words_out.append({
                                "text": str(ltext).strip(),
                                "vertices": verts_norm,
                                "page": page_idx
                            })
                    else:
                        continue

    # last fallback: top-level paragraphs
    if not words_out:
        paragraphs = full_text_annotation.get("paragraphs", []) or []
        for para in paragraphs:
            wlist = para.get("words", []) or []
            if wlist:
                for w in wlist:
                    text = w.get("text") or w.get("description")
                    bbox = w.get("boundingBox") or w.get("bounding_box") or {}
                    verts = bbox.get("vertices") or bbox.get("normalizedVertices") or []
                    verts_norm = _normalize_vertices(verts)
                    if text and verts_norm:
                        words_out.append({
                            "text": str(text).strip(),
                            "vertices": verts_norm,
                            "page": 0
                        })
            else:
                # fallback: paragraph as single token
                ptext = para.get("text") or para.get("description")
                bbox = para.get("boundingBox") or para.get("bounding_box") or {}
                verts = bbox.get("vertices") or bbox.get("normalizedVertices") or []
                verts_norm = _normalize_vertices(verts)
                if ptext and verts_norm:
                    words_out.append({
                        "text": str(ptext).strip(),
                        "vertices": verts_norm,
                        "page": 0
                    })

    return words_out


# ---------- Grouping words into lines (tolerance + baseline) ----------
def _centroid_of_vertices(vertices: List[Tuple[float, float]]) -> Tuple[float, float]:
    xs = [float(v[0]) for v in vertices]
    ys = [float(v[1]) for v in vertices]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _fit_baseline_and_sort(items: List[Dict[str, Any]]) -> List[Tuple[float, str]]:
    pts = [it.get("centroid", (it.get("x_left", 0.0), it.get("y_center", 0.0))) for it in items]
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    if len(xs) < 2 or np.allclose(xs.std(), 0.0):
        return sorted([(it["x_left"], it["text"]) for it in items], key=lambda x: x[0])
    try:
        m, c = np.polyfit(xs, ys, 1)
    except Exception:
        m, c = 0.0, float(np.mean(ys))
    dir_vec = np.array([1.0, m], dtype=float)
    norm = np.linalg.norm(dir_vec)
    dir_unit = dir_vec / norm if norm != 0.0 else np.array([1.0, 0.0])
    projections = []
    for it, (x, y) in zip(items, pts):
        proj = x * dir_unit[0] + y * dir_unit[1]
        projections.append((proj, it["text"]))
    projections.sort(key=lambda t: t[0])
    return projections


def group_words_to_lines(words: List[Dict[str, Any]],
                         tolerance_multiplier: float = 0.6,
                         min_tolerance_px: int = 10,
                         use_baseline: bool = True) -> Dict[Tuple[int,int], List[Tuple[float,str]]]:
    """
    Cluster words into lines by center-y using a dynamic tolerance (median height * multiplier)
    Returns dict keyed by (y_min,y_max) -> list of (sort_key,text) sorted left->right along baseline
    """
    items = []
    for w in words:
        verts = w.get("vertices", [])
        if not verts:
            continue
        xs = [float(p[0]) for p in verts]; ys = [float(p[1]) for p in verts]
        x_left = min(xs); y_top = min(ys); y_bottom = max(ys)
        y_center = (y_top + y_bottom) / 2.0
        height = max(1.0, y_bottom - y_top)
        centroid = _centroid_of_vertices(verts)
        items.append({
            "text": str(w.get("text", "")).strip(),
            "x_left": float(x_left),
            "y_top": float(y_top),
            "y_bottom": float(y_bottom),
            "y_center": float(y_center),
            "height": float(height),
            "centroid": centroid
        })

    if not items:
        return {}

    heights = [it["height"] for it in items if it["height"] > 0]
    if heights:
        try:
            median_h = statistics.median(heights)
        except Exception:
            median_h = sum(heights) / len(heights)
    else:
        median_h = 12.0

    tolerance = max(min_tolerance_px, int(round(median_h * float(tolerance_multiplier))))

    # cluster top->bottom by y_center
    items.sort(key=lambda e: e["y_center"])
    lines = []
    for it in items:
        placed = False
        for line in lines:
            if abs(it["y_center"] - line["center"]) <= tolerance:
                line["items"].append(it)
                line["center"] = sum(x["y_center"] for x in line["items"]) / len(line["items"])
                line["y_min"] = min(line["y_min"], it["y_top"])
                line["y_max"] = max(line["y_max"], it["y_bottom"])
                placed = True
                break
        if not placed:
            lines.append({
                "center": it["y_center"],
                "y_min": it["y_top"],
                "y_max": it["y_bottom"],
                "items": [it]
            })

    grouped: Dict[Tuple[int,int], List[Tuple[float,str]]] = {}
    for line in lines:
        li = line["items"]
        if use_baseline and len(li) >= 1:
            sorted_elems = _fit_baseline_and_sort(li)  # returns (proj, text)
            grouped[(int(round(line["y_min"])), int(round(line["y_max"])))] = sorted_elems
        else:
            simple_sorted = sorted([(it["x_left"], it["text"]) for it in li], key=lambda t: t[0])
            grouped[(int(round(line["y_min"])), int(round(line["y_max"])))] = simple_sorted

    return grouped


def create_sorted_text_file_from_words(words: List[Dict[str, Any]],
                                       output_path: str,
                                       tolerance_multiplier: float = 0.6,
                                       min_tolerance_px: int = 10,
                                       use_baseline: bool = True) -> str:
    grouped = group_words_to_lines(words, tolerance_multiplier=tolerance_multiplier,
                                   min_tolerance_px=min_tolerance_px, use_baseline=use_baseline)
    sorted_lines = sorted(grouped.items(), key=lambda x: x[0][0])
    with open(output_path, 'w', encoding='utf-8') as fh:
        for y_range, elems in sorted_lines:
            line_text = " ".join(t for _, t in elems).strip()
            fh.write(line_text + "\n")
    return output_path


# ---------- drawing (unchanged filename conventions) ----------
def draw_annotations(img, texts, boxes, out_path):
    img_draw = img.copy()
    h, w = img.shape[:2]
    any_drawn = False

    for i, text in enumerate(texts):
        bbox = boxes[i] if i < len(boxes) else None
        if not bbox or len(bbox) < 2:
            continue
        pts = np.array(bbox, dtype=np.int32)

        x, y, bw, bh = cv2.boundingRect(pts)
        cv2.rectangle(img_draw, (x, y), (x + bw, y + bh), (0, 200, 0), thickness=1, lineType=cv2.LINE_AA)

        font_scale = max(0.2, min(0.35, w / 900.0))
        text_thickness = max(1, int(round(font_scale * 2)))
        (w_text, h_text), baseline = cv2.getTextSize(str(text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
        pad = 1

        rect_x1 = max(0, x)
        rect_y2 = max(0, y)
        rect_y1 = max(0, rect_y2 - (h_text + pad))
        rect_x2 = min(w - 1, x + w_text + pad)

        cv2.rectangle(img_draw, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 255, 0), thickness=-1)
        cv2.putText(img_draw, str(text), (rect_x1 + 3, rect_y2 - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), text_thickness, cv2.LINE_AA)

        any_drawn = True

    if not any_drawn:
        cv2.putText(img_draw, "NO DETECTIONS", (20, max(50, h // 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imwrite(out_path, img_draw)
    return out_path


# ---------- Save raw response for debugging ----------
def save_raw_response(response: Dict[str, Any], output_dir: str, base_filename: str):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{base_filename}_raw_response.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(response, f, indent=2, ensure_ascii=False)
    return json_path


# ---------- main modification: process_sorted_text_from_response now uses WORD-LEVEL grouping ----------
def process_sorted_text_from_response(response_json, output_dir, base_filename, tolerance_multiplier: float = 0.6, min_tolerance_px: int = 10, use_baseline: bool = True):
    """
    Process the JSON response to create a sorted text file using word-level boxes.
    Saves output in same naming/location convention as before:
      OCR_SORTED/<base_filename>_sorted.txt
    """
    # Try to find fullTextAnnotation in response
    fta = find_full_text_annotation(response_json)
    if not fta:
        print("Could not find 'fullTextAnnotation' in response.")
        return None

    # Extract words (word-level bounding boxes)
    words = extract_words_from_fulltext(fta)

    # Create sorted text file (same output filename convention as your original pipeline)
    sorted_text_path = os.path.join(output_dir, f"{base_filename}_sorted.txt")
    create_sorted_text_file_from_words(words, sorted_text_path,
                                       tolerance_multiplier=tolerance_multiplier,
                                       min_tolerance_px=min_tolerance_px,
                                       use_baseline=use_baseline)

    return sorted_text_path


# ---------- main usage (keeps filenames same as your original) ----------
if __name__ == "__main__":
    # change this path to your image
    img_path = "images_multi/voter.png"
    if not os.path.exists(img_path):
        raise SystemExit(f"Can't read image: {img_path}")

    img = cv2.imread(img_path)
    if img is None:
        raise SystemExit(f"Can't read image: {img_path}")

    # Create output directories (same names you used originally)
    paragraph_dir = "paragraph"
    ocr_text_dir = "OCR_TEXT"
    ocr_sorted_dir = "OCR_SORTED"
    ocr_image_dir = "OCR_IMAGE"

    os.makedirs(paragraph_dir, exist_ok=True)
    os.makedirs(ocr_text_dir, exist_ok=True)
    os.makedirs(ocr_sorted_dir, exist_ok=True)
    os.makedirs(ocr_image_dir, exist_ok=True)

    # Prepare base64 and payload for Digivision
    b64 = encode_image_file_to_base64(img_path)
    payload = prepare_payload(
        txn_id="TXN0001",
        document_name=os.path.basename(img_path),
        doc_type=os.path.splitext(img_path)[1] or ".PNG",
        case_no="case001",
        document_blob_b64=b64
    )

    # Call Digivision OCR
    try:
        resp = post_to_api(payload)
    except Exception as e:
        raise SystemExit(f"Digivision API call failed: {e}")

    # Save the raw response for debugging (in paragraph folder)
    base_filename = os.path.splitext(os.path.basename(img_path))[0]
    raw_response_path = save_raw_response(resp, paragraph_dir, base_filename)
    print(f"Saved raw API response to {raw_response_path}")

    # Try to find fullTextAnnotation in response
    fta = find_full_text_annotation(resp)
    if not fta:
        print("Could not find 'fullTextAnnotation' in response. Response keys:", list(resp.keys()))
        raise SystemExit("No OCR data found in Digivision response.")

    # Extract and save all paragraphs (unchanged)
    json_path, simplified_path, txt_path = extract_and_save_paragraphs(fta, paragraph_dir, ocr_text_dir, base_filename)
    print(f"Saved complete paragraph data to {json_path}")
    print(f"Saved simplified paragraph data to {simplified_path}")
    print(f"Saved extracted text to {txt_path}")

    # Process the response to create sorted text file (in OCR_SORTED folder) -- NOW WORD-LEVEL
    # Tune these parameters as needed:
    TOLERANCE_MULTIPLIER = 0.6   # try 0.5, 0.6, 0.7 etc.
    MIN_TOL_PX = 10
    USE_BASELINE = True

    sorted_text_path = process_sorted_text_from_response(resp, ocr_sorted_dir, base_filename,
                                                         tolerance_multiplier=TOLERANCE_MULTIPLIER,
                                                         min_tolerance_px=MIN_TOL_PX,
                                                         use_baseline=USE_BASELINE)
    if sorted_text_path:
        print(f"Saved sorted text to {sorted_text_path}")

    # Convert Digivision fullTextAnnotation -> texts & boxes per-paragraph/line (keeps this for backward compatibility)
    # But for annotation we want word-level boxes so we'll extract words and draw them
    texts_par, boxes_par = text_and_boxes_from_fulltext(fta)

    # Also extract words to draw word-level rectangles (this does not write extra json)
    words = extract_words_from_fulltext(fta)
    # Prepare lists to pass to draw_annotations while keeping original filename naming
    texts_for_draw = [w["text"] for w in words]
    boxes_for_draw = [w["vertices"] for w in words]

    # Ensure boxes list length matches texts by padding (keeps original behavior if necessary)
    while len(boxes_for_draw) < len(texts_for_draw):
        boxes_for_draw.append([])

    # draw annotations on the image (in OCR_IMAGE folder) - filename unchanged
    annotated_image_path = os.path.join(ocr_image_dir, f"{base_filename}_ocr.png")
    annotated = draw_annotations(img, texts_for_draw, boxes_for_draw, out_path=annotated_image_path)
    print(f"Saved annotated image to {annotated}")

    print("Done.")
