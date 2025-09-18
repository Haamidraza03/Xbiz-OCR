#!/usr/bin/env python3
"""
digivision_extract_enhanced.py

Enhanced version of digivision_extract.py.

Features:
- Writes paragraph text files to <out>/paragraphs/paragraph_###.txt
- Writes a combined all_texts.txt and full_text.txt (if available)
- Writes per-paragraph element JSON to <out>/elements/element_###.json
- Writes per-paragraph raw JSON to <out>/paragraphs_json/paragraph_###.json
- Writes per-paragraph confidence to <out>/confidences/conf_###.txt
- Writes per-paragraph vertices to <out>/vertices/vertices_###.json
- Saves the base64 image blob to <out>/document_blob.b64
- Robust look-up for fullTextAnnotation in varied response wrappers

Usage examples:
    python digivision_extract_enhanced.py --image ./adharsamp.png --txn TXN0001 --case case001 --out ./out_folder
    python digivision_extract_enhanced.py --base64 "<BASE64>" --doc .PNG --out ./out_folder
"""

import os
import sys
import json
import base64
import argparse
import datetime
from typing import Any, Dict, List, Optional

import requests  # ensure 'requests' is installed: pip install requests

API_URL = "https://bankdevapi.digivision.ai/digivision/ai/rawtext-extraction"


def encode_image_file_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def prepare_payload(
    txn_id: str,
    document_name: str,
    doc_type: str,
    case_no: str,
    document_blob_b64: str,
) -> Dict[str, Any]:
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
    try:
        return resp.json()
    except ValueError:
        raise RuntimeError("Response was not valid JSON: " + resp.text)


def find_full_text_annotation(resp_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Try common places where fullTextAnnotation might be in the returned JSON.
    Returns a dict-like object representing the fullTextAnnotation (or a dict
    containing 'pages'/'text'), or None if not found.
    """
    if not isinstance(resp_json, dict):
        return None

    # Direct
    if "fullTextAnnotation" in resp_json and isinstance(resp_json["fullTextAnnotation"], dict):
        return resp_json["fullTextAnnotation"]

    # Common wrapper keys
    for key in ("data", "response", "result", "payload", "results", "Data"):
        wrapper = resp_json.get(key)
        if isinstance(wrapper, dict) and "fullTextAnnotation" in wrapper:
            return wrapper["fullTextAnnotation"]
        if isinstance(wrapper, list):
            for item in wrapper:
                if isinstance(item, dict) and "fullTextAnnotation" in item:
                    return item["fullTextAnnotation"]

    # fallback: look recursively for a dict that has 'pages' or 'text'
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
    # Some APIs put the paragraph text in 'text' or 'description'
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


def average_confidence_from_paragraph(paragraph: Dict[str, Any]) -> Optional[float]:
    if paragraph.get("confidence") is not None:
        try:
            return float(paragraph.get("confidence"))
        except Exception:
            pass

    word_confs: List[float] = []
    for word in paragraph.get("words", []):
        try:
            if word.get("confidence") is not None:
                word_confs.append(float(word.get("confidence")))
                continue
        except Exception:
            pass
        for s in word.get("symbols", []):
            if s.get("confidence") is not None:
                try:
                    word_confs.append(float(s.get("confidence")))
                except Exception:
                    pass

    if word_confs:
        return sum(word_confs) / len(word_confs)
    return None


def get_vertices_from_bounding_box(bbox: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(bbox, dict):
        return None
    return bbox.get("vertices") or bbox.get("normalizedVertices")


def extract_paragraphs_and_elements(
    full_text_annotation: Dict[str, Any],
    output_dir: str,
    document_blob_b64: Optional[str] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    paragraphs_dir = os.path.join(output_dir, "paragraphs")
    elements_dir = os.path.join(output_dir, "elements")
    confidences_dir = os.path.join(output_dir, "confidences")
    vertices_dir = os.path.join(output_dir, "vertices")
    paragraphs_json_dir = os.path.join(output_dir, "paragraphs_json")

    os.makedirs(paragraphs_dir, exist_ok=True)
    os.makedirs(elements_dir, exist_ok=True)
    os.makedirs(confidences_dir, exist_ok=True)
    os.makedirs(vertices_dir, exist_ok=True)
    os.makedirs(paragraphs_json_dir, exist_ok=True)

    # Save full_text if present
    fulltext = full_text_annotation.get("text")
    if fulltext:
        with open(os.path.join(output_dir, "full_text.txt"), "w", encoding="utf-8") as f:
            f.write(str(fulltext))

    # Save document blob if provided
    if document_blob_b64:
        try:
            with open(os.path.join(output_dir, "document_blob.b64"), "w", encoding="utf-8") as bf:
                bf.write(document_blob_b64)
        except Exception:
            pass

    all_text_path = os.path.join(output_dir, "all_texts.txt")
    # create/clear the combined file
    with open(all_text_path, "w", encoding="utf-8") as _:
        pass

    count = 0
    pages = full_text_annotation.get("pages") or []
    # If paragraphs may live directly under full_text_annotation
    if not pages:
        direct_paragraphs = full_text_annotation.get("paragraphs") or []
        if direct_paragraphs:
            pages = [{"blocks": [{"paragraphs": direct_paragraphs}]}]

    for page_idx, page in enumerate(pages, start=1):
        blocks = page.get("blocks", []) or []
        for block_idx, block in enumerate(blocks, start=1):
            paragraphs = block.get("paragraphs", []) or []
            for para_idx, para in enumerate(paragraphs, start=1):
                count += 1

                # Save raw paragraph JSON
                pjpath = os.path.join(paragraphs_json_dir, f"paragraph_{count:03d}.json")
                try:
                    with open(pjpath, "w", encoding="utf-8") as pjf:
                        json.dump(para, pjf, indent=2, ensure_ascii=False)
                except Exception:
                    # fallback to string representation
                    try:
                        with open(pjpath, "w", encoding="utf-8") as pjf:
                            pjf.write(str(para))
                    except Exception:
                        pass

                para_text = reconstruct_text_from_paragraph(para)
                para_conf = average_confidence_from_paragraph(para)
                para_desc = para.get("description") or para.get("text") or ""
                bbox = para.get("boundingBox") or para.get("bounding_box") or {}
                vertices = get_vertices_from_bounding_box(bbox)

                # Paragraph text file
                ppath = os.path.join(paragraphs_dir, f"paragraph_{count:03d}.txt")
                with open(ppath, "w", encoding="utf-8") as pf:
                    pf.write(para_text or "")

                # Append to combined text file
                with open(all_text_path, "a", encoding="utf-8") as allf:
                    allf.write(f"--- paragraph_{count:03d} (page {page_idx} block {block_idx}) ---\n")
                    allf.write((para_text or "") + "\n\n")

                # Element JSON
                elem = {
                    "text": para_text,
                    "confidence": para_conf,
                    "vertices": vertices,
                    "description": para_desc,
                }
                epath = os.path.join(elements_dir, f"element_{count:03d}.json")
                with open(epath, "w", encoding="utf-8") as ef:
                    json.dump(elem, ef, indent=2, ensure_ascii=False)

                # Confidence file
                cpath = os.path.join(confidences_dir, f"conf_{count:03d}.txt")
                with open(cpath, "w", encoding="utf-8") as cf:
                    cf.write(str(para_conf) if para_conf is not None else "null")

                # Vertices file
                vpath = os.path.join(vertices_dir, f"vertices_{count:03d}.json")
                with open(vpath, "w", encoding="utf-8") as vf:
                    json.dump(vertices or [], vf, indent=2, ensure_ascii=False)

    print(f"Extracted {count} paragraphs into '{paragraphs_dir}' and element files into '{elements_dir}'.")
    print(f"Raw paragraph JSON saved at: {paragraphs_json_dir}")
    print(f"Combined text saved at: {all_text_path}")
    if fulltext:
        print(f"Full text saved at: {os.path.join(output_dir, 'full_text.txt')}")
    if document_blob_b64:
        print(f"Document blob saved at: {os.path.join(output_dir, 'document_blob.b64')}")


def robust_extract_and_save(
    resp_json: Dict[str, Any],
    base_output_dir: Optional[str] = None,
    document_blob_b64: Optional[str] = None,
) -> None:
    fta = find_full_text_annotation(resp_json)
    if not fta:
        # be explicit about keys for easier debugging
        keys = ", ".join(sorted(resp_json.keys())) if isinstance(resp_json, dict) else str(type(resp_json))
        raise RuntimeError("Could not find 'fullTextAnnotation' in the API response. Response keys: " + keys)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = base_output_dir or f"digivision_output_{ts}"
    extract_paragraphs_and_elements(fta, base_dir, document_blob_b64=document_blob_b64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Call Digivision OCR endpoint and extract paragraphs/confidence/vertices."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", "-i", help="Path to local image file to encode as base64.")
    group.add_argument("--base64", "-b", help="Provide the base64 image string directly.")
    parser.add_argument("--txn", default="TXN0000001", help="Transaction ID (txnId).")
    parser.add_argument(
        "--doc",
        default=None,
        help="doctType (e.g. .JPG or .PNG). If omitted and --image given, derived from file extension.",
    )
    parser.add_argument("--name", default=None, help="documentName (defaults to filename or 'document').")
    parser.add_argument("--case", default="case001", help="caseNo.")
    parser.add_argument("--api", default=API_URL, help="API URL (default is the bankdevapi endpoint).")
    parser.add_argument("--out", default=None, help="Base output folder. If omitted a timestamped folder is created.")
    args = parser.parse_args()

    # prepare base64 blob
    document_blob: Optional[str] = None
    if args.image:
        if not os.path.exists(args.image):
            print("Image file not found:", args.image, file=sys.stderr)
            sys.exit(2)
        doc_type = args.doc or os.path.splitext(args.image)[1] or ".JPG"
        document_name = args.name or os.path.basename(args.image)
        document_blob = encode_image_file_to_base64(args.image)
    else:
        # base64 provided directly - doctype must be provided or default to .JPG
        document_blob = args.base64
        doc_type = args.doc or ".JPG"
        document_name = args.name or "document"

    payload = prepare_payload(
        txn_id=args.txn,
        document_name=document_name,
        doc_type=doc_type,
        case_no=args.case,
        document_blob_b64=document_blob or "",
    )

    print("Sending request to", args.api)
    try:
        resp_json = post_to_api(payload, url=args.api)
    except Exception as e:
        print("Error while posting to API:", str(e), file=sys.stderr)
        sys.exit(3)

    # Save full response for debugging
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_fname = f"raw_response_{ts}.json"
    with open(debug_fname, "w", encoding="utf-8") as df:
        json.dump(resp_json, df, indent=2, ensure_ascii=False)
    print("Saved raw response to", debug_fname)

    # Extract paragraphs/elements and write text files
    try:
        robust_extract_and_save(resp_json, base_output_dir=args.out, document_blob_b64=document_blob)
    except Exception as e:
        print("Failed to extract 'fullTextAnnotation':", str(e), file=sys.stderr)
        sys.exit(4)

    print("Done.")


if __name__ == "__main__":
    main()
