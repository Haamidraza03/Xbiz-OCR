# app.py
"""
PaddleOCR example (uses ocr.predict to avoid deprecation warning).
Creates <imagename>.txt next to each processed image with the recognized text.
"""

import argparse
from pathlib import Path
from paddleocr import PaddleOCR
import json

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}


def extract_texts_from_predict(result):
    """
    Robust extractor for various nested shapes returned by PaddleOCR.predict().
    Returns list of text strings in detection order.
    """
    texts = []

    def rec(obj):
        # If it's a string, treat as text
        if isinstance(obj, str):
            if obj.strip():
                texts.append(obj.strip())
            return

        # If numeric, skip
        if isinstance(obj, (int, float)):
            return

        # If tuple/list, try some likely patterns first
        if isinstance(obj, (list, tuple)):
            # Pattern: (text, confidence)  -> (str, float)
            if len(obj) == 2 and isinstance(obj[0], str) and isinstance(obj[1], (int, float)):
                if obj[0].strip():
                    texts.append(obj[0].strip())
                return

            # Pattern: [box, (text, conf)] or [box, [ (text,conf), ... ]]
            if len(obj) >= 2:
                # if second element is (text, conf)
                sec = obj[1]
                if isinstance(sec, (list, tuple)) and len(sec) >= 1 and isinstance(sec[0], str):
                    # sec may be (text, conf) or [ (text, conf), ... ]
                    if isinstance(sec[0], str):
                        if sec[0].strip():
                            texts.append(sec[0].strip())
                        return
                    else:
                        # iterate inside sec
                        for item in sec:
                            rec(item)
                        return

            # Otherwise, recursively walk children
            for item in obj:
                rec(item)
            return

        # If dict, inspect values
        if isinstance(obj, dict):
            for v in obj.values():
                rec(v)
            return

        # Anything else - ignore
        return

    rec(result)
    # keep unique-ish but preserve order: (not removing duplicates aggressively)
    cleaned = [t for t in texts if t]
    return cleaned


def ocr_image_to_txt(ocr, image_path: Path, out_path: Path = None):
    """
    Run PaddleOCR.predict on a single image and save results to a .txt file.
    Uses ocr.predict(...) (no cls kwarg) to avoid deprecation/TypeError issues.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Use predict (recommended) to avoid deprecation warning.
    # predict may return nested lists/tuples; we parse it robustly below.
    result = ocr.predict(str(image_path))

    texts = extract_texts_from_predict(result)

    # If we couldn't extract text, write a debug dump so you can inspect result
    if not texts:
        debug_txt = f"# No text extracted by parser. Raw predict() output below (JSON-ish):\n\n{json.dumps(result, default=str, ensure_ascii=False, indent=2)}\n"
        if out_path is None:
            out_path = image_path.with_suffix(".txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(debug_txt, encoding="utf-8")
        print(f"Warning: no text extracted for {image_path.name}. Wrote debug dump to {out_path}")
        return out_path

    if out_path is None:
        out_path = image_path.with_suffix(".txt")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(texts), encoding="utf-8")
    return out_path


def batch_ocr_folder(ocr, images_folder: Path, recursive: bool = False):
    if not images_folder.exists() or not images_folder.is_dir():
        raise FileNotFoundError(f"Images folder not found: {images_folder}")

    if recursive:
        files = [p for p in images_folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXT]
    else:
        files = [p for p in images_folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXT]

    if not files:
        print("No supported image files found in", images_folder)
        return

    for img in sorted(files):
        try:
            out = ocr_image_to_txt(ocr, img)
            print(f"Processed: {img.name} -> {out.name}")
        except Exception as e:
            print(f"Failed to process {img.name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="PaddleOCR (predict) - single image or folder -> .txt files")
    parser.add_argument("--image", "-i", help="Path to single image (single-image mode).")
    parser.add_argument("--folder", "-f", help="Path to images folder for batch OCR (one .txt per image).")
    parser.add_argument("--lang", "-l", default="en", help="Recognition language(s), e.g. 'en' or 'en|ch'.")
    parser.add_argument("--no-cls", action="store_true", help="Disable angle classifier (use_angle_cls=False).")
    parser.add_argument("--recursive", action="store_true", help="Recursively search folder for images.")
    args = parser.parse_args()

    print("Initializing PaddleOCR (this may download models on first run)...")
    ocr = PaddleOCR(use_angle_cls=not args.no_cls, lang=args.lang)

    if args.image:
        img_path = Path(args.image)
        out_path = ocr_image_to_txt(ocr, img_path)
        print("Saved text to:", out_path)
    elif args.folder:
        folder = Path(args.folder)
        batch_ocr_folder(ocr, folder, recursive=args.recursive)
    else:
        default_folder = Path(__file__).parent / "images"
        print("No --image or --folder provided. Using default folder:", default_folder)
        batch_ocr_folder(ocr, default_folder, recursive=args.recursive)


if __name__ == "__main__":
    main()
