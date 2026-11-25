"""Image -> OCR boxes in the same format as SROIE boxes (needed for UNSEEN images).
pip install rapidocr-onnxruntime   (light, CPU)  | fallback: pip install easyocr"""
import hashlib, json
from pathlib import Path
from PIL import Image
from sroie_data import make_box, assign_rows

_engine = None


def _rapid(path):
    global _engine
    import numpy as np
    from rapidocr_onnxruntime import RapidOCR
    _engine = _engine or RapidOCR()
    im = Image.open(path).convert("RGB")
    if im.width < 900:                       # small photos: upscale so the OCR keeps word gaps (boxes are normalised)
        s = 1200 / im.width
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    res, _ = _engine(np.array(im)[:, :, ::-1])
    return [(r[0], r[1]) for r in (res or [])], im.size


def _easy(path):
    global _engine
    import easyocr
    _engine = _engine or easyocr.Reader(["en"], gpu=False)
    return [(q, t) for q, t, _ in _engine.readtext(str(path))], Image.open(path).size


def run_ocr(path, cache_dir=None):
    path = Path(path)
    cp = None
    if cache_dir:
        cp = Path(cache_dir) / (hashlib.md5(str(path.resolve()).encode()).hexdigest() + ".json")
        if cp.exists():
            return json.load(open(cp))
    try:
        raw, (W, H) = _rapid(path)           # W, H = size of the image the OCR actually saw (may be upscaled)
    except ImportError:
        raw, (W, H) = _easy(path)
    boxes = []
    for quad, text in raw:
        if text.strip():
            boxes.append(make_box(len(boxes), text, [p[0] for p in quad], [p[1] for p in quad], W, H))
    boxes = assign_rows(boxes)
    boxes.sort(key=lambda b: (b["row"], b["x0"]))
    for i, b in enumerate(boxes):
        b["idx"] = i
    if cp:
        cp.parent.mkdir(parents=True, exist_ok=True); json.dump(boxes, open(cp, "w"))
    return boxes
