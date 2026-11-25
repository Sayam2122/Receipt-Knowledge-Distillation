"""CLI inference on ANY receipt image (OCR -> student -> JSON). No teacher, no SROIE files needed.
python predict.py --ckpt runs/kd/seed0.pt --image my_receipt.jpg [--save-vis out.jpg]"""
import argparse, json, time
from extract import ReceiptExtractor
from visualize import draw

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--image", required=True); ap.add_argument("--save-vis"); ap.add_argument("--show-ocr", action="store_true")
a = ap.parse_args()
ex = ReceiptExtractor(a.ckpt)
t = time.time(); res, boxes = ex.from_image(a.image); dt = time.time() - t
if a.show_ocr:
    for b in boxes: print(f"[row {b['row']}] {b['text']!r}")
print(json.dumps({**ReceiptExtractor.simple(res), "confidence": {f: res[f]["confidence"] for f in ("vendor", "date", "total")}}, indent=2)); print(f"end-to-end latency (OCR+student): {dt*1000:.0f} ms")
if a.save_vis:
    draw(a.image, {"student": [(f, res[f]["bbox"]) for f in ("vendor", "date", "total")]}).save(a.save_vis)
