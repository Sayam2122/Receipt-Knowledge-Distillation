"""Real-OCR evaluation: the SAME student, scored on SROIE test receipts using (a) SROIE's annotated boxes and
(b) boxes produced by RapidOCR from the raw image. Same receipts, same checkpoint -> the gap is the cost of real OCR.
python eval_ocr.py --ckpt runs/ce_all/seed0.pt --dataset receipt_dataset --limit 100
(RapidOCR takes ~5-7 s per receipt on CPU; results are cached in kd_ocr_eval/ocr_cache, so a re-run is fast.)"""
import argparse, json
from pathlib import Path
from extract import ReceiptExtractor
from features import featurize
from metrics import evaluate_student
from sroie_data import prepare_split, FIELDS

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--dataset", default="receipt_dataset")
ap.add_argument("--limit", type=int, default=100, help="first N test receipts (sorted by file name)")
ap.add_argument("--out", default="kd_ocr_eval")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
model = ReceiptExtractor(a.ckpt).model
res = {}
for src in ("gt", "ocr"):
    recs = prepare_split(a.dataset, "test", a.limit, src, out / "ocr_cache")
    res[src] = evaluate_student(model, recs, featurize)
    print(f"\n[{src.upper()} boxes] {len(recs)} test receipts")
    for f in FIELDS:
        m = res[src][f]; print(f"  {f:7s} value-exact {m['value_exact']*100:5.1f}%   localisation {m['localisation_acc']*100:5.1f}% (n={m['n_localisable']})")
    print(f"  mean value-exact {res[src]['mean_value_exact']*100:.1f}%")
json.dump(res, open(out / "ocr_vs_gt.json", "w"), indent=2)
print("\nsaved", out / "ocr_vs_gt.json")
