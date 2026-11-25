"""Step 1: SROIE -> cached candidate JSON for each split.
python prepare_data.py --dataset receipt_dataset --out kd_data [--box-source gt|ocr]"""
import argparse, json
from pathlib import Path
from sroie_data import prepare_split, FIELDS

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="receipt_dataset")
ap.add_argument("--out", default="kd_data")
ap.add_argument("--box-source", choices=["gt", "ocr"], default="gt")
ap.add_argument("--limit", type=int, default=None)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
for split in ["train", "eval", "test"]:
    recs = prepare_split(a.dataset, split, a.limit, a.box_source, out / "ocr_cache")
    json.dump(recs, open(out / f"{split}.json", "w"), ensure_ascii=False)
    n = len(recs)
    print(split, n, "receipts; GT value present among candidates:",
          {f: f"{sum(bool(r['fields'][f]['targets']) for r in recs)}/{n}" for f in FIELDS})
