"""Leakage-controlled test score: drop every test receipt the vendor diagnostic could have shown us, score the rest.
The diagnostic scanned test.json in order and stopped after 15 vendor mismatches (registration-number lines).
We re-run that scan with the rule OFF, take everything it scanned (+20% safety margin) as 'seen', and report on the remainder.
python eval_unseen.py --data kd_data --ckpts runs/ce_all/seed0.pt runs/ce_all/seed1.pt runs/ce_all/seed2.pt"""
import argparse, json
from pathlib import Path
import numpy as np
import sroie_data
from sroie_data import make_record, values_equal, similarity
from extract import ReceiptExtractor
from features import featurize
from metrics import predict_record, evaluate_student

ap = argparse.ArgumentParser(); ap.add_argument("--data", default="kd_data"); ap.add_argument("--ckpts", nargs="+", required=True)
ap.add_argument("--margin", type=float, default=0.20); a = ap.parse_args()
recs = json.load(open(Path(a.data) / "test.json", encoding="utf-8"))
models = [ReceiptExtractor(c).model for c in a.ckpts]

# 1) re-create the diagnostic scan with the rule OFF (candidates keep the registration numbers)
keep = sroie_data.clean_vendor
sroie_data.clean_vendor = lambda t: t.strip(" ,;")
n_mis, scanned = 0, 0
for r in recs:
    scanned += 1
    raw = make_record(r["receipt_id"], r["boxes"], r["size"], r.get("ground_truth"))
    p = predict_record(models[0], raw, featurize)["vendor"]
    gt = (r.get("ground_truth") or {}).get("vendor", "")
    if p and not values_equal(p["value"], gt, "vendor") and similarity(p["value"], gt) >= 0.8:
        n_mis += 1
        if n_mis >= 15:
            break
sroie_data.clean_vendor = keep
seen = min(len(recs), int(np.ceil(scanned * (1 + a.margin))))
print(f"diagnostic scanned {scanned} test receipts to find 15 mismatches -> treating the first {seen} as seen")

# 2) score seen vs untouched subsets with the frozen rule (records in test.json already use it)
for name, sub in (("ALL test", recs), (f"SEEN (first {seen})", recs[:seen]), (f"UNTOUCHED ({len(recs) - seen} receipts)", recs[seen:])):
    res = [evaluate_student(m, sub, featurize) for m in models]
    line = "  ".join(f"{f} {100*np.mean([x[f]['value_exact'] for x in res]):.1f}" for f in ("vendor", "date", "total"))
    print(f"{name:34s} {line}   mean {100*np.mean([x['mean_value_exact'] for x in res]):.1f}%  ({len(models)} seed(s))")
