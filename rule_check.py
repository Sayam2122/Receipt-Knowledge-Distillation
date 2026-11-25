"""Leakage check for the vendor-registration rule, using TRAIN data only (test.json is never opened).
1) lists the text that trails the annotated vendor name in the OCR line (what a rule must remove),
2) counts on train / on the held-out 15% of train how often clean_vendor() fixes or breaks the vendor line.
python rule_check.py --data kd_data"""
import argparse, json, collections, re
from pathlib import Path
from sroie_data import norm_text, clean_vendor, VENDOR_WINDOW

ap = argparse.ArgumentParser(); ap.add_argument("--data", default="kd_data"); a = ap.parse_args()
recs = sorted(json.load(open(Path(a.data) / "train.json", encoding="utf-8")), key=lambda r: r["receipt_id"])
cut = int(len(recs) * 0.85)
suffixes = collections.Counter(); stats = {"train": [0, 0, 0, 0], "heldout": [0, 0, 0, 0]}   # n, fixed, broken, still-wrong
examples = []
for i, r in enumerate(recs):
    gt = (r.get("ground_truth") or {}).get("vendor")
    if not gt or not norm_text(gt):
        continue
    lines = [b["text"] for b in r["boxes"][:VENDOR_WINDOW]]
    g = norm_text(gt)
    # the OCR line that starts with the annotated name, if any
    line = next((t for t in lines if norm_text(t).startswith(g)), None)
    if line is None:
        continue
    for name, grp in (("train", i < cut), ("heldout", i >= cut)):
        if not grp:
            continue
        s = stats[name]; s[0] += 1
        raw_ok, cl_ok = norm_text(line) == g, norm_text(clean_vendor(line)) == g
        s[1] += (not raw_ok) and cl_ok
        s[2] += raw_ok and (not cl_ok)
        s[3] += (not raw_ok) and (not cl_ok)
    if norm_text(line) != g:
        tail = norm_text(line)[len(g):]
        suffixes[re.sub(r"[0-9]", "9", re.sub(r"[a-z]", "a", tail))] += 1
        if len(examples) < 12:
            examples.append((line, gt))
print("Shapes of the text trailing the annotated vendor (digits->9, letters->a):")
for k, v in suffixes.most_common(10):
    print(f"  {v:3d}  {k}")
print("\nExamples (OCR line | annotation):")
for l, g in examples:
    print("  ", repr(l), "|", repr(g))
for name, (n, fx, br, wr) in stats.items():
    print(f"\n[{name}] vendor lines with the name at the start: {n} | rule fixes: {fx} | rule breaks: {br} | still wrong after rule: {wr}")
