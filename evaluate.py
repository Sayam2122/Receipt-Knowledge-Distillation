"""Step 4: held-out evaluation + side-by-side GT / teacher / student table + annotated images.
python evaluate.py --data kd_data --dataset receipt_dataset --split test --ckpt runs/kd/seed0.pt \
       --ce-ckpt runs/ce/seed0.pt --teacher-json kd_data/teacher_json_test.json --out report_eval --n-images 12"""
import argparse, csv, json
from pathlib import Path
import torch
from sroie_data import FIELDS
from features import featurize
from extract import ReceiptExtractor
from metrics import predict_record, score_field, summarize, iou, gt_boxes
from visualize import draw

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="kd_data"); ap.add_argument("--dataset", default="receipt_dataset")
ap.add_argument("--split", default="test"); ap.add_argument("--ckpt", required=True)
ap.add_argument("--ce-ckpt", default=None, help="annotation-only student for the 'what does the teacher add' column")
ap.add_argument("--teacher-json", default=None, help="output of teacher_qwen.py --mode json on the same split")
ap.add_argument("--only-teacher-covered", action="store_true", help="score only receipts the teacher JSON covers (fair 3-way table)")
ap.add_argument("--teacher-box", choices=["raw", "matched"], default="raw",
                help="raw = Qwen's own bbox_2d; matched = box of the OCR candidate matching Qwen's text")
ap.add_argument("--out", default="report_eval"); ap.add_argument("--n-images", type=int, default=12)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
recs = json.load(open(Path(a.data) / f"{a.split}.json"))
kd = ReceiptExtractor(a.ckpt).model
ce = ReceiptExtractor(a.ce_ckpt).model if a.ce_ckpt else None
teacher = json.load(open(a.teacher_json, encoding="utf-8")) if a.teacher_json else {}
if a.only_teacher_covered and teacher:
    recs = [r for r in recs if r["receipt_id"] in teacher]
print("evaluating", len(recs), "receipts")

def teacher_pred(rid, f, rec=None):
    """Teacher value + box. raw: Qwen's own box (None if it gave none). matched: box of the OCR candidate matching its text."""
    t = teacher.get(rid, {}).get(f)
    if not t:
        return None, None
    val, bb = t.get("value"), t.get("bbox")
    if a.teacher_box == "matched" and val and rec is not None:
        from soft_from_json import map_to_candidates
        q = map_to_candidates(f, val, rec["fields"][f]["candidates"])
        bb = rec["fields"][f]["candidates"][int(q.argmax())]["bbox"] if q is not None else None
    return val, bb

rows = {"student_kd": [], "student_ce": [], "teacher": []}
lines = []
for n, r in enumerate(recs):
    pk = predict_record(kd, r, featurize)
    pc = predict_record(ce, r, featurize) if ce else None
    row_k, row_c, row_t = {}, {}, {}
    for f in FIELDS:
        gt = r["ground_truth"][f]
        row_k[f] = score_field(r, f, pk[f]["value"] if pk[f] else None, pk[f]["bbox"] if pk[f] else None)
        line = {"receipt": r["receipt_id"], "field": f, "gt_value": gt,
                "student_kd_value": pk[f]["value"] if pk[f] else "", "student_kd_value_ok": row_k[f][0], "student_kd_loc_ok": row_k[f][1]}
        if pc:
            row_c[f] = score_field(r, f, pc[f]["value"] if pc[f] else None, pc[f]["bbox"] if pc[f] else None)
            line.update(student_ce_value=pc[f]["value"] if pc[f] else "", student_ce_value_ok=row_c[f][0], student_ce_loc_ok=row_c[f][1])
        if teacher:
            tv, tb = teacher_pred(r["receipt_id"], f, r)
            row_t[f] = score_field(r, f, tv, tb)
            line.update(teacher_value=tv, teacher_value_ok=row_t[f][0], teacher_loc_ok=row_t[f][1])
        lines.append(line)
    rows["student_kd"].append(row_k)
    if pc: rows["student_ce"].append(row_c)
    if teacher: rows["teacher"].append(row_t)
    if n < a.n_images:
        src = {"gt": [(f, b) for f in FIELDS for b in gt_boxes(r, f)],
               "student": [(f, pk[f]["bbox"]) for f in FIELDS if pk[f]]}
        if teacher:
            src["teacher"] = [(f, teacher_pred(r["receipt_id"], f, r)[1]) for f in FIELDS]
        draw(Path(a.dataset) / r["image"], src).save(out / f"{r['receipt_id']}_boxes.jpg")

with open(out / "side_by_side.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=sorted({k for l in lines for k in l}, key=lambda k: (k not in ("receipt", "field", "gt_value"), k)))
    w.writeheader(); w.writerows(lines)
summ = {k: summarize(v) for k, v in rows.items() if v}
json.dump(summ, open(out / "metrics.json", "w"), indent=1)
print(f"{'model':12s} {'field':7s} {'value-exact':>12s} {'localisation':>13s}")
for k, s in summ.items():
    for f in FIELDS:
        print(f"{k:12s} {f:7s} {s[f]['value_exact']:12.3f} {s[f]['localisation_acc']:13.3f}")
