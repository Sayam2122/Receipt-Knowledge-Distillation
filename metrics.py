"""Prediction + evaluation. Localisation and value correctness are reported SEPARATELY."""
import numpy as np
import torch
from sroie_data import FIELDS, values_equal


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def gt_boxes(rec, field):
    d = rec["fields"][field]
    return [d["candidates"][i]["bbox"] for i in d["targets"]]


@torch.no_grad()
def predict_record(model, rec, featurize, device="cpu"):
    """Return {field: {value, bbox, box_ids, prob, idx}} (None if a field has no candidate)."""
    model.eval()
    out = {}
    for fid, f in enumerate(FIELDS):
        cands = rec["fields"][f]["candidates"]
        if not cands:
            out[f] = None
            continue
        x = torch.tensor(featurize(cands, f)).unsqueeze(0).to(device)
        p = torch.softmax(model(x, torch.tensor([fid], device=device))[0], -1).cpu()
        i = int(p.argmax())
        out[f] = {"value": cands[i]["value"], "bbox": cands[i]["bbox"], "box_ids": cands[i]["box_ids"],
                  "prob": float(p[i]), "idx": i}
    return out


def score_field(rec, field, pred_value, pred_bbox, iou_thr=0.5):
    """-> (value_exact, localised or None if no GT box exists)"""
    gt = rec["ground_truth"][field]
    v = values_equal(pred_value, gt, field)
    boxes = gt_boxes(rec, field)
    loc = None
    if boxes:
        loc = bool(pred_bbox is not None and max(iou(pred_bbox, b) for b in boxes) >= iou_thr)
    return v, loc


def summarize(rows):
    """rows: list of {field: (value_exact, loc)} -> per-field metrics."""
    res = {}
    for f in FIELDS:
        v = [r[f][0] for r in rows]
        l = [r[f][1] for r in rows if r[f][1] is not None]
        res[f] = {"n": len(v), "value_exact": float(np.mean(v)) if v else 0.0,
                  "n_localisable": len(l), "localisation_acc": float(np.mean(l)) if l else 0.0}
    res["mean_value_exact"] = float(np.mean([res[f]["value_exact"] for f in FIELDS]))
    return res


def evaluate_student(model, recs, featurize, device="cpu"):
    rows = []
    for r in recs:
        p = predict_record(model, r, featurize, device)
        rows.append({f: score_field(r, f, p[f]["value"] if p[f] else None, p[f]["bbox"] if p[f] else None)
                     for f in FIELDS})
    return summarize(rows)
