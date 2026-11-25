"""Build the final teacher distributions (run LOCALLY, no GPU needed).

Why: asking the 3B VLM to pick from a numbered text list works for amounts but fails for the vendor
(it picks owner-name / address lines). Reading the image and answering in free text works much better,
so we map the teacher's free-form answer onto the student's candidates:
  vendor     : softmax over string-similarity(candidate, teacher text)          (tau = sharpness)
  date/total : mass on candidates whose value equals the teacher's value,
               optionally blended with the teacher's candidate-scoring distribution (--scores, --alpha)
All distributions are smoothed with eps*uniform. Prints teacher accuracy vs the TRAIN annotations.
python soft_from_json.py --data kd_data --split train --scores kd_data/teacher_soft_train.json --out kd_data/teacher_soft_train_v2.json
"""
import argparse, json, math
from pathlib import Path
import numpy as np
from sroie_data import FIELDS, similarity, values_equal, clean_vendor


def entropy(p):
    p = np.asarray(p); p = p[p > 0]
    return float(-(p * np.log(p)).sum() / math.log(max(len(p), 2)))


def map_to_candidates(field, value, cands, tau=0.05):
    """teacher free-form value -> distribution over candidates, or None if it cannot be mapped."""
    if not value or not cands:
        return None
    if field == "vendor":
        s = np.array([similarity(clean_vendor(str(value)), c["value"]) for c in cands])
        if s.max() < 0.6:
            return None
        e = np.exp((s - s.max()) / tau)
        return e / e.sum()
    m = np.array([float(values_equal(c["value"], value, field)) for c in cands])
    return m / m.sum() if m.sum() > 0 else None


def build(records, tjson, scores=None, tau=0.05, alpha=0.5, eps=0.02):
    out, stats = {}, {f: [0, 0, 0] for f in FIELDS}      # [ok, mappable, n_targets_present]
    for r in records:
        rid = r["receipt_id"]
        if rid not in tjson:
            continue
        out[rid] = {}
        for f in FIELDS:
            d = r["fields"][f]; cands = d["candidates"]
            if len(cands) < 2:
                continue
            q = map_to_candidates(f, (tjson[rid].get(f) or {}).get("value"), cands, tau)
            sp = None
            if scores and rid in scores and f in scores[rid]:
                sp = np.asarray(scores[rid][f]["probs"], float)
                sp = np.pad(sp, (0, len(cands) - len(sp)))[:len(cands)]
                sp = sp / sp.sum() if sp.sum() > 0 else None
            if q is None and sp is None:
                continue
            if q is not None and sp is not None and f != "vendor":
                q = alpha * q + (1 - alpha) * sp
            elif q is None:
                q = sp
            q = (1 - eps) * q + eps / len(cands)
            out[rid][f] = {"probs": q.tolist(), "entropy": entropy(q), "argmax": int(q.argmax())}
            if d["targets"]:
                stats[f][1] += 1; stats[f][0] += int(q.argmax() in d["targets"])
    return out, stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="kd_data"); ap.add_argument("--split", default="train")
    ap.add_argument("--scores", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=0.05); ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--eps", type=float, default=0.02)
    a = ap.parse_args()
    recs = json.load(open(Path(a.data) / f"{a.split}.json", encoding="utf-8"))
    tj = json.load(open(Path(a.data) / f"teacher_json_{a.split}.json", encoding="utf-8"))
    sc = json.load(open(a.scores, encoding="utf-8")) if a.scores else None
    out, st = build(recs, tj, sc, a.tau, a.alpha, a.eps)
    json.dump(out, open(a.out, "w"))
    print(f"receipts with teacher labels: {len(out)}")
    for f in FIELDS:
        ok, n, _ = st[f]
        print(f"teacher top-1 vs train annotations  {f:7s}: {ok}/{n}" + (f" = {ok/n:.1%}" if n else ""))
