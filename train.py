"""Step 3: train the student. Same code path for the two arms of the ablation:
  --mode ce   : annotations only  (the 'student without the teacher' baseline)
  --mode kd   : CE + lambda*KL(teacher || student)
  --mode kd_conf : KD with teacher-reliability weighting (novelty idea, see README)
Matched comparison => both arms use --subset teacher (the receipts the teacher scored), same seed => same init + order.
python train.py --data kd_data --teacher kd_data/teacher_soft.json --mode kd --subset teacher --seeds 0 1 2 --out runs/kd"""
import argparse, json, random
from pathlib import Path
import numpy as np, torch
from sroie_data import FIELDS
from features import featurize, FEATURE_DIM
from student import CandidateStudent, kd_loss, n_params
from metrics import evaluate_student

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="kd_data"); ap.add_argument("--teacher", default=None)
ap.add_argument("--mode", choices=["ce", "kd", "kd_conf"], default="ce")
ap.add_argument("--subset", choices=["teacher", "all"], default="teacher")
ap.add_argument("--lam", type=float, default=0.5); ap.add_argument("--T", type=float, default=2.0)
ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--bs", type=int, default=16); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--annotated-n", type=int, default=None,
                help="label budget: only the first N (sorted by id) training receipts keep their annotations. "
                     "ce: trains on those N only. kd/kd_conf: teacher labels ALL receipts, annotations only for the first N (0 = teacher-only)")
ap.add_argument("--out", default="runs/run")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
data = Path(a.data); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
load = lambda s: json.load(open(data / f"{s}.json"))
train, val, test = load("train"), load("eval"), load("test")
if not val:   # official SROIE has no eval split -> hold out the LAST 15% of train (teacher scoring uses the FIRST N, so no overlap)
    train = sorted(train, key=lambda r: r["receipt_id"]); cut = int(len(train) * 0.85)
    train, val = train[:cut], train[cut:]
    print(f"no eval split found: holding out {len(val)} train receipts for model selection")
teacher = json.load(open(a.teacher)) if a.teacher else {}
if a.mode != "ce" and not teacher:
    raise SystemExit("--teacher required for kd modes")
if a.subset == "teacher":
    train = [r for r in train if r["receipt_id"] in teacher]
train = sorted(train, key=lambda r: r["receipt_id"])
if a.annotated_n is not None and a.mode == "ce":
    train = train[: a.annotated_n]
print(f"device={dev} train={len(train)} val={len(val)} test={len(test)} mode={a.mode}")


def groups(recs):
    gs = []
    for ridx, r in enumerate(recs):
        annotated = a.annotated_n is None or ridx < a.annotated_n
        for fid, f in enumerate(FIELDS):
            d = r["fields"][f]; K = len(d["candidates"])
            if not K:
                continue
            tp = np.zeros(K, np.float32); w = 1.0
            info = teacher.get(r["receipt_id"], {}).get(f) if a.mode != "ce" else None
            if info and K > 1:
                p = np.asarray(info["probs"], np.float32)[:K]
                if p.sum() > 0:
                    tp[:len(p)] = p / p.sum()
                    if a.mode == "kd_conf":
                        # reliability: low-entropy teacher (and, where the annotation is known, teacher argmax agrees with it)
                        ent = -(tp[tp > 0] * np.log(tp[tp > 0])).sum() / np.log(K)
                        w = float(1 - ent)
                        if annotated and d["targets"]:
                            w *= float(int(tp.argmax()) in d["targets"])
            tm = np.zeros(K, bool)
            if annotated and d["targets"]:
                tm[d["targets"]] = True
            if not tm.any() and tp.sum() == 0:
                continue
            gs.append((featurize(d["candidates"], f), fid, tm, tp, w))
    return gs


def collate(batch):
    K = max(len(b[0]) for b in batch); D = FEATURE_DIM; B = len(batch)
    x = np.zeros((B, K, D), np.float32); m = np.zeros((B, K), bool); tm = np.zeros((B, K), bool)
    tp = np.zeros((B, K), np.float32)
    for i, (f, fid, t, p, w) in enumerate(batch):
        k = len(f); x[i, :k] = f; m[i, :k] = True; tm[i, :k] = t; tp[i, :k] = p
    T = lambda z, dt=None: torch.tensor(z).to(dev)
    return (T(x), torch.tensor([b[1] for b in batch]).to(dev), T(m), T(tm), T(tp),
            torch.tensor([b[4] for b in batch], dtype=torch.float32).to(dev))


tr = groups(train)
print("training groups:", len(tr), "| with annotation:", sum(g[2].any() for g in tr), "| with teacher distribution:", sum(g[3].sum() > 0 for g in tr))
summary = []
for seed in a.seeds:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = CandidateStudent(FEATURE_DIM).to(dev)           # identical init for ce/kd at the same seed
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)
    rng = np.random.RandomState(seed); best, best_state, hist = -1, None, []
    for ep in range(a.epochs):
        model.train(); order = rng.permutation(len(tr)); tot = 0.0
        for i in range(0, len(order), a.bs):
            x, fid, m, tm, tp, w = collate([tr[j] for j in order[i:i + a.bs]])
            loss = kd_loss(model(x, fid), m, tm, tp, a.lam if a.mode != "ce" else 0.0, a.T, w)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        v = evaluate_student(model, val, featurize, dev)["mean_value_exact"]
        hist.append({"epoch": ep + 1, "loss": tot, "val_mean_value_exact": v})
        if v > best:
            best, best_state = v, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    res = evaluate_student(model, test, featurize, dev)
    torch.save({"state": best_state, "in_dim": FEATURE_DIM, "mode": a.mode, "seed": seed}, out / f"seed{seed}.pt")
    json.dump({"history": hist, "test": res, "val_best": best}, open(out / f"seed{seed}_results.json", "w"), indent=1)
    summary.append(res)
    print(f"seed {seed}: val={best:.3f} test mean value-exact={res['mean_value_exact']:.3f} "
          + " ".join(f"{f}={res[f]['value_exact']:.3f}/loc={res[f]['localisation_acc']:.3f}" for f in FIELDS))

agg = {"mode": a.mode, "params": n_params(model), "args": vars(a), "seeds": a.seeds}
for f in FIELDS:
    for k in ["value_exact", "localisation_acc"]:
        vals = [s[f][k] for s in summary]; agg[f"{f}_{k}"] = [float(np.mean(vals)), float(np.std(vals))]
vals = [s["mean_value_exact"] for s in summary]; agg["mean_value_exact"] = [float(np.mean(vals)), float(np.std(vals))]
json.dump(agg, open(out / "summary.json", "w"), indent=1)
print("SUMMARY (mean, std over seeds):", json.dumps({k: v for k, v in agg.items() if "_" in k and isinstance(v, list)}, indent=1))
