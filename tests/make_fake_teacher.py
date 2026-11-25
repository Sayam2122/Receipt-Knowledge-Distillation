"""TEST-ONLY stand-in for teacher_qwen.py: noisy soft labels built from the ground truth, to exercise the KD code path."""
import json, sys, numpy as np
d, split, out = sys.argv[1], sys.argv[2], sys.argv[3]
rng = np.random.RandomState(0); res = {}
for r in json.load(open(f"{d}/{split}.json")):
    res[r["receipt_id"]] = {}
    for f, v in r["fields"].items():
        K = len(v["candidates"])
        if not K: continue
        l = rng.randn(K) * 0.5
        if v["targets"]: l[v["targets"][0]] += 2.0
        p = np.exp(l) / np.exp(l).sum()
        res[r["receipt_id"]][f] = {"probs": p.tolist(), "entropy": 0.5, "argmax": int(p.argmax())}
json.dump(res, open(out, "w"))
