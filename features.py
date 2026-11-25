"""Layout-aware candidate features (the 'student input').

Why this differs from the original: the original only looked at the previous/next box in file order.
Receipts print label and amount on the same visual ROW, so we use row-aligned left context, the row
above, and (for totals) amount-rank features such as 'is the largest amount' / 'value printed twice'.
"""
import re
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sroie_data import FIELDS, norm_amount

D_OWN, D_CTX = 128, 64
_own = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4), n_features=D_OWN, alternate_sign=False, norm="l2")
_ctx = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4), n_features=D_CTX, alternate_sign=False, norm="l2")

KW = {
    "total": r"\btotal\b", "grand": r"grand\s*total", "due": r"\b(due|payable|balance)\b", "net": r"\bnett?\b",
    "sub": r"sub\s*-?\s*total", "cash": r"\b(cash|tender|paid|payment|visa|master|card|amex|debit)\b",
    "change": r"\bchange\b", "round": r"round", "tax": r"\b(gst|tax|sst|vat)\b",
    "qty": r"\b(qty|quantity|items?|pcs)\b", "disc": r"disc", "incl": r"incl", "rm": r"\brm\b",
}
KW_ABOVE = ["total", "sub", "cash", "change", "tax", "qty"]
VENDOR_NEG = r"\b(tel|fax|gst|invoice|receipt|tax|date|no\.?|reg|co\.? ?no|address|jalan|jln)\b"
VENDOR_POS = r"\b(sdn|bhd|enterprise|trading|restaurant|store|shop|mart|cafe|resources|industries|supermarket|hardware|bakery|stationery|pharmacy|centre|center)\b"
_FLAGS = [re.compile(v, re.I) for v in KW.values()]
_FL = {k: re.compile(v, re.I) for k, v in KW.items()}
_VN, _VP = re.compile(VENDOR_NEG, re.I), re.compile(VENDOR_POS, re.I)


def _scalars(c, i, cands, field):
    f = []
    x0, y0, x1, y1 = c["bbox"]
    f += [x0, y0, x1, y1, x1 - x0, y1 - y0, (x0 + x1) / 2, (y0 + y1) / 2]
    f += [c["pos"], c["row_rank"], min(c["row_n"], 6) / 6, float(c["row_rank"] >= 0.999),
          min(len(c["text"]), 60) / 60, min(len(c["value"]), 40) / 40]
    v = c["value"]
    f += [sum(ch.isdigit() for ch in v) / max(len(v), 1), sum(ch.isalpha() for ch in v) / max(len(v), 1),
          sum(ch.isupper() for ch in v) / max(sum(ch.isalpha() for ch in v), 1)]
    left, above = c["left"].lower(), c["above"].lower()
    f += [float(bool(r.search(left))) for r in _FLAGS]                      # 13 flags: label left of value
    f += [float(bool(_FL[k].search(above))) for k in KW_ABOVE]               # 6 flags: row above
    f += [float(bool(_FL["total"].search(c["text"].lower())))]               # label inside same box
    n = len(cands)
    f += [i / max(n - 1, 1), min(n, 40) / 40, float(i == 0), float(i == n - 1)]
    # ---- total-specific, amount-rank features (zeros for other fields)
    t = [0.0] * 6
    if field == "total":
        vals = [norm_amount(k["value"]) for k in cands]
        fv = [float(x) if x else 0.0 for x in vals]
        mine = fv[i]
        rank = sum(1 for x in fv if x > mine)
        same = sum(1 for x in vals if x == vals[i])
        t = [rank / max(n - 1, 1), float(rank == 0), float(rank == 1), min(same, 4) / 4,
             np.log1p(mine) / 10, float(mine == 0)]
    f += t
    # ---- vendor-specific
    vv = [0.0] * 4
    if field == "vendor":
        vv = [float(bool(_VN.search(v))), float(bool(_VP.search(v))), min(c["box_ids"][0], 20) / 20,
              float(len(c["box_ids"]) > 1)]
    f += vv
    f += [float(field == k) for k in FIELDS]
    return f


def featurize(cands, field):
    if not cands:
        return np.zeros((0, FEATURE_DIM), np.float32)
    own = _own.transform([c["value"] + " " + c["text"] for c in cands]).toarray()
    ctx = _ctx.transform([(c["left"] + " | " + c["above"]) for c in cands]).toarray()
    sc = np.array([_scalars(c, i, cands, field) for i, c in enumerate(cands)], np.float32)
    return np.concatenate([own, ctx, sc], 1).astype(np.float32)


def _dim():
    d = {"field": "total", "value": "1.00", "box_ids": [0], "bbox": [0, 0, 1, 1], "text": "t", "pos": 0.0,
         "left": "", "above": "", "row_text": "", "row_n": 1, "row_rank": 0.0}
    return D_OWN + D_CTX + len(_scalars(d, 0, [d], "total"))


FEATURE_DIM = _dim()
