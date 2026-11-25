"""SROIE-style data handling: OCR boxes -> field candidates -> ground-truth alignment.

Expected dataset layout (same as your original project):
    receipt_dataset/{train,eval,test}/{img/*.jpg, box/*.txt, entities/*.txt}
Box file line : x1,y1,x2,y2,x3,y3,x4,y4,TEXT        Entities file: JSON {company,date,address,total}
"""
import json
import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from pathlib import Path

from PIL import Image

FIELDS = ["vendor", "date", "total"]
VENDOR_WINDOW = 12  # vendor is almost always in the first lines of the receipt

AMOUNT_RE = re.compile(r"(?<![\d.])(?:RM\s*)?(\d[\d,]*\.\d{2})(?!\d)", re.I)
_MONTHS = "JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC"
DATE_RE = re.compile(
    rf"(?<!\d)\d{{1,2}}[-/.]\d{{1,2}}[-/.](?:(?:19|20)\d{{2}}|\d{{2}}(?!\d))"      # 15/01/2019, also glued to a time: 15/01/201911:05
    rf"|(?<!\d)\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}(?!\d)"                          # 2019-01-15
    rf"|(?<!\d)\d{{1,2}}[\s-]*(?:{_MONTHS})[A-Z]*[\s-]*(?:(?:19|20)\d{{2}}|\d{{2}}(?!\d))",   # 12 JAN 2019
    re.I,
)


# ------------------------------------------------------------------ text helpers
def norm_text(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def norm_amount(s):
    t = re.sub(r"[^0-9.]", "", str(s))          # drops RM / $ / commas / spaces ("RM12.5" -> "12.5")
    try:
        return str(Decimal(t).quantize(Decimal("0.01")))
    except InvalidOperation:
        return None


def similarity(a, b):
    a, b = norm_text(a), norm_text(b)
    if not a or not b:
        return 0.0
    return 1.0 if a == b else SequenceMatcher(None, a, b).ratio()


def values_equal(pred, gt, field):
    """Strict (normalised) exact match used for every reported 'value' metric."""
    if pred is None or gt is None or not str(gt).strip():
        return False
    if field == "total":
        a, b = norm_amount(pred), norm_amount(gt)
        return a is not None and a == b
    return norm_text(pred) == norm_text(gt)


_REG = re.compile(r"[\s,;]*\(?\s*\d{5,7}\s*-?\s*[A-Za-z]\s*\)?[\s,;.]*$")
_TRAIL_CODE = re.compile(r"[\s,;]*\(\s*[A-Za-z0-9]{1,4}\s*\)[\s,;]*$")


def clean_vendor(text):
    """Drop a trailing company registration number like '(1171881-W)' / '273500-U (GKJ)'.
    SROIE ground truth omits it. Applied identically at train and inference time (no label leakage)."""
    t = text.strip()
    tc = _TRAIL_CODE.search(t)
    if tc and _REG.search(t[:tc.start()]):      # '<reg-number> (BRANCH)' -> look before the branch code
        t = t[:tc.start()]
    m = _REG.search(t)
    if not m:
        return text.strip(" ,;")
    return t[:m.start()].strip(" ,;") or text.strip()


# ------------------------------------------------------------------ boxes
def assign_rows(boxes):
    """Group boxes into visual rows (same line) using vertical centres."""
    if not boxes:
        return boxes
    hs = sorted(b["y1"] - b["y0"] for b in boxes)
    med = hs[len(hs) // 2] or 0.01
    order = sorted(range(len(boxes)), key=lambda i: boxes[i]["y0"] + boxes[i]["y1"])
    row, members = -1, []
    for i in order:
        cy = (boxes[i]["y0"] + boxes[i]["y1"]) / 2
        if not members or abs(cy - sum(members) / len(members)) > 0.6 * med:
            row += 1
            members = [cy]
        else:
            members.append(cy)
        boxes[i]["row"] = row
    return boxes


def make_box(idx, text, xs, ys, W, H):
    c = lambda v: min(max(v, 0.0), 1.0)
    return {"idx": idx, "text": text.strip(),
            "x0": c(min(xs) / W), "y0": c(min(ys) / H), "x1": c(max(xs) / W), "y1": c(max(ys) / H)}


def parse_box_file(path, size):
    W, H = size
    boxes = []
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            parts = line.strip().split(",", 8)
            if len(parts) != 9 or not parts[8].strip():
                continue
            try:
                coords = list(map(int, parts[:8]))
            except ValueError:
                continue
            boxes.append(make_box(len(boxes), parts[8], coords[0::2], coords[1::2], W, H))
    return assign_rows(boxes)


# ------------------------------------------------------------------ candidates
def _union(bs):
    return [min(b["x0"] for b in bs), min(b["y0"] for b in bs), max(b["x1"] for b in bs), max(b["y1"] for b in bs)]


def build_candidates(boxes, field):
    rows = defaultdict(list)
    for b in boxes:
        rows[b["row"]].append(b)
    for r in rows.values():
        r.sort(key=lambda b: b["x0"])

    def ctx(b, start=0):
        same = rows[b["row"]]
        left = " ".join(o["text"] for o in same if o["x0"] < b["x0"] and o is not b)
        above = " ".join(o["text"] for o in rows.get(b["row"] - 1, []))
        return {
            "left": (left + " " + b["text"][:start]).strip(),
            "above": above,
            "row_text": " ".join(o["text"] for o in same),
            "row_n": len(same),
            "row_rank": same.index(b) / max(len(same) - 1, 1),
        }

    def cand(value, bs, start=0, text=None):
        b0 = bs[0]
        text = text if text is not None else b0["text"]
        c = {"field": field, "value": value, "box_ids": [b["idx"] for b in bs], "bbox": _union(bs),
             "text": text, "pos": start / max(len(text), 1), **ctx(b0, start)}
        return c

    out = []
    if field == "vendor":
        ordered = sorted(boxes, key=lambda b: (b["row"], b["x0"]))[:VENDOR_WINDOW]
        for i, b in enumerate(ordered):
            out.append(cand(clean_vendor(b["text"]), [b], text=b["text"]))
            if i + 1 < len(ordered):  # two-line vendor names
                n = ordered[i + 1]
                joined = b["text"] + " " + n["text"]
                out.append(cand(clean_vendor(joined), [b, n], text=joined))
    else:
        rx = DATE_RE if field == "date" else AMOUNT_RE
        for b in sorted(boxes, key=lambda b: (b["row"], b["x0"])):
            for m in rx.finditer(b["text"]):
                v = m.group(0) if field == "date" else m.group(1)
                out.append(cand(v, [b], m.start()))
    return out


def find_targets(cands, gt, field):
    """Indices of ALL candidates that equal the ground truth (value printed twice => both correct)."""
    if not cands or gt is None or not str(gt).strip():
        return []
    if field == "vendor":
        sc = [similarity(c["value"], gt) for c in cands]
        best = max(sc)
        return [i for i, s in enumerate(sc) if s >= 0.8 and s >= best - 1e-9]
    return [i for i, c in enumerate(cands) if values_equal(c["value"], gt, field)]


def make_record(rid, boxes, size, gt=None, split="", image=""):
    rec = {"receipt_id": rid, "split": split, "image": image, "size": list(size),
           "boxes": boxes, "ground_truth": gt, "fields": {}}
    for f in FIELDS:
        cands = build_candidates(boxes, f)
        rec["fields"][f] = {"candidates": cands,
                            "targets": find_targets(cands, gt.get(f) if gt else None, f)}
    return rec


# ------------------------------------------------------------------ dataset IO
def load_gt(entity_path):
    with open(entity_path, encoding="utf-8-sig", errors="replace") as f:
        g = json.load(f)
    return {"vendor": g.get("company", ""), "date": g.get("date", ""), "total": g.get("total", "")}


def prepare_split(dataset_dir, split, limit=None, box_source="gt", ocr_cache=None):
    """box_source='gt' uses the SROIE box files; 'ocr' runs a real OCR engine on the image
    (recommended for the final model: it removes the train/test gap for unseen images)."""
    dataset_dir = Path(dataset_dir)
    imgs = sorted((dataset_dir / split / "img").glob("*.jpg"))[:limit]
    recs = []
    for p in imgs:
        ent = dataset_dir / split / "entities" / f"{p.stem}.txt"
        if not ent.exists():
            continue
        W, H = Image.open(p).size
        if box_source == "ocr":
            from ocr import run_ocr
            boxes = run_ocr(p, cache_dir=ocr_cache)
        else:
            bx = dataset_dir / split / "box" / f"{p.stem}.txt"
            if not bx.exists():
                continue
            boxes = parse_box_file(bx, (W, H))
        recs.append(make_record(p.stem, boxes, (W, H), load_gt(ent), split, p.relative_to(dataset_dir).as_posix()))
    return recs
