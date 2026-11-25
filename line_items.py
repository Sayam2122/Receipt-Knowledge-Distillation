"""Heuristic line-item extractor (NOT learned, NOT evaluated: SROIE has no line-item labels).
Item region = rows after the column-header row (Description/Qty/Price/Amount) up to the first total/GST row.
An item = a row ending in an amount; its description is the alphabetic text on that row, or on the wrapped
row(s) right below it. Quantity is only reported when a standalone small integer box exists."""
import re
from collections import defaultdict
from sroie_data import AMOUNT_RE
from prettify import prettify

STOP = re.compile(r"\b(sub\s*-?\s*total|total|cash|change|gst|tax|rounding|tender|visa|master|discount|balance|qty\s*:)", re.I)
HEADER = re.compile(r"(description|qty|price|amount|item)", re.I)
SKIP = re.compile(r"\b(tel|fax|invoice|receipt|cashier|table|bill|email|address|roc|reg)\b", re.I)


def _alpha(s):
    return len(re.findall(r"[A-Za-z]", s))


def _is_text(t):
    return _alpha(t) >= 3 and not AMOUNT_RE.fullmatch(t.strip()) and not HEADER.fullmatch(t.strip(" .:"))


def extract_line_items(boxes, vendor_bbox=None):
    rows = defaultdict(list)
    for b in boxes:
        rows[b["row"]].append(b)
    order = sorted(rows)
    texts = {r: " ".join(b["text"] for b in sorted(rows[r], key=lambda b: b["x0"])) for r in order}
    # start after the last header-like row that comes before the first amount-bearing, non-stop row
    start = 0
    for k, r in enumerate(order):
        if len(HEADER.findall(texts[r])) >= 2:
            start = k + 1
            break
    items, k = [], start
    while k < len(order):
        r = order[k]
        bs = sorted(rows[r], key=lambda b: b["x0"]); text = texts[r]
        if STOP.search(text):
            if items:
                break
            k += 1; continue
        if SKIP.search(text) or (vendor_bbox and bs[0]["y1"] <= vendor_bbox[3]):
            k += 1; continue
        last = bs[-1]
        m = list(AMOUNT_RE.finditer(last["text"]))
        if not m:
            k += 1; continue
        amt = m[-1].group(1)
        desc_parts = [b["text"] for b in bs if _is_text(b["text"])]
        j = k + 1
        if not desc_parts:                       # description wrapped onto the next row(s)
            while j < len(order) and not STOP.search(texts[order[j]]):
                nxt = sorted(rows[order[j]], key=lambda b: b["x0"])
                if AMOUNT_RE.search(nxt[-1]["text"]):
                    break
                desc_parts += [b["text"] for b in nxt if _is_text(b["text"])]
                j += 1
        qty = next((int(b["text"]) for b in bs if re.fullmatch(r"\d{1,2}", b["text"].strip())), None)
        if desc_parts:
            items.append({"description": prettify(" ".join(desc_parts)), "quantity": qty, "amount": amt,
                          "bbox": [min(b["x0"] for b in bs), min(b["y0"] for b in bs),
                                   max(b["x1"] for b in bs), max(b["y1"] for b in bs)]})
        k = max(j, k + 1)
    return items
