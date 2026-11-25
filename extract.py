"""Stand-alone inference: no teacher, no dataset. image -> OCR -> student -> structured JSON."""
import torch
from sroie_data import FIELDS, make_record
from features import featurize, FEATURE_DIM
from student import CandidateStudent
from metrics import predict_record
from line_items import extract_line_items
from prettify import prettify


class ReceiptExtractor:
    def __init__(self, ckpt, device="cpu"):
        c = torch.load(ckpt, map_location="cpu", weights_only=True)
        self.model = CandidateStudent(c["in_dim"]); self.model.load_state_dict(c["state"])
        self.model.to(device).eval(); self.device = device

    def from_boxes(self, boxes, size, rid="query"):
        rec = make_record(rid, boxes, size)
        pred = predict_record(self.model, rec, featurize, self.device)
        out = {f: ({"value": pred[f]["value"], "confidence": round(pred[f]["prob"], 4), "bbox": pred[f]["bbox"]}
                   if pred[f] else {"value": "", "confidence": 0.0, "bbox": None}) for f in FIELDS}
        out["line_items"] = extract_line_items(boxes, out["vendor"]["bbox"])
        return out

    def from_image(self, path):
        from PIL import Image
        from ocr import run_ocr
        boxes = run_ocr(path)
        return self.from_boxes(boxes, Image.open(path).size, str(path)), boxes

    @staticmethod
    def simple(out):
        return {"vendor": prettify(out["vendor"]["value"]), "date": out["date"]["value"], "total": out["total"]["value"],
                "line_items": [{k: i[k] for k in ("description", "quantity", "amount")} for i in out["line_items"]]}
