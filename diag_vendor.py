import json
from sroie_data import values_equal, similarity
from extract import ReceiptExtractor
from features import featurize
from metrics import predict_record

m = ReceiptExtractor("runs/ce_all/seed0.pt").model
n = 0
for r in json.load(open("kd_data/test.json", encoding="utf-8")):
    p = predict_record(m, r, featurize)["vendor"]
    gt = r["ground_truth"]["vendor"]
    if p and not values_equal(p["value"], gt, "vendor") and similarity(p["value"], gt) >= 0.8:
        print(repr(p["value"]), "|", repr(gt)); n += 1
        if n >= 15: break