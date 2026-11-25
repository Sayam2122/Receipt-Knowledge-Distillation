"""Step 2: frozen Qwen2.5-VL-3B-Instruct teacher (HuggingFace transformers; run on a GPU, e.g. Colab/Kaggle T4).

Two outputs, both from the SAME frozen teacher:
 --mode soft : for every field, a TRUE probability distribution over the student's candidates, read from the
               teacher's next-token logits for "ID=<k>" (no sampling noise; temperature-controlled).
               -> kd_data/teacher_soft_<split>.json  {receipt: {field: {probs:[..], entropy, argmax}}}
 --mode json : free-form teacher extraction WITH grounding boxes (value + bbox_2d + line items)
               -> kd_data/teacher_json_<split>.json  {receipt: {field: {value, bbox(norm)}, line_items}}
               used for the side-by-side 'teacher vs student vs ground truth' figure and for teacher accuracy.
python teacher_qwen.py --mode soft --split train --data kd_data --dataset receipt_dataset [--limit 200]
"""
import argparse, json, math, re
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MAX_CAND = 100
INSTR = {
    "vendor": "Identify the business that issued this receipt (the merchant), not a product brand or an address line.",
    "date": "Identify the transaction date of the receipt.",
    "total": "Identify the final amount payable (grand total). Not the subtotal, cash tendered, change, tax or an item price.",
}


def combine_digit_probs(p1, p2_by_d1, n, width):
    """P(id) for ids 0..n-1 from digit-by-digit probabilities. p1: list[10] first digit;
    p2_by_d1: {d1: list[10]} second-digit probs for the first digits we actually evaluated.
    First digits we skipped (negligible mass) are spread uniformly over their valid ids."""
    if width == 1:
        pr = [p1[i] for i in range(n)]
    else:
        pr = []
        for i in range(n):
            d1, d2 = divmod(i, 10)
            if d1 in p2_by_d1:
                pr.append(p1[d1] * p2_by_d1[d1][d2])
            else:
                k = min(10, n - d1 * 10)
                pr.append(p1[d1] / k)
    s = sum(pr)
    return [x / s for x in pr] if s > 0 else [1.0 / n] * n


def entropy(p):
    return -sum(x * math.log(x) for x in p if x > 0) / (math.log(len(p)) if len(p) > 1 else 1.0)


def build_prompt(field, cands, width):
    lines = [f"{i:0{width}d}: {c['value']}  | line: {c['row_text'][:80]}" for i, c in enumerate(cands)]
    return (f"You are reading a receipt image. {INSTR[field]}\nCandidates found by OCR:\n" + "\n".join(lines) +
            f"\nAnswer with the candidate number only, in the form ID=<number> ({width} digits).")


def load_model(device_map="auto"):
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map=device_map).eval()
    proc = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
    return model, proc


def _inputs(proc, model, image, prompt, suffix=""):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + suffix
    return proc(text=[text], images=[image], return_tensors="pt").to(model.device)


def score_candidates(model, proc, image, field, cands, T=1.0, skip_below=0.02):
    import torch
    n = min(len(cands), MAX_CAND)
    if n <= 1:
        return [1.0] * n
    width = len(str(n - 1))
    prompt = build_prompt(field, cands[:n], width)
    tok = proc.tokenizer
    dig = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]

    @torch.no_grad()
    def digit_probs(suffix):
        inp = _inputs(proc, model, image, prompt, suffix)
        logits = model(**inp).logits[0, -1].float()[dig] / T
        return torch.softmax(logits, -1).tolist()

    p1 = digit_probs("ID=")
    p2 = {}
    if width == 2:
        for d1 in range(10):
            if d1 * 10 < n and p1[d1] >= skip_below:
                p2[d1] = digit_probs(f"ID={d1}")
    return combine_digit_probs(p1, p2, n, width)


def teacher_json(model, proc, image):
    prompt = ('Locate on this receipt: the actual merchant (business) name (not the owner, cashier or a person\'s name, and not an address), the transaction date, the final total amount paid, and the purchased items. '
              'Return ONLY JSON: {"vendor":{"text":..,"bbox_2d":[x1,y1,x2,y2]},"date":{"text":..,"bbox_2d":[..]},'
              '"total":{"text":..,"bbox_2d":[..]},"line_items":[{"description":..,"quantity":..,"amount":..}]}')
    inp = _inputs(proc, model, image, prompt)
    out = model.generate(**inp, max_new_tokens=600, do_sample=False)
    txt = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    m = re.search(r"\{.*\}", txt.replace("```json", "").replace("```", ""), re.S)
    try:
        d = json.loads(m.group(0))
    except Exception:
        return {"raw": txt}
    _, gh, gw = inp["image_grid_thw"][0].tolist()
    H, W = gh * 14, gw * 14                       # coordinates are in the resized-image pixel space
    res = {"raw": txt, "line_items": d.get("line_items", [])}
    for f in ("vendor", "date", "total"):
        x = d.get(f) or {}
        bb = x.get("bbox_2d")
        res[f] = {"value": x.get("text") if isinstance(x, dict) else x,
                  "bbox": [bb[0] / W, bb[1] / H, bb[2] / W, bb[3] / H] if bb and len(bb) == 4 else None}
    return res


if __name__ == "__main__":
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["soft", "json"], required=True)
    ap.add_argument("--split", default="train"); ap.add_argument("--data", default="kd_data")
    ap.add_argument("--dataset", default="receipt_dataset"); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--fields", nargs="+", default=["vendor", "date", "total"], help="fields to score in --mode soft")
    a = ap.parse_args()
    recs = json.load(open(Path(a.data) / f"{a.split}.json"))[: a.limit]
    outp = Path(a.data) / f"teacher_{a.mode if a.mode == 'json' else 'soft'}_{a.split}.json"
    res = json.load(open(outp)) if outp.exists() else {}     # resumable
    model, proc = load_model()
    for n, r in enumerate(recs):
        if r["receipt_id"] in res:
            continue
        img = Image.open(Path(a.dataset) / r["image"].replace("\\", "/")).convert("RGB")
        if a.mode == "soft":
            res[r["receipt_id"]] = {}
            for f in a.fields:
                c = r["fields"][f]["candidates"]
                if c:
                    p = score_candidates(model, proc, img, f, c, a.T)
                    res[r["receipt_id"]][f] = {"probs": p, "entropy": entropy(p), "argmax": max(range(len(p)), key=p.__getitem__)}
        else:
            res[r["receipt_id"]] = teacher_json(model, proc, img)
        if n % 10 == 0:
            json.dump(res, open(outp, "w")); print(n, "/", len(recs))
    json.dump(res, open(outp, "w")); print("saved", outp)
