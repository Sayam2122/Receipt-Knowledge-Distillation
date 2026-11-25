"""Demo interface (deliverable #2). python app.py --ckpt runs/kd/seed0.pt   -> http://127.0.0.1:7860"""
import argparse, json, time
import gradio as gr
from extract import ReceiptExtractor
from visualize import draw

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--share", action="store_true")
a, _ = ap.parse_known_args()
ex = ReceiptExtractor(a.ckpt)


def run(image_path):
    if image_path is None:
        return None, {}, ""
    t = time.time(); res, _ = ex.from_image(image_path); ms = (time.time() - t) * 1000
    vis = draw(image_path, {"student": [(f, res[f]["bbox"]) for f in ("vendor", "date", "total")]})
    return vis, {**ReceiptExtractor.simple(res), "confidence": {f: res[f]["confidence"] for f in ("vendor", "date", "total")}}, f"{ms:.0f} ms (OCR + student)"


demo = gr.Interface(run, gr.Image(type="filepath", label="Receipt / invoice image"),
                    [gr.Image(label="Student's localised fields"), gr.JSON(label="Structured output"), gr.Textbox(label="Latency")],
                    title="Receipt extraction - distilled compact student", flagging_mode="never")
if __name__ == "__main__":
    demo.launch(share=a.share)
