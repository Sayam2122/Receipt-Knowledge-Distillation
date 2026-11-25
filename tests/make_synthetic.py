"""Generate a SMALL fake SROIE-format dataset (for smoke-testing the pipeline plumbing ONLY - not for accuracy claims)."""
import json, random, sys
from pathlib import Path
from PIL import Image, ImageDraw

VENDORS = ["ABC TRADING SDN BHD", "KEDAI RUNCIT AMIR", "SUPER MART ENTERPRISE", "GOLDEN CAFE", "HOME HARDWARE SDN. BHD."]
ITEMS = ["MILK", "BREAD", "RICE 5KG", "EGGS", "SOAP", "COFFEE", "NOODLES", "SUGAR"]


def make(root, split, n, seed):
    rnd = random.Random(seed)
    for d in ("img", "box", "entities"):
        (Path(root) / split / d).mkdir(parents=True, exist_ok=True)
    for k in range(n):
        rid = f"S{seed}{k:04d}"
        W, H = 400, 800
        im = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(im)
        lines, y = [], 20
        def add(text, x=20, w=None):
            nonlocal y
            w = w or 9 * len(text)
            dr.text((x, y), text, fill="black")
            lines.append((x, y, x + w, y + 16, text)); return y
        vendor = rnd.choice(VENDORS); add(vendor); y += 24
        add("TEL: 03-1234567"); y += 24
        date = f"{rnd.randint(1,28):02d}/{rnd.randint(1,12):02d}/2018"; add(f"DATE: {date}"); y += 30
        tot = 0.0
        for _ in range(rnd.randint(2, 6)):
            p = round(rnd.uniform(1, 30), 2); tot += p
            add(rnd.choice(ITEMS), 20, 120); add(f"{p:.2f}", 300, 50); y += 22
        tot = round(tot, 2); y += 10
        if rnd.random() < .5:
            add("SUBTOTAL", 20, 80); add(f"{tot:.2f}", 300, 50); y += 22
        if rnd.random() < .5:
            add(f"TOTAL {tot:.2f}", 20, 200)       # label+amount in one box
        else:
            add("TOTAL", 20, 60); add(f"{tot:.2f}", 300, 50)
        y += 22
        cash = round(tot + rnd.choice([0, 0.5, 3, 10, 20]), 2)
        add("CASH", 20, 60); add(f"{cash:.2f}", 300, 50); y += 22
        add("CHANGE", 20, 70); add(f"{cash - tot:.2f}", 300, 50)
        im.save(Path(root) / split / "img" / f"{rid}.jpg")
        with open(Path(root) / split / "box" / f"{rid}.txt", "w") as f:
            for x0, y0, x1, y1, t in lines:
                f.write(f"{x0},{y0},{x1},{y0},{x1},{y1},{x0},{y1},{t}\n")
        json.dump({"company": vendor, "date": date, "address": "x", "total": f"{tot:.2f}"},
                  open(Path(root) / split / "entities" / f"{rid}.txt", "w"))


if __name__ == "__main__":
    root = sys.argv[1]
    make(root, "train", 80, 1); make(root, "eval", 20, 2); make(root, "test", 30, 3)
