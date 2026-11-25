from PIL import Image, ImageDraw

COL = {"gt": (0, 170, 0), "teacher": (30, 90, 255), "student": (230, 30, 30)}


def draw(image_path, boxes_by_source, size=None):
    """boxes_by_source: {'gt'|'teacher'|'student': [(label, [x0,y0,x1,y1] normalised), ...]}"""
    im = Image.open(image_path).convert("RGB"); W, H = im.size; d = ImageDraw.Draw(im)
    lw = max(2, W // 300)
    for k, items in boxes_by_source.items():
        off = {"gt": 0, "teacher": lw, "student": 2 * lw}[k]
        for label, b in items:
            if b is None:
                continue
            r = [b[0] * W - off, b[1] * H - off, b[2] * W + off, b[3] * H + off]
            d.rectangle(r, outline=COL[k], width=lw)
            d.text((r[0], max(r[1] - 12, 0)), f"{k[0].upper()}:{label}", fill=COL[k])
    return im
