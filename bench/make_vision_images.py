"""Make the fixed test images for the vision oracle (bench/llama_vision.cpp).

Run with the project venv:  .venv\\Scripts\\python.exe bench\\make_vision_images.py
Writes PNG files to bench/out/vision/ (git-ignored). The images are deterministic.

  sq448.png         448x448   synthetic pattern (gradients, shapes, stripes)
  photo640x480.png  640x480   non-square synthetic scene (sky, sun, hills, house) with mild noise
  chart800x600.png  800x600   bar chart with title, axis labels, values and a legend
  doc1300x850.png   1300x850  text document above 1 Mpx (not a multiple of 32: exercises the rounding path)
  tall360x640.png   360x640   portrait phone-style list screen (height > width)
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "vision")


def font(size, bold=False):
    names = ["arialbd.ttf", "segoeuib.ttf"] if bold else ["arial.ttf", "segoeui.ttf"]
    for n in names:
        try:
            return ImageFont.truetype(os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", n), size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def sq448():
    w = h = 448
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r = 255 * x / (w - 1)
    g = 255 * y / (h - 1)
    b = 127.5 + 127.5 * np.sin(x / 23.0) * np.cos(y / 31.0)
    img = np.stack([r, g, b], -1)
    img[300:340, :, :] = np.where(((x[300:340] // 4) % 2 == 0)[..., None], 240, 20)  # fine stripes
    im = Image.fromarray(img.clip(0, 255).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(im)
    d.ellipse([60, 60, 200, 200], fill=(250, 220, 40), outline=(0, 0, 0), width=4)
    d.rectangle([250, 80, 400, 230], fill=(30, 90, 200), outline=(255, 255, 255), width=3)
    d.polygon([(100, 420), (220, 250), (340, 420)], fill=(200, 40, 60))
    d.text((240, 360), "Q27", font=font(48, True), fill=(255, 255, 255))
    return im


def photo640x480():
    w, h = 640, 480
    rng = np.random.default_rng(1234)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    t = y / (h - 1)
    sky = np.stack([90 + 120 * t, 150 + 80 * t, 235 - 20 * t], -1)
    img = sky.copy()
    hill1 = 300 + 40 * np.sin(x / 70.0)
    hill2 = 360 + 30 * np.sin(x / 45.0 + 1.3)
    img[y > hill1] = [70, 140, 60]
    img[y > hill2] = [50, 110, 45]
    img += rng.normal(0, 4, img.shape)
    im = Image.fromarray(img.clip(0, 255).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(im)
    d.ellipse([480, 50, 560, 130], fill=(255, 230, 120))
    d.rectangle([180, 300, 330, 400], fill=(200, 170, 130), outline=(60, 40, 30), width=2)
    d.polygon([(165, 300), (255, 230), (345, 300)], fill=(150, 50, 40))
    d.rectangle([235, 345, 270, 400], fill=(90, 60, 40))
    d.rectangle([195, 320, 225, 345], fill=(170, 210, 240), outline=(60, 40, 30))
    d.rectangle([285, 320, 315, 345], fill=(170, 210, 240), outline=(60, 40, 30))
    return im


def chart800x600():
    w, h = 800, 600
    im = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.text((w // 2, 30), "Quarterly sales 2026 (thousand units)", font=font(26, True), fill=(20, 20, 20), anchor="mm")
    x0, y0, x1, y1 = 90, 520, 760, 90  # plot area: left, bottom, right, top
    d.line([x0, y0, x1, y0], fill=(0, 0, 0), width=2)
    d.line([x0, y0, x0, y1], fill=(0, 0, 0), width=2)
    vmax = 100
    for v in range(0, vmax + 1, 20):
        yy = y0 - (y0 - y1) * v / vmax
        d.line([x0, yy, x1, yy], fill=(220, 220, 220), width=1)
        d.text((x0 - 10, yy), str(v), font=font(16), fill=(60, 60, 60), anchor="rm")
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    series = {"North": ([42, 55, 61, 78], (52, 101, 164)), "South": ([30, 47, 39, 66], (230, 126, 34))}
    gw = (x1 - x0) / len(quarters)
    bw = gw * 0.3
    for qi, q in enumerate(quarters):
        cx = x0 + gw * (qi + 0.5)
        for si, (name, (vals, col)) in enumerate(series.items()):
            bx = cx - bw + si * bw
            by = y0 - (y0 - y1) * vals[qi] / vmax
            d.rectangle([bx + 2, by, bx + bw - 2, y0 - 1], fill=col)
            d.text((bx + bw / 2, by - 12), str(vals[qi]), font=font(15), fill=(30, 30, 30), anchor="mm")
        d.text((cx, y0 + 20), q, font=font(18), fill=(30, 30, 30), anchor="mm")
    d.text((w // 2, 575), "Quarter", font=font(18, True), fill=(30, 30, 30), anchor="mm")
    lx, ly = 600, 100
    d.rectangle([lx - 10, ly - 10, lx + 150, ly + 60], outline=(150, 150, 150), fill=(250, 250, 250))
    for si, (name, (vals, col)) in enumerate(series.items()):
        d.rectangle([lx, ly + si * 26, lx + 18, ly + si * 26 + 18], fill=col)
        d.text((lx + 28, ly + si * 26 + 9), name, font=font(17), fill=(30, 30, 30), anchor="lm")
    return im


def doc1300x850():
    w, h = 1300, 850
    im = Image.new("RGB", (w, h), (252, 252, 248))
    d = ImageDraw.Draw(im)
    d.text((60, 40), "Maintenance report: pump station 7", font=font(40, True), fill=(10, 10, 10))
    d.line([60, 100, w - 60, 100], fill=(120, 120, 120), width=2)
    para = [
        "On 3 October 2026 the team replaced the main bearing of pump B. The old bearing showed",
        "pitting on the outer race and the vibration level had risen to 7.1 mm/s over two weeks.",
        "After the repair the vibration dropped to 1.8 mm/s. The pump ran for six hours without alarms.",
        "",
        "Open items: order a spare seal kit (part 44-1093), check the flow meter calibration, and",
        "schedule the next inspection for January 2027.",
    ]
    y = 130
    for line in para:
        d.text((60, y), line, font=font(26), fill=(25, 25, 25))
        y += 38
    # small table
    cols = [60, 360, 620, 900]
    rows = [("Item", "Before", "After", "Unit"), ("Vibration", "7.1", "1.8", "mm/s"),
            ("Bearing temp.", "71", "48", "deg C"), ("Flow", "212", "238", "m3/h")]
    ty = 400
    for ri, row in enumerate(rows):
        f = font(24, ri == 0)
        if ri == 0:
            d.rectangle([50, ty - 6, 1100, ty + 34], fill=(225, 232, 240))
        for ci, cell in enumerate(row):
            d.text((cols[ci], ty), cell, font=f, fill=(20, 20, 20))
        ty += 48
        d.line([50, ty - 8, 1100, ty - 8], fill=(190, 190, 190), width=1)
    d.text((60, h - 60), "Signed: J. Ferrer, shift lead", font=font(22), fill=(70, 70, 70))
    return im


def tall360x640():
    w, h = 360, 640
    im = Image.new("RGB", (w, h), (245, 245, 247))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w, 28], fill=(30, 30, 30))
    d.text((12, 14), "09:41", font=font(15, True), fill=(255, 255, 255), anchor="lm")
    d.text((w - 12, 14), "87%", font=font(15), fill=(255, 255, 255), anchor="rm")
    d.rectangle([0, 28, w, 84], fill=(0, 120, 215))
    d.text((16, 56), "Shopping list", font=font(24, True), fill=(255, 255, 255), anchor="lm")
    items = [("Milk", "2 L"), ("Eggs", "12"), ("Tomatoes", "1 kg"), ("Bread", "1 loaf"), ("Coffee", "500 g"),
             ("Olive oil", "1 L"), ("Apples", "6"), ("Rice", "2 kg"), ("Cheese", "250 g")]
    y = 96
    for i, (name, qty) in enumerate(items):
        d.rectangle([10, y, w - 10, y + 52], fill=(255, 255, 255), outline=(220, 220, 225))
        done = i in (1, 4)
        d.rectangle([22, y + 16, 42, y + 36], outline=(0, 120, 215), width=2, fill=(0, 120, 215) if done else None)
        d.text((56, y + 26), name, font=font(19), fill=(150, 150, 150) if done else (20, 20, 20), anchor="lm")
        d.text((w - 24, y + 26), qty, font=font(17), fill=(90, 90, 90), anchor="rm")
        y += 58
    return im


def main():
    os.makedirs(OUT, exist_ok=True)
    for fn in (sq448, photo640x480, chart800x600, doc1300x850, tall360x640):
        im = fn()
        p = os.path.join(OUT, fn.__name__ + ".png")
        im.save(p)
        print(p, im.size)


if __name__ == "__main__":
    main()
