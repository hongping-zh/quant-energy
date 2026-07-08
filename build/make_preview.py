#!/usr/bin/env python3
"""Generate preview.png (1200x630 Open Graph card) for quantenergy.tech."""
from PIL import Image, ImageDraw, ImageFont
import os

W, H = 1200, 630
BG = (11, 16, 32)
BG2 = (22, 34, 74)
TXT = (232, 237, 247)
MUTED = (148, 163, 184)
ACCENT = (56, 189, 248)
NF4 = (16, 185, 129)
BAD = (244, 63, 94)

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)

# subtle top radial-ish band
for y in range(0, 240):
    t = 1 - y / 240
    c = tuple(int(BG[i] + (BG2[i] - BG[i]) * t) for i in range(3))
    d.line([(0, y), (W, y)], fill=c)


def font(sz, bold=False):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


PAD = 70
# brand (name left, url right-aligned)
d.text((PAD, 60), "EcoCompute", font=font(46, True), fill=TXT)
url = "quantenergy.tech"
uw = d.textlength(url, font=font(28))
d.text((W - PAD - uw, 74), url, font=font(28), fill=ACCENT)

# headline (two lines)
d.text((PAD, 168), "Quantization Doesn't", font=font(72, True), fill=TXT)
d.text((PAD, 250), "Always Save Energy", font=font(72, True), fill=NF4)

# subhead
d.text((PAD, 360),
       "Weight-only NF4/INT8 can raise energy on small LLMs",
       font=font(30), fill=MUTED)
d.text((PAD, 402),
       "and only saves on larger ones. See it for your model & GPU.",
       font=font(30), fill=MUTED)

# stat chips
chips = [
    ("360+", "GPU measurements", ACCENT),
    ("4", "architectures", NF4),
    ("NVML", "direct power sampling", MUTED),
]
x = PAD
cy = 490
for big, small, col in chips:
    bw = d.textlength(big, font=font(40, True))
    d.text((x, cy), big, font=font(40, True), fill=col)
    d.text((x, cy + 50), small, font=font(22), fill=MUTED)
    sw = d.textlength(small, font=font(22))
    x += max(bw, sw) + 70

# footer accent bar
d.rectangle([(0, H - 10), (W, H)], fill=NF4)

img.save("/home/ubuntu/ecocompute-demo/preview.png", "PNG")
print("wrote preview.png", img.size)
