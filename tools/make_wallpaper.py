#!/usr/bin/env python3
"""Original Leonida sunset wallpaper in the theme palette."""
import math
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H = 3840, 2160
SS = 2  # supersampling for silhouettes
HORIZON = int(H * 0.64)
random.seed(6)
rng = np.random.default_rng(6)


def hex2rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=float)


def gradient(stops, n):
    """stops: list of (pos 0..1, hex). Returns n x 3 array."""
    pos = np.array([s[0] for s in stops])
    cols = np.array([hex2rgb(s[1]) for s in stops])
    t = np.linspace(0, 1, n)
    return np.stack([np.interp(t, pos, cols[:, c]) for c in range(3)], axis=1)


# Sky
sky = gradient([
    (0.00, "#0b0714"), (0.22, "#1d1236"), (0.45, "#4a1f66"),
    (0.66, "#a8327a"), (0.82, "#ff4fa3"), (0.92, "#ff8a4c"), (1.00, "#ffcc33"),
], HORIZON)
img = np.zeros((H, W, 3))
img[:HORIZON] = sky[:, None, :]

# Stars in the upper sky
for _ in range(420):
    x, y = rng.integers(0, W), int(rng.beta(1.2, 4) * HORIZON * 0.7)
    b = rng.uniform(0.25, 0.9) * (1 - y / (HORIZON * 0.7))
    r = 1 if rng.random() < 0.85 else 2
    img[y:y + r, x:x + r] = img[y:y + r, x:x + r] * (1 - b) + 255 * b

# Sun: vertical gold -> pink with retro stripes on its lower half
cx, cy, R = W // 2, HORIZON - int(H * 0.11), int(H * 0.2)
yy, xx = np.mgrid[0:HORIZON, 0:W]
d = np.hypot(xx - cx, yy - cy)
sun_cols = gradient([(0, "#ffe27a"), (0.5, "#ffb04c"), (1, "#ff4fa3")], 2 * R + 1)
rel = np.clip(yy - (cy - R), 0, 2 * R)
in_sun = d <= R
# stripes: gaps grow toward the bottom
stripe = np.ones_like(in_sun)
for i in range(7):
    y0 = cy + int(R * (0.08 + i * 0.14))
    thick = int(R * (0.012 + i * 0.012))
    stripe[y0:y0 + thick] = False
in_sun &= stripe
img[:HORIZON][in_sun] = sun_cols[rel[in_sun]]
# Glow around the sun
glow = np.exp(-np.clip(d - R, 0, None) / (R * 0.9)) * 0.55
glow[d <= R] = 0
img[:HORIZON] = img[:HORIZON] * (1 - glow[..., None]) + hex2rgb("#ff7aa8") * glow[..., None]

# Sea
sea = gradient([(0, "#4a1f66"), (0.25, "#2a1645"), (1, "#0b0714")], H - HORIZON)
img[HORIZON:] = sea[:, None, :]
# Sun reflection streaks
for i in range(160):
    t = (i / 160) ** 1.6
    y = HORIZON + int(t * (H - HORIZON) * 0.85) + rng.integers(0, 6)
    half = int(R * (1.05 - 0.55 * t) * rng.uniform(0.35, 1.0))
    off = rng.integers(-int(R * 0.15), int(R * 0.15))
    thick = 2 + int(t * 7)
    col = hex2rgb(["#ffcc33", "#ff8a4c", "#ff4fa3", "#e07be0"][min(3, int(t * 4))])
    a = 0.75 * (1 - t) + 0.1
    x0, x1 = max(0, cx + off - half), min(W, cx + off + half)
    img[y:y + thick, x0:x1] = img[y:y + thick, x0:x1] * (1 - a) + col * a
# Aqua shimmer across the water
for _ in range(500):
    t = rng.random() ** 1.3
    y = HORIZON + int(t * (H - HORIZON))
    x = rng.integers(0, W)
    ln = int(rng.uniform(20, 140) * (0.4 + t))
    a = rng.uniform(0.08, 0.3)
    img[y:y + 2, x:x + ln] = img[y:y + 2, x:x + ln] * (1 - a) + hex2rgb("#4fe0e0") * a

base = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

# Silhouettes drawn supersampled
sil = Image.new("L", (W * SS, H * SS), 0)
sd = ImageDraw.Draw(sil)
skyline = Image.new("L", (W * SS, H * SS), 0)
kd = ImageDraw.Draw(skyline)

# Distant art-deco skyline on the horizon, away from the sun
hz = HORIZON * SS
x = 0
while x < W * SS:
    bw = random.randint(60, 220) * SS
    if abs((x + bw / 2) - cx * SS) < R * SS * 1.35:
        x += bw
        continue
    bh = random.randint(30, 230) * SS
    kd.rectangle([x, hz - bh, x + bw, hz], fill=255)
    if random.random() < 0.35:  # stepped deco top
        kd.rectangle([x + bw * 0.25, hz - bh - bh * 0.18, x + bw * 0.75, hz], fill=255)
    if random.random() < 0.2:
        kd.rectangle([x + bw * 0.47, hz - bh - bh * 0.45, x + bw * 0.53, hz], fill=255)
    x += bw + random.randint(0, 30) * SS


def bezier(p0, p1, p2, n=80):
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1])
            for t in np.linspace(0, 1, n)]


def tapered(draw, pts, w0, w1):
    left, right = [], []
    for i, (px, py) in enumerate(pts):
        j = min(i + 1, len(pts) - 1)
        k = max(i - 1, 0)
        dx, dy = pts[j][0] - pts[k][0], pts[j][1] - pts[k][1]
        n = math.hypot(dx, dy) or 1
        nx, ny = -dy / n, dx / n
        w = w0 + (w1 - w0) * i / (len(pts) - 1)
        left.append((px + nx * w, py + ny * w))
        right.append((px - nx * w, py - ny * w))
    draw.polygon(left + right[::-1], fill=255)


def palm(draw, bx, by, height, lean, s):
    top = (bx + lean, by - height)
    trunk = bezier((bx, by), (bx + lean * 0.15, by - height * 0.55), top)
    tapered(draw, trunk, 26 * s, 13 * s)
    # trunk rings
    for (px, py) in trunk[4::5]:
        draw.ellipse([px - 17 * s, py - 4 * s, px + 17 * s, py + 4 * s], fill=255)
    for ang in [-170, -150, -128, -100, -70, -45, -20, -5, 15, -85, -160, -35]:
        a = math.radians(ang + random.uniform(-6, 6))
        ln = height * random.uniform(0.36, 0.5)
        tip = (top[0] + math.cos(a) * ln * 1.15, top[1] + math.sin(a) * ln * 0.35 + ln * 0.28)
        ctrl = (top[0] + math.cos(a) * ln * 0.6, top[1] + math.sin(a) * ln * 0.8 - ln * 0.22)
        frond = bezier(top, ctrl, tip, 60)
        tapered(draw, frond, 9 * s, 1 * s)
        # leaflets hanging off the frond
        for i in range(4, 58, 3):
            px, py = frond[i]
            qx, qy = frond[i + 1]
            dx, dy = qx - px, qy - py
            n = math.hypot(dx, dy) or 1
            ll = (1 - i / 60) * 62 * s + 14 * s
            for side in (1, -1):
                nx, ny = -dy / n * side, dx / n * side
                ex = px + (nx * 0.6 + dx / n * 0.5) * ll
                ey = py + (abs(ny) * 0.9 + 0.5) * ll
                draw.line([(px, py), (ex, ey)], fill=255, width=int(4 * s))
    draw.ellipse([top[0] - 22 * s, top[1] - 16 * s, top[0] + 22 * s, top[1] + 16 * s], fill=255)


s = SS
palm(sd, 300 * s, H * s + 40 * s, 1700 * s, 230 * s, s * 1.6)
palm(sd, 820 * s, H * s + 40 * s, 1150 * s, -120 * s, s * 1.2)
palm(sd, 3560 * s, H * s + 40 * s, 1550 * s, -260 * s, s * 1.5)
palm(sd, 3140 * s, H * s + 40 * s, 900 * s, 140 * s, s * 1.0)

skyline = skyline.resize((W, H), Image.LANCZOS)
sil = sil.resize((W, H), Image.LANCZOS)

out = Image.composite(Image.new("RGB", (W, H), "#2a1645"), base,
                      skyline.point(lambda v: int(v * 0.85)))
out = Image.composite(Image.new("RGB", (W, H), "#0b0714"), out, sil)

# Grain and vignette
arr = np.asarray(out).astype(float)
arr += rng.normal(0, 3.2, arr.shape)
vy, vx = np.mgrid[0:H, 0:W]
v = 1 - 0.35 * (((vx - W / 2) / (W / 2)) ** 2 + ((vy - H / 2) / (H / 2)) ** 2) / 2
arr *= v[..., None]
Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(sys.argv[1], quality=92)
