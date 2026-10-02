"""Generates the filter icon (docs/icon.svg, docs/icon.png).

A sphere blurred in world space: its checker is crisp inside one cap (where the
blur map is black) and blurred everywhere else, more and more away from the cap.
The PNG (128x128) is embedded in the .sbs by build_wsnub.py; the SVG keeps the
outline as a vector and embeds the blurred interior as a 512px image (a spatially
varying blur can't be expressed with plain SVG shapes).
"""
import base64
import io
import math
import os

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(os.path.dirname(HERE), 'docs')

SIZE = 128
CX, CY, R = 64.0, 64.0, 57.0            # sphere silhouette
RIM = 2.6                                # outline width
CAP_X, CAP_Y, CAP_R = 95.0, 39.0, 33.0   # the sharp (unblurred) area
CELL = 12.0                              # checker cell
ROT = math.radians(-18)                  # checker rotation, to suggest curvature
BLUR_MIN, BLUR_MAX = 1.6, 7.0            # blur sigma (icon px) just outside the cap / far away
BLUR_RAMP = 60.0                         # distance (icon px) over which the blur grows


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def interior(res):
    """alpha of the checker interior at res x res (no outline)."""
    s = res / SIZE
    c = (np.arange(res) + 0.5) / s
    X, Y = np.meshgrid(c, c)
    dx, dy = X - CX, Y - CY
    u = (dx * math.cos(ROT) - dy * math.sin(ROT)) / CELL
    v = (dx * math.sin(ROT) + dy * math.cos(ROT)) / CELL
    chk = ((np.floor(u) + np.floor(v)) % 2 == 0).astype(np.float64)
    # spatially varying blur: interpolate between a stack of blurred copies
    dcap = np.hypot(X - CAP_X, Y - CAP_Y) - CAP_R
    sigma = np.where(dcap <= 0, 0.0, BLUR_MIN + (BLUR_MAX - BLUR_MIN) * smoothstep(dcap / BLUR_RAMP))
    levels = [0.0, 1.0, 1.6, 2.5, 3.5, 5.0, 7.0]
    stack = [chk if l == 0 else gaussian_filter(chk, l * s, mode='nearest') for l in levels]
    out = np.zeros_like(chk)
    for k in range(len(levels) - 1):
        lo, hi = levels[k], levels[k + 1]
        t = np.clip((sigma - lo) / (hi - lo), 0, 1)
        m = (sigma >= lo) & ((sigma < hi) | (k == len(levels) - 2))
        out = np.where(m, stack[k] * (1 - t) + stack[k + 1] * t, out)
    # clip to the sphere, inside the outline
    d = np.hypot(X - CX, Y - CY)
    clip = np.clip((R - RIM - 1.5 - d) * s + 0.5, 0, 1)
    return out * clip


def outline(res):
    s = res / SIZE
    c = (np.arange(res) + 0.5) / s
    X, Y = np.meshgrid(c, c)
    d = np.hypot(X - CX, Y - CY)
    return np.clip((RIM / 2 - np.abs(d - (R - RIM / 2))) * s + 0.5, 0, 1)


def to_rgba(alpha):
    a = (np.clip(alpha, 0, 1) * 255 + 0.5).astype(np.uint8)
    im = Image.new('RGBA', a.shape[::-1], (255, 255, 255, 0))
    im.putalpha(Image.fromarray(a))
    return im


def png(scale=4):
    res = SIZE * scale
    a = np.maximum(interior(res), outline(res))
    return to_rgba(a).resize((SIZE, SIZE), Image.LANCZOS)


def svg():
    buf = io.BytesIO()
    to_rgba(interior(512)).save(buf, 'PNG', optimize=True)
    data = base64.b64encode(buf.getvalue()).decode()
    return '\n'.join([
        '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        'viewBox="0 0 %d %d" width="%d" height="%d">' % ((SIZE,) * 4),
        '<image x="0" y="0" width="%d" height="%d" xlink:href="data:image/png;base64,%s"/>' % (SIZE, SIZE, data),
        '<circle cx="%.1f" cy="%.1f" r="%.2f" fill="none" stroke="#ffffff" stroke-width="%.1f"/>'
        % (CX, CY, R - RIM / 2, RIM),
        '</svg>'])


if __name__ == '__main__':
    os.makedirs(DOCS, exist_ok=True)
    with open(os.path.join(DOCS, 'icon.svg'), 'w') as f:
        f.write(svg())
    png().save(os.path.join(DOCS, 'icon.png'))
    print('wrote docs/icon.svg, docs/icon.png')
