"""Generates the filter icon (docs/icon.svg, docs/icon.png).

A five-blade bokeh made of halftone dots that grow from left to right (the
non-uniform blur amount), split by the dashed seam line used by the other world
space filters.  The PNG (128x128) is embedded in the .sbs by build_wsnub.py.
"""
import math
import os

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(os.path.dirname(HERE), 'docs')

SIZE = 128
CX, CY = 64.0, 68.0
R_OUT = 56.0                 # pentagon circumradius
RINGS = 4
SEAM_X = 64.0
SEAM_CLEAR = 4.0


def pentagon(r):
    return [(CX + r * math.sin(2 * math.pi * k / 5), CY - r * math.cos(2 * math.pi * k / 5)) for k in range(5)]


def edge_distance(x, y):
    """signed distance to the pentagon edges (positive inside)."""
    pts = pentagon(R_OUT)
    d = 1e9
    for k in range(5):
        (x0, y0), (x1, y1) = pts[k], pts[(k + 1) % 5]
        nx, ny = y1 - y0, -(x1 - x0)          # outward normal for clockwise points
        ln = math.hypot(nx, ny)
        d = min(d, -((x - x0) * nx + (y - y0) * ny) / ln)
    return d


def dots():
    """concentric pentagon rings: bigger towards the centre (bokeh) and towards
    the right (the blur amount grows across the image)."""
    out = []
    for k in range(1, RINGS + 1):
        rr = R_OUT * k / RINGS
        pts = pentagon(rr)
        per_side = k
        for e in range(5):
            (x0, y0), (x1, y1) = pts[e], pts[(e + 1) % 5]
            for m in range(per_side):
                t = m / per_side
                x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
                if abs(x - SEAM_X) < SEAM_CLEAR:
                    continue
                u = min(max((x - (CX - R_OUT)) / (2 * R_OUT), 0.0), 1.0)
                grow = 0.28 + 0.72 * u ** 0.8
                ring = 1.0 - 0.55 * (k - 1) / (RINGS - 1)
                r = 8.2 * grow * ring
                r = min(r, abs(x - SEAM_X) - 2.25 - 3.0)   # keep a clear gap along the seam
                if r >= 1.0:
                    out.append((x, y, r))
    return out


def dashes():
    out = []
    y = 4.0
    while y < SIZE - 4:
        out.append((y, min(y + 12.0, SIZE - 4.0)))
        y += 18.0
    return out


def svg():
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d">' % (SIZE, SIZE, SIZE, SIZE),
             '<g fill="#ffffff">']
    for x, y, r in dots():
        parts.append('<circle cx="%.2f" cy="%.2f" r="%.2f"/>' % (x, y, r))
    for y0, y1 in dashes():
        parts.append('<rect x="%.1f" y="%.1f" width="4.5" height="%.1f" rx="2.25"/>' % (SEAM_X - 2.25, y0, y1 - y0))
    parts.append('</g></svg>')
    return '\n'.join(parts)


def png(scale=8):
    s = SIZE * scale
    im = Image.new('L', (s, s), 0)
    dr = ImageDraw.Draw(im)
    for x, y, r in dots():
        dr.ellipse([(x - r) * scale, (y - r) * scale, (x + r) * scale, (y + r) * scale], fill=255)
    for y0, y1 in dashes():
        dr.rounded_rectangle([(SEAM_X - 2.25) * scale, y0 * scale, (SEAM_X + 2.25) * scale, y1 * scale], radius=2.25 * scale, fill=255)
    a = im.resize((SIZE, SIZE), Image.LANCZOS)
    out = Image.new('RGBA', (SIZE, SIZE), (255, 255, 255, 0))
    out.putalpha(a)
    return out


if __name__ == '__main__':
    os.makedirs(DOCS, exist_ok=True)
    with open(os.path.join(DOCS, 'icon.svg'), 'w') as f:
        f.write(svg())
    png().save(os.path.join(DOCS, 'icon.png'))
    print('wrote docs/icon.svg, docs/icon.png')
