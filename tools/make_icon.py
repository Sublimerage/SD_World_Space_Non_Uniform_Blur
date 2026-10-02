"""Generates the filter icon (docs/icon.svg, docs/icon.png).

A sphere blurred in world space: most of it is a soft checker drawn as halftone
dots, while one cap (where the blur map is black) stays a crisp checker.  The
PNG (128x128) is embedded in the .sbs by build_wsnub.py.
"""
import math
import os

from PIL import Image, ImageChops, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(os.path.dirname(HERE), 'docs')

SIZE = 128
CX, CY, R = 64.0, 64.0, 57.0            # sphere silhouette
RIM = 2.6                                # outline width
CAP_X, CAP_Y, CAP_R = 95.0, 39.0, 33.0   # the sharp (unblurred) area
GAP = 3.5                                # empty band between the dots and the sharp area
CELL = 12.0                              # checker cell
DOT_PITCH = 7.0
ROT = math.radians(-18)                  # checker rotation, to suggest curvature


def checker_uv(x, y):
    dx, dy = x - CX, y - CY
    u = (dx * math.cos(ROT) - dy * math.sin(ROT)) / CELL
    v = (dx * math.sin(ROT) + dy * math.cos(ROT)) / CELL
    return u, v


def dots():
    """blurred checker as halftone: one dot per checker cell, big on white cells and
    small on black ones.  The contrast fades away from the sharp cap: the blur gets
    stronger with distance (non-uniform)."""
    out = []
    n = int(R / CELL) + 2
    for jj in range(-n, n + 1):
        for ii in range(-n, n + 1):
            u, v = ii + 0.5, jj + 0.5
            x = CX + (u * math.cos(ROT) + v * math.sin(ROT)) * CELL
            y = CY + (-u * math.sin(ROT) + v * math.cos(ROT)) * CELL
            white = 1.0 if (ii + jj) % 2 == 0 else 0.0
            dcap = math.hypot(x - CAP_X, y - CAP_Y) - CAP_R
            contrast = min(max(1.0 - dcap / 70.0, 0.2), 1.0)
            soft = 0.5 + (white - 0.5) * contrast
            r = CELL * (0.12 + 0.30 * soft)
            d = math.hypot(x - CX, y - CY)
            inside = d + r <= R - RIM - 2.0
            clear = dcap - r >= GAP
            if inside and clear and r >= 0.9:
                out.append((x, y, r))
    return out


def checker_cells():
    """the white checker cells (unclipped, icon space)."""
    cells = []
    for j in range(-8, 9):
        for i in range(-8, 9):
            if (i + j) % 2:
                continue
            pts = []
            for a, b in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
                x = CX + (a * math.cos(ROT) + b * math.sin(ROT)) * CELL
                y = CY + (-a * math.sin(ROT) + b * math.cos(ROT)) * CELL
                pts.append((x, y))
            cells.append(pts)
    return cells


def svg():
    p = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d">' % ((SIZE,) * 4),
         '<defs><clipPath id="sphere"><circle cx="%.1f" cy="%.1f" r="%.1f"/></clipPath>' % (CX, CY, R - RIM - 1.5),
         '<clipPath id="cap"><circle cx="%.1f" cy="%.1f" r="%.1f"/></clipPath></defs>' % (CAP_X, CAP_Y, CAP_R),
         '<g fill="#ffffff">',
         '<circle cx="%.1f" cy="%.1f" r="%.2f" fill="none" stroke="#ffffff" stroke-width="%.1f"/>'
         % (CX, CY, R - RIM / 2, RIM)]
    for x, y, r in dots():
        p.append('<circle cx="%.2f" cy="%.2f" r="%.2f"/>' % (x, y, r))
    p.append('<g clip-path="url(#sphere)"><g clip-path="url(#cap)">')
    for pts in checker_cells():
        p.append('<polygon points="%s"/>' % ' '.join('%.2f,%.2f' % q for q in pts))
    p.append('</g></g></g></svg>')
    return '\n'.join(p)


def png(scale=8):
    s = SIZE * scale

    def S(v):
        return v * scale

    def disc(cx, cy, r):
        im = Image.new('L', (s, s), 0)
        ImageDraw.Draw(im).ellipse([S(cx - r), S(cy - r), S(cx + r), S(cy + r)], fill=255)
        return im
    base = Image.new('L', (s, s), 0)
    dr = ImageDraw.Draw(base)
    dr.ellipse([S(CX - R), S(CY - R), S(CX + R), S(CY + R)], outline=255, width=int(RIM * scale))
    for x, y, r in dots():
        dr.ellipse([S(x - r), S(y - r), S(x + r), S(y + r)], fill=255)
    chk = Image.new('L', (s, s), 0)
    dc = ImageDraw.Draw(chk)
    for pts in checker_cells():
        dc.polygon([(S(x), S(y)) for x, y in pts], fill=255)
    chk = ImageChops.multiply(ImageChops.multiply(chk, disc(CX, CY, R - RIM - 1.5)), disc(CAP_X, CAP_Y, CAP_R))
    a = ImageChops.lighter(base, chk).resize((SIZE, SIZE), Image.LANCZOS)
    out = Image.new('RGBA', (SIZE, SIZE), (255, 255, 255, 0))
    out.putalpha(a)
    return out


if __name__ == '__main__':
    os.makedirs(DOCS, exist_ok=True)
    with open(os.path.join(DOCS, 'icon.svg'), 'w') as f:
        f.write(svg())
    png().save(os.path.join(DOCS, 'icon.png'))
    print('wrote docs/icon.svg, docs/icon.png')
