#!/usr/bin/env python3
"""Export Twemoji SVGs to Blender Grease Pencil, with face-part tagging.

Usage (system Python 3.8+, no third-party packages needed):

    python3 blender.py --emojis joy,1f600,1f60d --output ./output/blender/
    python3 blender.py --emojis all --output ./output/blender/ --facemap-only

For every emoji this writes:

    <name>.blend          Grease Pencil object, one layer per face part
    <name>.facemap.json   which shapes were identified as eyes, mouth, ...
    <name>.facemap.svg    (--debug-svg) the labelled shapes, to eyeball the mapping

How it works
------------
This one file runs in two modes.

1. Host mode (plain ``python3``): parses the SVG, flattens every path into
   polygons, works out which polygons are the face / eyes / mouth / ..., and
   writes a job file. It then starts Blender in the background as a subprocess,
   running *this same file* with ``--python``.
2. Blender mode (``--blender-side``): reads the job file and builds the
   Grease Pencil data with ``bpy``. All the geometry work has already been
   done, so this half is deliberately thin.

Blender >= 4.3 is required (the new "Grease Pencil v3" data model).

Layer naming (what you will find in the .blend)
-----------------------------------------------
    face                      the head shape
    eye_left / eye_right      left/right *as seen by the viewer* (screen side)
    pupil_*  / highlight_*    shapes contained in an eye
    brow_left / brow_right
    mouth, teeth, tongue, mouth_detail
    tear_*, blush_*
    other_01, other_02, ...   everything else, split into runs so the original
                              SVG paint order is preserved

Each layer's origin (pivot) sits at the centre of its shapes, so scaling an
eye layer on Z blinks around the eye, rotating a brow pivots around the brow.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent
SVG_DIR = REPO_ROOT / "data" / "svg"
EMOJI_JSON = REPO_ROOT / "data" / "emojis.json"

MIN_BLENDER = (4, 3)  # Grease Pencil v3 data model

Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# Small geometry helpers
# ---------------------------------------------------------------------------

def signed_area(pts: List[Point]) -> float:
    s = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s / 2.0


def polygon_centroid(pts: List[Point]) -> Point:
    a = signed_area(pts)
    if abs(a) < 1e-9:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
    cx = cy = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        f = x1 * y2 - x2 * y1
        cx += (x1 + x2) * f
        cy += (y1 + y2) * f
    return (cx / (6 * a), cy / (6 * a))


def point_in_polygon(p: Point, pts: List[Point]) -> bool:
    x, y = p
    inside = False
    n = len(pts)
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (yi > y) != (yj > y):
            if x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                inside = not inside
        j = i
    return inside


def interior_point(pts: List[Point]) -> Point:
    """A point that is reliably *inside* the polygon (centre of its widest
    horizontal span), unlike the centroid which can fall outside for crescents
    and arcs."""
    ys = [p[1] for p in pts]
    y0, y1 = min(ys), max(ys)
    h = y1 - y0
    best: Optional[Tuple[float, Point]] = None
    n = len(pts)
    for frac in (0.5, 0.35, 0.65, 0.25, 0.75):
        y = y0 + h * frac + 1e-7
        xs = []
        for i in range(n):
            xa, ya = pts[i]
            xb, yb = pts[(i + 1) % n]
            if (ya <= y < yb) or (yb <= y < ya):
                xs.append(xa + (y - ya) * (xb - xa) / (yb - ya))
        xs.sort()
        for k in range(0, len(xs) - 1, 2):
            width = xs[k + 1] - xs[k]
            if best is None or width > best[0]:
                best = (width, ((xs[k] + xs[k + 1]) / 2, y))
    if best is not None:
        return best[1]
    return polygon_centroid(pts)


def mat_mul(m1, m2):
    """Compose affine matrices (a, b, c, d, e, f): result applies m2 first."""
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def mat_apply(m, p: Point) -> Point:
    a, b, c, d, e, f = m
    return (a * p[0] + c * p[1] + e, b * p[0] + d * p[1] + f)


_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def parse_transform(s: str):
    m = IDENTITY
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", s or ""):
        v = [float(x) for x in _NUM_RE.findall(args)]
        if name == "matrix" and len(v) == 6:
            t = tuple(v)
        elif name == "translate":
            t = (1, 0, 0, 1, v[0], v[1] if len(v) > 1 else 0.0)
        elif name == "scale":
            t = (v[0], 0, 0, v[1] if len(v) > 1 else v[0], 0, 0)
        elif name == "rotate":
            a = math.radians(v[0])
            ca, sa = math.cos(a), math.sin(a)
            t = (ca, sa, -sa, ca, 0, 0)
            if len(v) == 3:
                t = mat_mul(mat_mul((1, 0, 0, 1, v[1], v[2]), t), (1, 0, 0, 1, -v[1], -v[2]))
        elif name == "skewX":
            t = (1, 0, math.tan(math.radians(v[0])), 1, 0, 0)
        elif name == "skewY":
            t = (1, math.tan(math.radians(v[0])), 0, 1, 0, 0)
        else:
            continue
        m = mat_mul(m, t)
    return m


# ---------------------------------------------------------------------------
# SVG path flattening
# ---------------------------------------------------------------------------

class _Scanner:
    def __init__(self, s: str):
        self.s = s
        self.i = 0

    def _skip(self):
        s, n = self.s, len(self.s)
        while self.i < n and s[self.i] in " \t\r\n,":
            self.i += 1

    def command(self) -> Optional[str]:
        self._skip()
        if self.i < len(self.s) and self.s[self.i].isalpha():
            c = self.s[self.i]
            self.i += 1
            return c
        return None

    def has_number(self) -> bool:
        self._skip()
        return self.i < len(self.s) and (self.s[self.i] in "+-." or self.s[self.i].isdigit())

    def number(self) -> float:
        self._skip()
        m = _NUM_RE.match(self.s, self.i)
        if not m:
            raise ValueError(f"expected number at {self.i} in path data")
        self.i = m.end()
        return float(m.group())

    def flag(self) -> int:
        # Arc flags may be packed with no separators ("a2 2 0 0112 3").
        self._skip()
        c = self.s[self.i] if self.i < len(self.s) else ""
        if c not in ("0", "1"):
            raise ValueError(f"expected arc flag at {self.i} in path data")
        self.i += 1
        return int(c)


def _cubic(p0, p1, p2, p3, n):
    out = []
    for i in range(1, n + 1):
        t = i / n
        mt = 1 - t
        out.append((
            mt ** 3 * p0[0] + 3 * mt * mt * t * p1[0] + 3 * mt * t * t * p2[0] + t ** 3 * p3[0],
            mt ** 3 * p0[1] + 3 * mt * mt * t * p1[1] + 3 * mt * t * t * p2[1] + t ** 3 * p3[1],
        ))
    return out


def _quad(p0, p1, p2, n):
    out = []
    for i in range(1, n + 1):
        t = i / n
        mt = 1 - t
        out.append((
            mt * mt * p0[0] + 2 * mt * t * p1[0] + t * t * p2[0],
            mt * mt * p0[1] + 2 * mt * t * p1[1] + t * t * p2[1],
        ))
    return out


def _arc(x1, y1, rx, ry, phi_deg, fa, fs, x2, y2, n):
    """SVG elliptical arc (spec appendix F.6.5) flattened to points."""
    if x1 == x2 and y1 == y2:
        return []
    rx, ry = abs(rx), abs(ry)
    if rx == 0 or ry == 0:
        return [(x2, y2)]
    phi = math.radians(phi_deg)
    cp, sp = math.cos(phi), math.sin(phi)
    dx, dy = (x1 - x2) / 2, (y1 - y2) / 2
    x1p = cp * dx + sp * dy
    y1p = -sp * dx + cp * dy
    lam = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if lam > 1:
        s = math.sqrt(lam)
        rx *= s
        ry *= s
    num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    coef = math.sqrt(max(0.0, num / den)) if den else 0.0
    if fa == fs:
        coef = -coef
    cxp = coef * rx * y1p / ry
    cyp = -coef * ry * x1p / rx
    cx = cp * cxp - sp * cyp + (x1 + x2) / 2
    cy = sp * cxp + cp * cyp + (y1 + y2) / 2

    def ang(ux, uy, vx, vy):
        return math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)

    th1 = ang(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dth = ang((x1p - cxp) / rx, (y1p - cyp) / ry, (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if fs == 0 and dth > 0:
        dth -= 2 * math.pi
    elif fs == 1 and dth < 0:
        dth += 2 * math.pi
    segs = max(1, math.ceil(n * abs(dth) / (math.pi / 2)))
    out = []
    for i in range(1, segs + 1):
        t = th1 + dth * i / segs
        out.append((
            cp * rx * math.cos(t) - sp * ry * math.sin(t) + cx,
            sp * rx * math.cos(t) + cp * ry * math.sin(t) + cy,
        ))
    out[-1] = (x2, y2)
    return out


def flatten_path(d: str, n: int = 8) -> List[List[Point]]:
    """Flatten SVG path data into polylines (one per sub-path).

    ``n`` is the number of line segments per Bezier curve (and per 90 degrees
    of an arc) - the same knob as ``--curve-segments`` in the TS CLI.
    """
    sc = _Scanner(d)
    subpaths: List[List[Point]] = []
    cur: Optional[List[Point]] = None
    x = y = sx = sy = 0.0
    cmd: Optional[str] = None
    last_c2: Optional[Point] = None  # last cubic control point (for S)
    last_q: Optional[Point] = None   # last quad control point (for T)

    def finish():
        nonlocal cur
        if cur is not None and len(cur) >= 2:
            subpaths.append(cur)
        cur = None

    while True:
        c = sc.command()
        if c is not None:
            cmd = c
        elif sc.has_number() and cmd is not None and cmd not in "Zz":
            if cmd == "M":
                cmd = "L"
            elif cmd == "m":
                cmd = "l"
        else:
            break

        rel = cmd.islower()
        up = cmd.upper()
        ox, oy = (x, y) if rel else (0.0, 0.0)

        if up == "Z":
            if cur is not None:
                finish()
            x, y = sx, sy
            last_c2 = last_q = None
            continue

        if up == "M":
            finish()
            x, y = ox + sc.number(), oy + sc.number()
            sx, sy = x, y
            cur = [(x, y)]
            last_c2 = last_q = None
            continue

        if cur is None:
            cur = [(x, y)]
            sx, sy = x, y

        if up == "L":
            x, y = ox + sc.number(), oy + sc.number()
            cur.append((x, y))
            last_c2 = last_q = None
        elif up == "H":
            x = (x if rel else 0.0) + sc.number()
            cur.append((x, y))
            last_c2 = last_q = None
        elif up == "V":
            y = (y if rel else 0.0) + sc.number()
            cur.append((x, y))
            last_c2 = last_q = None
        elif up == "C":
            p1 = (ox + sc.number(), oy + sc.number())
            p2 = (ox + sc.number(), oy + sc.number())
            p3 = (ox + sc.number(), oy + sc.number())
            cur.extend(_cubic((x, y), p1, p2, p3, n))
            x, y = p3
            last_c2, last_q = p2, None
        elif up == "S":
            p1 = (2 * x - last_c2[0], 2 * y - last_c2[1]) if last_c2 else (x, y)
            p2 = (ox + sc.number(), oy + sc.number())
            p3 = (ox + sc.number(), oy + sc.number())
            cur.extend(_cubic((x, y), p1, p2, p3, n))
            x, y = p3
            last_c2, last_q = p2, None
        elif up == "Q":
            p1 = (ox + sc.number(), oy + sc.number())
            p2 = (ox + sc.number(), oy + sc.number())
            cur.extend(_quad((x, y), p1, p2, n))
            x, y = p2
            last_q, last_c2 = p1, None
        elif up == "T":
            p1 = (2 * x - last_q[0], 2 * y - last_q[1]) if last_q else (x, y)
            p2 = (ox + sc.number(), oy + sc.number())
            cur.extend(_quad((x, y), p1, p2, n))
            x, y = p2
            last_q, last_c2 = p1, None
        elif up == "A":
            rx, ry, rot = sc.number(), sc.number(), sc.number()
            fa, fs = sc.flag(), sc.flag()
            ex, ey = ox + sc.number(), oy + sc.number()
            cur.extend(_arc(x, y, rx, ry, rot, fa, fs, ex, ey, n))
            x, y = ex, ey
            last_c2 = last_q = None
        else:
            raise ValueError(f"unsupported path command {cmd!r}")

    finish()
    return subpaths


def _clean_polygon(pts: List[Point]) -> List[Point]:
    out: List[Point] = []
    for p in pts:
        if not out or abs(p[0] - out[-1][0]) > 1e-9 or abs(p[1] - out[-1][1]) > 1e-9:
            out.append(p)
    while len(out) > 1 and abs(out[0][0] - out[-1][0]) < 1e-9 and abs(out[0][1] - out[-1][1]) < 1e-9:
        out.pop()
    return out


def _ellipse_polygon(cx, cy, rx, ry, n) -> List[Point]:
    count = max(8, 4 * n)
    return [
        (cx + rx * math.cos(2 * math.pi * i / count), cy + ry * math.sin(2 * math.pi * i / count))
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

_NAMED = {
    "red": "#ff0000", "green": "#008000", "navy": "#000080",
    "white": "#ffffff", "black": "#000000",
}


def parse_color(s: str) -> Optional[str]:
    """Return '#rrggbb', or None if unparseable."""
    s = s.strip().lower()
    if s in _NAMED:
        return _NAMED[s]
    m = re.fullmatch(r"#([0-9a-f]{3})", s)
    if m:
        return "#" + "".join(ch * 2 for ch in m.group(1))
    if re.fullmatch(r"#[0-9a-f]{6}", s):
        return s
    return None


def hex_rgb(h: str) -> Tuple[int, int, int]:
    return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)


def color_dist(a: str, b: str) -> float:
    ra, rb = hex_rgb(a), hex_rgb(b)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(ra, rb)))


def luminance(h: str) -> float:
    r, g, b = hex_rgb(h)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


# ---------------------------------------------------------------------------
# SVG -> units (one unit = one filled polygon)
# ---------------------------------------------------------------------------

@dataclass
class Unit:
    """One filled polygon. A single SVG <path> with several sub-paths becomes
    several units - this matters because Twemoji often draws both eyes, the
    brows *and* the mouth in one <path>."""
    pts: List[Point]
    color: str
    alpha: float
    elem: int
    order: float = 0.0
    hole: bool = False
    parent: Optional["Unit"] = None  # for holes: the shape they are cut from
    uid: int = -1
    label: str = "other"
    rule: str = ""
    confidence: float = 0.0
    bbox: Tuple[float, float, float, float] = field(init=False)
    area: float = field(init=False)
    centroid: Point = field(init=False)

    def __post_init__(self):
        xs = [p[0] for p in self.pts]
        ys = [p[1] for p in self.pts]
        self.bbox = (min(xs), min(ys), max(xs), max(ys))
        self.area = abs(signed_area(self.pts))
        self.centroid = polygon_centroid(self.pts)

    @property
    def w(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def h(self) -> float:
        return self.bbox[3] - self.bbox[1]

    def interior(self) -> Point:
        return interior_point(self.pts)


_SKIP_TAGS = {"defs", "clipPath", "title", "desc", "metadata", "style", "mask"}


def _local(tag: str) -> str:
    return tag.split("}")[-1]


@dataclass
class _Elem:
    id: int
    color: str
    alpha: float
    rule: str
    subpaths: List[List[Point]]


def _detect_holes(subpaths: List[List[Point]], rule: str):
    """For every sub-path of one SVG element decide whether it is a hole in
    another sub-path. Returns [(is_hole, parent_index_or_None)]."""
    n = len(subpaths)
    areas = [abs(signed_area(p)) for p in subpaths]
    orient = [1 if signed_area(p) > 0 else -1 for p in subpaths]
    # Nesting is decided from boundary vertices, not an "interior point": the
    # interior point of a ring's outer polygon sits in the middle of the ring,
    # i.e. inside the hole, which would make the two sub-paths contain each other.
    probes = [(p[0], p[len(p) // 2]) for p in subpaths]
    result = []
    for i in range(n):
        containers = [j for j in range(n)
                      if j != i and areas[j] > areas[i]
                      and all(point_in_polygon(q, subpaths[j]) for q in probes[i])]
        parent = min(containers, key=lambda j: areas[j]) if containers else None
        if rule == "evenodd":
            is_hole = len(containers) % 2 == 1
        else:  # nonzero
            wn = sum(orient[j] for j in containers)
            is_hole = wn != 0 and wn + orient[i] == 0
        result.append((is_hole, parent))
    return result


@dataclass
class ParsedSvg:
    vb: Tuple[float, float, float, float]  # minx, miny, w, h
    units: List[Unit]
    warnings: List[str]


def parse_svg(path: Path, curve_segments: int = 8) -> ParsedSvg:
    root = ET.parse(str(path)).getroot()
    vb_attr = root.get("viewBox")
    if vb_attr:
        vb = tuple(float(v) for v in re.split(r"[\s,]+", vb_attr.strip()))
    else:
        vb = (0.0, 0.0, float(root.get("width", 36)), float(root.get("height", 36)))
    warnings: List[str] = []
    elems: List[_Elem] = []

    def warn(msg: str):
        if msg not in warnings:
            warnings.append(msg)

    def walk(el, ctm, st):
        tag = _local(el.tag)
        if tag in _SKIP_TAGS:
            return
        st = dict(st)
        for k in ("fill", "fill-opacity", "fill-rule"):
            if el.get(k) is not None:
                st[k] = el.get(k)
        if el.get("opacity") is not None:
            st["opacity"] = st["opacity"] * float(el.get("opacity"))
        if el.get("clip-path"):
            warn("clip-path ignored")
        stroke = el.get("stroke")
        if stroke and stroke != "none":
            warn("stroke ignored (fills only)")
        m = mat_mul(ctm, parse_transform(el.get("transform"))) if el.get("transform") else ctm

        if tag in ("svg", "g"):
            for child in el:
                walk(child, m, st)
            return

        if tag == "path":
            raw = flatten_path(el.get("d", ""), curve_segments)
        elif tag == "circle":
            r = float(el.get("r", 0))
            raw = [_ellipse_polygon(float(el.get("cx", 0)), float(el.get("cy", 0)), r, r, curve_segments)]
        elif tag == "ellipse":
            raw = [_ellipse_polygon(float(el.get("cx", 0)), float(el.get("cy", 0)),
                                    float(el.get("rx", 0)), float(el.get("ry", 0)), curve_segments)]
        else:
            warn(f"<{tag}> not supported, skipped")
            return

        fill = st["fill"]
        if fill == "none":
            return
        color = parse_color(fill)
        if color is None:
            warn(f"unrecognised fill {fill!r}, used black")
            color = "#000000"
        subpaths = []
        for sp in raw:
            poly = _clean_polygon([mat_apply(m, p) for p in sp])
            if len(poly) >= 3 and abs(signed_area(poly)) > 1e-9:
                subpaths.append(poly)
        if subpaths:
            alpha = max(0.0, min(1.0, st["opacity"] * float(st["fill-opacity"])))
            elems.append(_Elem(len(elems), color, alpha, st["fill-rule"], subpaths))

    walk(root, IDENTITY, {"fill": "#000000", "fill-opacity": "1", "fill-rule": "nonzero", "opacity": 1.0})

    # Elements -> units, resolving holes.
    units: List[Unit] = []
    seq = 0
    for e in elems:
        holes = _detect_holes(e.subpaths, e.rule)
        local: Dict[int, Unit] = {}
        for k, poly in enumerate(e.subpaths):
            if not holes[k][0]:
                u = Unit(poly, e.color, e.alpha, e.id, order=float(seq))
                seq += 1
                local[k] = u
                units.append(u)
        last_order = max((u.order for u in local.values()), default=float(seq))
        for k, poly in enumerate(e.subpaths):
            if not holes[k][0]:
                continue
            # A hole shows whatever was painted underneath: use the colour of
            # the topmost earlier shape covering it (painter's algorithm).
            ip = interior_point(poly)
            under = None
            for cand in reversed(units):
                if cand.elem < e.id and not cand.hole and point_in_polygon(ip, cand.pts):
                    under = cand
                    break
            if under is None:
                warn("hole with nothing underneath dropped")
                continue
            parent_k = holes[k][1]
            for _ in range(len(e.subpaths)):  # bounded walk up the nesting chain
                if parent_k is None or parent_k in local:
                    break
                parent_k = holes[parent_k][1]
            else:
                parent_k = None
            parent = local.get(parent_k) if parent_k is not None else None
            base = parent.order if parent is not None else last_order
            n_before = sum(1 for u in units if u.hole and u.elem == e.id)
            h = Unit(poly, under.color, 1.0, e.id, order=base + 0.001 * (n_before + 1),
                     hole=True, parent=parent)
            units.append(h)

    units.sort(key=lambda u: u.order)
    for i, u in enumerate(units):
        u.uid = i
    return ParsedSvg(vb, units, warnings)


# ---------------------------------------------------------------------------
# Face-part classification
# ---------------------------------------------------------------------------

def _side(u: float) -> str:
    return "left" if u < 0 else "right"


def _is_whiteish(c: str) -> bool:
    r, g, b = hex_rgb(c)
    return min(r, g, b) >= 200 and max(r, g, b) - min(r, g, b) <= 40


def _is_reddish(c: str) -> bool:
    r, g, b = hex_rgb(c)
    return r >= 170 and r - g >= 60 and r - b >= 30


def _is_bluish(c: str) -> bool:
    r, g, b = hex_rgb(c)
    return b >= 180 and b - r >= 50 and b >= g


def _is_blush(c: str) -> bool:
    """Soft pink (cheek blush). Deliberately excludes saturated reds such as
    the hearts in heart-eyes, which have a much lower green channel."""
    r, g, b = hex_rgb(c)
    return r >= 190 and r - g >= 50 and 90 <= g <= 175 and 100 <= b <= 200


def _is_roundish(u: Unit) -> bool:
    """Circle/ellipse-like outline (rules out hearts, hands, stars)."""
    fit = u.area / (math.pi * (u.w / 2) * (u.h / 2))
    return 0.85 <= fit <= 1.15 and 0.4 <= u.w / u.h <= 2.5


def _mostly_inside(inner: Unit, outer: Unit) -> bool:
    if not point_in_polygon(inner.interior(), outer.pts):
        return False
    inside = sum(1 for p in inner.pts if point_in_polygon(p, outer.pts))
    return inside / len(inner.pts) >= 0.6


def classify(units: List[Unit], vb) -> dict:
    """Label each unit with a face part. Mutates ``unit.label/rule/confidence``
    and returns a summary dict."""
    _, _, vw, vh = vb
    for u in units:
        u.label, u.rule, u.confidence = "other", "", 0.0

    solids = [u for u in units if not u.hole]
    summary = {"is_face": False, "head": None}

    # --- head: the big shape that fills (most of) the canvas ---------------
    # It must also be round: an ellipse has area = pi*a*b, whereas the rounded
    # square backgrounds of flags and buttons come out at ~4/pi = 1.27.
    head = None
    for u in sorted(solids, key=lambda u: -u.area):
        if u.w >= 0.6 * vw and u.h >= 0.6 * vh and u.area >= 0.3 * vw * vh:
            ellipse_fit = u.area / (math.pi * (u.w / 2) * (u.h / 2))
            if 0.75 <= ellipse_fit <= 1.12 and 0.75 <= u.w / u.h <= 1.33:
                head = u
                break
    if head is None:
        return summary

    hcx, hcy = (head.bbox[0] + head.bbox[2]) / 2, (head.bbox[1] + head.bbox[3]) / 2
    hw, hh = head.w / 2, head.h / 2
    norm: Dict[int, Tuple[float, float]] = {}

    def nm(u: Unit):
        if u.uid not in norm:
            norm[u.uid] = ((u.centroid[0] - hcx) / hw, (u.centroid[1] - hcy) / hh)
        return norm[u.uid]

    feats = [u for u in solids if u is not head and u.order > head.order
             and abs(nm(u)[0]) <= 1.1 and abs(nm(u)[1]) <= 1.1]
    taken: set = set()

    def mark(u: Unit, label: str, rule: str, conf: float):
        u.label, u.rule, u.confidence = label, rule, conf
        taken.add(u.uid)

    # --- eyes: a mirror-symmetric pair in the upper half --------------------
    cand = [f for f in feats
            if -0.85 <= nm(f)[1] <= 0.2 and 0.12 <= abs(nm(f)[0]) <= 0.85
            and 0.002 <= f.area / head.area <= 0.2 and f.w / (2 * hw) <= 0.5]

    # Tears and blush are never eyes, even when they happen to be symmetric.
    cand = [f for f in cand if not _is_bluish(f.color) and not _is_blush(f.color)]

    def valid_pairs(tier: str):
        found = []
        for i in range(len(cand)):
            for j in range(i + 1, len(cand)):
                a, b = cand[i], cand[j]
                ua, va = nm(a)
                ub, vb_ = nm(b)
                if ua * ub >= 0:
                    continue
                ratio = min(a.area, b.area) / max(a.area, b.area)
                dv, mir = abs(va - vb_), abs(ua + ub)
                cd = color_dist(a.color, b.color)
                if tier == "strict":
                    ok = ratio >= 0.5 and dv <= 0.12 and mir <= 0.15 and cd <= 48
                else:
                    ok = ratio >= 0.12 and dv <= 0.3 and mir <= 0.3 and cd <= 120
                if not ok:
                    continue
                found.append((dv + mir + 0.5 * (1 - ratio) + cd / 255, a, b))
        return found

    def choose_pair(pairs):
        """Best-scoring pair - but brows form a symmetric pair too, and sit
        above the eyes, so prefer the lowest pair among comparably good ones."""
        pairs = sorted(pairs, key=lambda p: p[0])
        best = pairs[0]

        def mean_v(p):
            return (nm(p[1])[1] + nm(p[2])[1]) / 2

        lowest = max((p for p in pairs if p[0] <= best[0] + 0.35), key=mean_v)
        return lowest if mean_v(lowest) >= mean_v(best) + 0.1 else best

    eyes: Dict[str, Unit] = {}
    pair, conf, rule = None, 1.0, "eye-pair"
    strict = valid_pairs("strict")
    if strict:
        pair = choose_pair(strict)
    else:
        relaxed = valid_pairs("relaxed")
        if relaxed:
            pair, conf, rule = choose_pair(relaxed), 0.6, "eye-pair-relaxed"
    if pair is not None:
        _, a, b = pair
        for e in (a, b):
            side = _side(nm(e)[0])
            eyes[side] = e
            mark(e, f"eye_{side}", rule, conf)

    # --- mouth: biggest non-skin shape low and central ----------------------
    mouth = None
    mcand = [f for f in feats if f.uid not in taken and nm(f)[1] >= 0.05
             and abs(nm(f)[0]) <= 0.45 and color_dist(f.color, head.color) > 40]
    # Prefer non-blue shapes (a pool of tears is blue and can be bigger than
    # the mouth) and ignore anything spanning the whole width of the face.
    mcand = [f for f in mcand if f.w / (2 * hw) <= 0.9]
    if mcand:
        mouth = max(mcand, key=lambda f: (not _is_bluish(f.color), f.area))
        mark(mouth, "mouth", "mouth-lower-center", 0.8)

    # --- single-eye fallback (winks, cyclops) only if a mouth was found -----
    if not eyes and mouth is not None:
        scand = [f for f in feats if f.uid not in taken and -0.7 <= nm(f)[1] <= 0.0
                 and 0.1 <= abs(nm(f)[0]) <= 0.75 and color_dist(f.color, head.color) > 40
                 and 0.002 <= f.area / head.area <= 0.2]
        if scand:
            e = max(scand, key=lambda f: f.area)
            side = _side(nm(e)[0])
            eyes[side] = e
            mark(e, f"eye_{side}", "eye-single", 0.4)

    # --- details inside eyes: pupils / highlights ---------------------------
    for side, eye in eyes.items():
        for f in feats:
            if f.uid in taken or f.order <= eye.order or f.area >= eye.area:
                continue
            if _mostly_inside(f, eye):
                kind = "pupil" if luminance(f.color) < luminance(eye.color) - 20 else "highlight"
                mark(f, f"{kind}_{side}", "inside-eye", 0.7)

    # --- teeth / tongue / extra shapes inside the mouth ---------------------
    if mouth is not None:
        for f in feats:
            if f.uid in taken or f.order <= mouth.order or f.area >= mouth.area:
                continue
            if _mostly_inside(f, mouth):
                if _is_whiteish(f.color):
                    mark(f, "teeth", "inside-mouth-white", 0.7)
                elif _is_reddish(f.color):
                    mark(f, "tongue", "inside-mouth-red", 0.7)
                else:
                    mark(f, "mouth_detail", "inside-mouth", 0.5)

    # --- eyebrows: thin dark shapes above an eye -----------------------------
    for side, eye in eyes.items():
        for f in feats:
            if f.uid in taken:
                continue
            u_, v_ = nm(f)
            eu, ev = nm(eye)
            if _side(u_) != side or not (0.02 <= ev - v_ <= 0.6) or v_ < -0.98:
                continue
            thin = f.area / max(f.w * f.h, 1e-9) < 0.45
            overlap = min(f.bbox[2], eye.bbox[2]) - max(f.bbox[0], eye.bbox[0])
            # Brows are dark. (Don't compare to the eye colour alone: eyes can be white.)
            dark_or_like_eye = luminance(f.color) < 120 or color_dist(f.color, eye.color) <= 90
            if thin and overlap >= 0.3 * min(f.w, eye.w) and f.w / (2 * hw) <= 0.5 and dark_or_like_eye:
                mark(f, f"brow_{side}", "thin-above-eye", 0.6)

    # --- tears / blush on the sides -----------------------------------------
    for f in feats:
        if f.uid in taken:
            continue
        u_, v_ = nm(f)
        if _is_bluish(f.color) and abs(u_) >= 0.2:
            mark(f, f"tear_{_side(u_)}", "blue-on-side", 0.5)
        elif _is_blush(f.color) and abs(u_) >= 0.3 and v_ >= -0.15 and _is_roundish(f):
            mark(f, f"blush_{_side(u_)}", "pink-on-cheek", 0.5)

    is_face = bool(eyes) or mouth is not None
    if not is_face:
        for u in units:
            u.label, u.rule, u.confidence = "other", "", 0.0
        return summary

    mark(head, "face", "head", 1.0)
    for u in units:
        if u.hole and u.parent is not None:
            u.label, u.rule, u.confidence = u.parent.label, "hole-of-" + u.parent.label, u.parent.confidence
    summary.update(
        is_face=True,
        # True only when both eyes AND a mouth were found - use this when you
        # need to be sure; `is_face` alone is deliberately permissive.
        complete=len(eyes) == 2 and mouth is not None,
        head={"bbox": list(head.bbox), "color": head.color},
    )
    return summary


# ---------------------------------------------------------------------------
# Units -> layered model (the thing the Blender side consumes)
# ---------------------------------------------------------------------------

def _bboxes_overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def build_layers(units: List[Unit]):
    """Group units into layers by part label, keeping SVG paint order.

    Labelled parts get one layer each (positioned at their first appearance);
    unlabelled shapes are split into runs so they stay in front of / behind
    the right parts. Returns (ordered list of (layer_name, part, [units]), warnings).
    """
    layers: Dict[str, List[Unit]] = {}
    parts: Dict[str, str] = {}
    run = 0
    prev_other = False
    for u in units:
        if u.label == "other":
            if not prev_other:
                run += 1
            key = f"other_{run:02d}"
            prev_other = True
        else:
            key = u.label
            prev_other = False
        layers.setdefault(key, []).append(u)
        parts.setdefault(key, u.label)
    ordered = sorted(layers.items(), key=lambda kv: kv[1][0].order)

    # Paint-order sanity check: layers are drawn whole, so an overlapping pair
    # of shapes can only come out wrong if their layers are ordered against
    # their original paint order.
    warnings: List[str] = []
    if len(units) <= 400:
        layer_pos = {name: i for i, (name, _) in enumerate(ordered)}
        key_of = {u.uid: name for name, us in ordered for u in us}
        bad = 0
        for i, a in enumerate(units):
            for b in units[i + 1:]:
                if key_of[a.uid] != key_of[b.uid] and layer_pos[key_of[a.uid]] > layer_pos[key_of[b.uid]] \
                        and _bboxes_overlap(a.bbox, b.bbox):
                    bad += 1
        if bad:
            warnings.append(f"{bad} overlapping shape pair(s) end up in the wrong paint order between layers")
    return [(name, parts[name], us) for name, us in ordered], warnings


def build_model(code: str, name: str, svg_path: Path, size: float, curve_segments: int):
    parsed = parse_svg(svg_path, curve_segments)
    summary = classify(parsed.units, parsed.vb)
    layer_list, order_warnings = build_layers(parsed.units)

    minx, miny, vw, vh = parsed.vb
    cx, cy = minx + vw / 2, miny + vh / 2
    s = size / max(vw, vh)

    def to_blender(p: Point):
        # SVG is x-right / y-down; Blender front view is X-right / Z-up.
        return ((p[0] - cx) * s, 0.0, -(p[1] - cy) * s)

    layers_out = []
    facemap_parts: Dict[str, dict] = {}
    for lname, part, us in layer_list:
        strokes = []
        xs, zs = [], []
        for u in us:
            pts = [to_blender(p) for p in u.pts]
            xs += [p[0] for p in pts]
            zs += [p[2] for p in pts]
            strokes.append({"points": pts, "color": u.color, "alpha": u.alpha, "hole": u.hole})
        ox, oz = (min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2
        for st in strokes:
            st["points"] = [[p[0] - ox, p[1], p[2] - oz] for p in st["points"]]
        layers_out.append({"name": lname, "part": part, "origin": [ox, 0.0, oz], "strokes": strokes})
        if part != "other":
            facemap_parts[part] = {
                "layer": lname,
                "strokes": len(strokes),
                "center": [ox, oz],
                "bbox": [min(xs), min(zs), max(xs), max(zs)],
                "svg_bbox": [min(u.bbox[0] for u in us), min(u.bbox[1] for u in us),
                             max(u.bbox[2] for u in us), max(u.bbox[3] for u in us)],
                "colors": sorted({u.color for u in us}),
                "rule": us[0].rule,
                "confidence": us[0].confidence,
            }

    groups = {
        "eyes": [p for p in ("eye_left", "eye_right") if p in facemap_parts],
        "pupils": [p for p in ("pupil_left", "pupil_right") if p in facemap_parts],
        "brows": [p for p in ("brow_left", "brow_right") if p in facemap_parts],
        "mouth": [p for p in ("mouth", "teeth", "tongue", "mouth_detail") if p in facemap_parts],
    }
    facemap = {
        "code": code,
        "name": name,
        "is_face": summary["is_face"],
        "complete": bool(summary.get("complete")),
        "note": "left/right are screen-side (as seen by the viewer), not the character's own left/right",
        "parts": facemap_parts,
        "groups": {k: v for k, v in groups.items() if v},
        "warnings": parsed.warnings + order_warnings,
    }
    model = {
        "name": name,
        "code": code,
        "size": size,
        "layers": layers_out,
        "facemap": facemap,
    }
    return model, parsed


# ---------------------------------------------------------------------------
# Debug SVG: shows what was labelled as what
# ---------------------------------------------------------------------------

_LABEL_COLORS = {
    "face": "#999999", "eye": "#00a0ff", "pupil": "#0040ff", "highlight": "#00d0c0",
    "brow": "#a000ff", "mouth": "#ff0060", "teeth": "#ff8000", "tongue": "#ff4040",
    "tear": "#0080c0", "blush": "#ff60a0",
}


def write_debug_svg(path: Path, parsed: ParsedSvg):
    minx, miny, vw, vh = parsed.vb
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{minx} {miny} {vw} {vh + 4}">',
           f'<rect x="{minx}" y="{miny}" width="{vw}" height="{vh + 4}" fill="#ffffff"/>']
    for u in parsed.units:
        pts = " ".join(f"{x:.3f},{y:.3f}" for x, y in u.pts)
        out.append(f'<polygon points="{pts}" fill="{u.color}" fill-opacity="{u.alpha:.2f}"/>')
    for u in parsed.units:
        if u.label in ("other", "face"):
            continue
        base = u.label.split("_")[0]
        col = _LABEL_COLORS.get(base, "#000000")
        pts = " ".join(f"{x:.3f},{y:.3f}" for x, y in u.pts)
        out.append(f'<polygon points="{pts}" fill="none" stroke="{col}" stroke-width="0.35"/>')
        out.append(f'<text x="{u.centroid[0]:.2f}" y="{u.centroid[1]:.2f}" font-size="1.6" '
                   f'fill="{col}" stroke="#fff" stroke-width="0.08" text-anchor="middle" '
                   f'font-family="monospace">{u.label}</text>')
    out.append("</svg>")
    path.write_text("\n".join(out), encoding="utf-8")


# ---------------------------------------------------------------------------
# Blender side (runs inside `blender --background --python blender.py -- ...`)
# ---------------------------------------------------------------------------

def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _build_emoji_in_blender(bpy, model: dict, out_path: str):
    import math as _m

    if tuple(bpy.app.version[:2]) < MIN_BLENDER:
        raise RuntimeError(
            f"Blender {bpy.app.version_string} is too old: Grease Pencil v3 needs >= "
            f"{MIN_BLENDER[0]}.{MIN_BLENDER[1]}")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    name = model["name"]

    # 4.3-4.5 expose the new data model as bpy.data.grease_pencils_v3; from 5.0
    # on it is simply bpy.data.grease_pencils.
    gp_collection = getattr(bpy.data, "grease_pencils_v3", None) or bpy.data.grease_pencils
    gp = gp_collection.new(name)
    obj = bpy.data.objects.new(name, gp)
    scene.collection.objects.link(obj)

    # Materials: one per distinct colour+alpha, fill only (these are flat fills).
    materials: Dict[Tuple[str, float], int] = {}

    def material_index(color: str, alpha: float) -> int:
        key = (color, round(alpha, 3))
        if key in materials:
            return materials[key]
        suffix = "" if alpha >= 0.999 else f"_a{int(round(alpha * 100))}"
        mat = bpy.data.materials.new(f"emoji_{color[1:]}{suffix}")
        bpy.data.materials.create_gpencil_data(mat)
        r, g, b = (_srgb_to_linear(v / 255) for v in hex_rgb(color))
        style = mat.grease_pencil
        style.fill_color = (r, g, b, alpha)
        style.color = (r, g, b, alpha)
        # Blender 4.3-4.5 switch fills on via these material flags. In 5.x they
        # are deprecated no-ops (fills come from the per-stroke `fill_id`
        # attribute set below), so silence the deprecation warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            style.show_fill = True
            style.show_stroke = False
        gp.materials.append(mat)
        materials[key] = len(gp.materials) - 1
        return materials[key]

    for layer_def in model["layers"]:
        layer = gp.layers.new(layer_def["name"])
        origin = tuple(layer_def["origin"])
        if hasattr(layer, "translation"):
            layer.translation = origin
        else:  # Blender 4.3 naming
            layer.location = origin
        drawing = layer.frames.new(1).drawing
        strokes = layer_def["strokes"]
        drawing.add_strokes([len(s["points"]) for s in strokes])
        for gp_stroke, sd in zip(drawing.strokes, strokes):
            gp_stroke.cyclic = True
            gp_stroke.material_index = material_index(sd["color"], sd["alpha"])
            for pt, co in zip(gp_stroke.points, sd["points"]):
                pt.position = co
                pt.radius = 0.002
                pt.opacity = 1.0

        # Blender 5.x only fills a stroke that has a non-zero `fill_id`
        # (0 means "plain line"), and hides the outline via `hide_stroke`.
        # Every shape here is a pure fill, so give each stroke its own id.
        n = len(strokes)
        for attr_name, attr_type, values in (
            ("fill_id", "INT", list(range(1, n + 1))),
            ("hide_stroke", "BOOLEAN", [True] * n),
        ):
            try:
                attr = drawing.attributes.get(attr_name) or drawing.attributes.new(attr_name, attr_type, "CURVE")
                attr.data.foreach_set("value", values)
            except Exception as exc:  # older 4.x builds: the material flags above do the job
                print(f"[blender.py] note: could not set {attr_name}: {exc}", flush=True)

    # Keep the face map with the data so animation scripts can find the parts.
    facemap_json = json.dumps(model["facemap"])
    obj["emoji_code"] = model["code"]
    obj["face_map"] = facemap_json
    gp["face_map"] = facemap_json

    # Front-view orthographic camera + colours that match the SVG hex values.
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = model["size"] * 1.15
    cam = bpy.data.objects.new("Camera", cam_data)
    cam.location = (0.0, -10.0, 0.0)
    cam.rotation_euler = (_m.radians(90), 0.0, 0.0)
    scene.collection.objects.link(cam)
    scene.camera = cam
    scene.render.resolution_x = scene.render.resolution_y = 512
    scene.render.film_transparent = True
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    bpy.ops.wm.save_as_mainfile(filepath=out_path, compress=True)


def blender_main(argv: List[str]):
    parser = argparse.ArgumentParser(prog="blender.py (blender side)")
    parser.add_argument("--blender-side", action="store_true")
    parser.add_argument("--job", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args(argv)

    import bpy  # only available inside Blender

    job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    out_dir = Path(job["output"])
    results = []
    models = job["emojis"]
    for i, model in enumerate(models, 1):
        out_path = str(out_dir / f"{model['name']}.blend")
        try:
            _build_emoji_in_blender(bpy, model, out_path)
            results.append({"name": model["name"], "ok": True, "blend": out_path})
            print(f"[blender.py] {i}/{len(models)} {model['name']} ok", flush=True)
        except Exception as exc:  # keep going so one bad emoji doesn't sink the batch
            import traceback
            traceback.print_exc()
            results.append({"name": model["name"], "ok": False, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[blender.py] {i}/{len(models)} {model['name']} FAILED", flush=True)
    Path(args.result).write_text(json.dumps(results), encoding="utf-8")


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------

def find_blender(explicit: Optional[str]) -> Optional[str]:
    candidates: List[str] = []
    if explicit:
        candidates.append(explicit)
    if os.environ.get("BLENDER"):
        candidates.append(os.environ["BLENDER"])
    which = shutil.which("blender")
    if which:
        candidates.append(which)
    candidates += ["/snap/bin/blender", "/usr/bin/blender", "/usr/local/bin/blender"]
    home = str(Path.home())
    for pattern in ("/opt/blender*/blender", f"{home}/blender*/blender",
                    f"{home}/Downloads/blender*/blender", f"{home}/Applications/blender*/blender"):
        candidates += sorted(glob.glob(pattern), reverse=True)
    for c in candidates:
        resolved = shutil.which(c) or (c if os.path.isfile(c) and os.access(c, os.X_OK) else None)
        if resolved:
            return resolved
    return None


def blender_version(exe: str) -> Optional[Tuple[int, int, int]]:
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=120).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"Blender\s+(\d+)\.(\d+)(?:\.(\d+))?", out)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)


INSTALL_HINT = """\
Could not find Blender >= 4.3.

On Ubuntu you can install a current version with:
    sudo snap install blender --classic
or download the Linux tarball from https://www.blender.org/download/ and either
put `blender` on your PATH, set $BLENDER, or pass --blender /path/to/blender.

(The `blender` package in Ubuntu's apt repositories is usually too old: it
predates the new Grease Pencil data model this script needs.)"""


def load_emoji_tables():
    by_short: Dict[str, str] = {}
    short_of: Dict[str, str] = {}
    data = json.loads(EMOJI_JSON.read_text(encoding="utf-8"))
    for e in data["emojis"]:
        if not e.get("unicode"):
            continue
        code = "-".join(cp.lower() for cp in e["unicode"].split(" "))
        bare = (e.get("shortname") or "").replace(":", "").lower()
        if bare:
            by_short.setdefault(bare, code)
            short_of.setdefault(code, bare)
            short_of.setdefault(code.replace("-fe0f", ""), bare)
    return by_short, short_of


def resolve_codes(spec: str, by_short: Dict[str, str]) -> List[str]:
    if spec.strip().lower() == "all":
        return [p.stem for p in sorted(SVG_DIR.glob("*.svg"))]
    codes: List[str] = []
    for tok in (t.strip() for t in spec.split(",")):
        if not tok:
            continue
        if re.fullmatch(r"[0-9a-f]+(-[0-9a-f]+)*", tok, re.I):
            code = tok.lower()
        else:
            code = by_short.get(tok.replace(":", "").lower())
            if code is None:
                raise SystemExit(f'Unknown emoji shortname: "{tok}"')
        if not (SVG_DIR / f"{code}.svg").exists():
            stripped = code.replace("-fe0f", "")
            if (SVG_DIR / f"{stripped}.svg").exists():
                code = stripped
            else:
                raise SystemExit(f'No SVG for "{tok}" (looked for data/svg/{code}.svg)')
        if code not in codes:
            codes.append(code)
    return codes


def run_blender(exe: str, job_path: Path, result_path: Path, verbose: bool) -> int:
    cmd = [exe, "--background", "--factory-startup", "-noaudio",
           "--python-exit-code", "1",
           "--python", str(Path(__file__).resolve()),
           "--", "--blender-side", "--job", str(job_path), "--result", str(result_path)]
    tail: List[str] = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        tail.append(line)
        del tail[:-60]
        if verbose or line.startswith("[blender.py]"):
            print(line, flush=True)
    code = proc.wait()
    if code != 0 and not verbose:
        print("\n--- last Blender output ---", file=sys.stderr)
        print("\n".join(tail), file=sys.stderr)
    return code


def host_main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Export Twemoji SVGs to Blender Grease Pencil with face-part tagging.")
    ap.add_argument("--emojis", required=True,
                    help="comma-separated shortnames (joy), unicode codes (1f602), a mix, or 'all'")
    ap.add_argument("--output", default="./output/blender", help="output directory (default: ./output/blender)")
    ap.add_argument("--naming", choices=["unicode", "shortname"], default="unicode")
    ap.add_argument("--blender", help="path to the Blender executable (default: $BLENDER, then PATH, then common locations)")
    ap.add_argument("--size", type=float, default=2.0, help="emoji size in Blender units (default: 2.0)")
    ap.add_argument("--curve-segments", type=int, default=8,
                    help="line segments per curve (default: 8, same as the TS CLI)")
    ap.add_argument("--facemap-only", action="store_true",
                    help="only write the face-map JSON (and debug SVG); do not run Blender")
    ap.add_argument("--debug-svg", action="store_true", help="also write <name>.facemap.svg showing the labelled shapes")
    ap.add_argument("--verbose", action="store_true", help="show all Blender output")
    args = ap.parse_args(argv)

    by_short, short_of = load_emoji_tables()
    codes = resolve_codes(args.emojis, by_short)
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    models, failures = [], []
    n_face = 0
    for code in codes:
        name = short_of.get(code, code) if args.naming == "shortname" else code
        try:
            model, parsed = build_model(code, name, SVG_DIR / f"{code}.svg", args.size, args.curve_segments)
        except Exception as exc:
            failures.append((code, f"{type(exc).__name__}: {exc}"))
            continue
        fm = model["facemap"]
        n_face += fm["is_face"]
        (out_dir / f"{name}.facemap.json").write_text(json.dumps(fm, indent=2), encoding="utf-8")
        if args.debug_svg:
            write_debug_svg(out_dir / f"{name}.facemap.svg", parsed)
        models.append(model)
        if len(codes) <= 20:
            parts = ", ".join(fm["parts"]) if fm["is_face"] else "(not a face)"
            print(f"{code:<24} {parts}")
            for w in fm["warnings"]:
                print(f"    warning: {w}")

    print(f"Parsed {len(models)}/{len(codes)} emoji, {n_face} detected as faces.")
    for code, err in failures:
        print(f"  FAILED {code}: {err}", file=sys.stderr)

    if args.facemap_only or not models:
        print(f"Wrote face maps to {out_dir}")
        return 1 if failures else 0

    exe = find_blender(args.blender)
    if exe is None:
        print(INSTALL_HINT, file=sys.stderr)
        print(f"\nFace maps were still written to {out_dir}", file=sys.stderr)
        return 2
    ver = blender_version(exe)
    if ver is None:
        print(f"Could not read the version of {exe}; is it really Blender?", file=sys.stderr)
        return 2
    if ver[:2] < MIN_BLENDER:
        print(f"{exe} is Blender {ver[0]}.{ver[1]}.{ver[2]}, but >= {MIN_BLENDER[0]}.{MIN_BLENDER[1]} is required.\n\n"
              f"{INSTALL_HINT}", file=sys.stderr)
        return 2
    print(f"Using Blender {ver[0]}.{ver[1]}.{ver[2]} ({exe})")

    with tempfile.TemporaryDirectory(prefix="emoji-gp-") as tmp:
        job_path, result_path = Path(tmp) / "job.json", Path(tmp) / "result.json"
        job_path.write_text(json.dumps({"output": str(out_dir), "emojis": models}),
                            encoding="utf-8")
        code = run_blender(exe, job_path, result_path, args.verbose)
        results = json.loads(result_path.read_text()) if result_path.exists() else []

    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    for r in bad:
        print(f"  FAILED {r['name']}: {r['error']}", file=sys.stderr)
    if code != 0 and not results:
        print("Blender exited with an error before producing any output.", file=sys.stderr)
    else:
        print(f"Wrote {len(ok)} .blend file(s) to {out_dir}")
    return 0 if code == 0 and not bad and not failures else 1


def main() -> int:
    if "--blender-side" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
        blender_main(argv)
        return 0
    return host_main()


if __name__ == "__main__":
    sys.exit(main())
