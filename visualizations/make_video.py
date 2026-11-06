"""Render the Qismet machine-learning explainer video.

Topics: the 784 -> 16 -> 10 network used in Qismet, how it learns, pruning,
and the Lottery Ticket Hypothesis. Every chart and image uses real numbers from
this repository (see prepare_assets.py and lottery_ticket_experiment.py).

    python make_video.py                    # full video -> qismet_ml_explained.mp4
    python make_video.py --scale 2          # 4K (3840x2160)
    python make_video.py --voice Daniel     # any voice from `say -v '?'`
    python make_video.py --still 6:9.5      # PNG of scene 6 at 9.5 s (for checking)

Needs skia-python, numpy and imageio-ffmpeg; narration uses the macOS `say` command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import wave
from multiprocessing import Pool
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import skia

HERE = Path(__file__).resolve().parent
BUILD = HERE / 'build'
W, H = 1920, 1080
FPS = 60
SR = 44100

A = np.load(HERE / 'video_assets.npz')
M = json.loads((HERE / 'video_assets.json').read_text())
CLASSES = M['classes']
CLASS_SHORT = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# ----------------------------------------------------------------------------
# Palette and drawing helpers
# ----------------------------------------------------------------------------
BG_TOP, BG_BOT = (9, 13, 24), (17, 24, 41)
PANEL = (22, 30, 50)
INK = (238, 242, 248)
MUTED = (143, 155, 180)
FAINT = (64, 76, 102)
ORANGE = (255, 158, 66)
BLUE = (82, 170, 255)
GREEN = (60, 212, 140)
CORAL = (255, 104, 104)
GOLD = (255, 199, 84)
VIOLET = (168, 140, 255)


def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def eio(x):
    x = clamp(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def eout(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def eback(x):
    x = clamp(x)
    c1 = 1.70158
    return 1 + (c1 + 1) * (x - 1) ** 3 + c1 * (x - 1) ** 2


def lerp(a, b, t):
    return a + (b - a) * t


def mix(c1, c2, t):
    return tuple(lerp(a, b, t) for a, b in zip(c1, c2))


def col(rgb, a=1.0):
    return skia.Color(int(rgb[0]), int(rgb[1]), int(rgb[2]), int(clamp(a) * 255))


def fill(rgb, a=1.0, blur=0):
    p = skia.Paint(AntiAlias=True)
    p.setColor(col(rgb, a))
    if blur:
        p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, blur))
    return p


def stroke(rgb, a=1.0, w=2.0, dash=None, blur=0):
    p = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=w)
    p.setColor(col(rgb, a))
    p.setStrokeCap(skia.Paint.kRound_Cap)
    p.setStrokeJoin(skia.Paint.kRound_Join)
    if dash:
        p.setPathEffect(skia.DashPathEffect.Make(dash, 0))
    if blur:
        p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, blur))
    return p


_FM = skia.FontMgr.RefDefault()
_TF, _FONTS = {}, {}


def font(size, weight=500, family='Avenir Next'):
    key = (round(size, 1), weight, family)
    if key not in _FONTS:
        tk = (family, weight)
        if tk not in _TF:
            _TF[tk] = _FM.matchFamilyStyle(family, skia.FontStyle(weight, 5, skia.FontStyle.kUpright_Slant))
        f = skia.Font(_TF[tk], size)
        f.setSubpixel(True)
        f.setEdging(skia.Font.Edging.kAntiAlias)
        _FONTS[key] = f
    return _FONTS[key]


def text_width(s, size, weight=500, family='Avenir Next'):
    return font(size, weight, family).measureText(s)


def text(c, s, x, y, size=32, rgb=INK, a=1.0, weight=500, align='l', family='Avenir Next', vcenter=False):
    """Draw a string. y is the baseline, or the visual centre if vcenter."""
    if a <= 0.003:
        return 0
    f = font(size, weight, family)
    w = f.measureText(s)
    if align == 'c':
        x -= w / 2
    elif align == 'r':
        x -= w
    if vcenter:
        y += size * 0.36
    c.drawString(s, x, y, f, fill(rgb, a))
    return w


def text_sub(c, base, sub, x, y, size, rgb=INK, a=1.0, weight=500):
    """'x' with a small subscript, e.g. x_1. Returns total width."""
    w = text(c, base, x, y, size, rgb, a, weight)
    w += text(c, sub, x + w + 1, y + size * 0.22, size * 0.62, rgb, a, weight)
    return w + 2


def rrect(c, x, y, w, h, r, paint):
    c.drawRoundRect(skia.Rect.MakeXYWH(x, y, w, h), r, r, paint)


def arrow(c, x1, y1, x2, y2, rgb=MUTED, a=1.0, w=3.0, head=14, progress=1.0):
    if progress <= 0 or a <= 0:
        return
    x2p, y2p = lerp(x1, x2, progress), lerp(y1, y2, progress)
    c.drawLine(x1, y1, x2p, y2p, stroke(rgb, a, w))
    ang = math.atan2(y2p - y1, x2p - x1)
    p = skia.Path()
    p.moveTo(x2p + math.cos(ang) * 2, y2p + math.sin(ang) * 2)
    p.lineTo(x2p - head * math.cos(ang - 0.45), y2p - head * math.sin(ang - 0.45))
    p.lineTo(x2p - head * math.cos(ang + 0.45), y2p - head * math.sin(ang + 0.45))
    p.close()
    c.drawPath(p, fill(rgb, a))


def check_mark(c, x, y, s, rgb=GREEN, a=1.0, progress=1.0, w=6):
    p = skia.Path()
    pts = [(x - s * 0.5, y), (x - s * 0.15, y + s * 0.35), (x + s * 0.55, y - s * 0.4)]
    p.moveTo(*pts[0])
    if progress < 0.4:
        q = progress / 0.4
        p.lineTo(lerp(pts[0][0], pts[1][0], q), lerp(pts[0][1], pts[1][1], q))
    else:
        q = (progress - 0.4) / 0.6
        p.lineTo(*pts[1])
        p.lineTo(lerp(pts[1][0], pts[2][0], q), lerp(pts[1][1], pts[2][1], q))
    c.drawPath(p, stroke(rgb, a, w))


def cross_mark(c, x, y, s, rgb=CORAL, a=1.0, progress=1.0, w=6):
    q1, q2 = clamp(progress * 2), clamp(progress * 2 - 1)
    c.drawLine(x - s / 2, y - s / 2, lerp(x - s / 2, x + s / 2, q1), lerp(y - s / 2, y + s / 2, q1), stroke(rgb, a, w))
    if q2 > 0:
        c.drawLine(x + s / 2, y - s / 2, lerp(x + s / 2, x - s / 2, q2), lerp(y - s / 2, y + s / 2, q2), stroke(rgb, a, w))


def image_from_rgb(rgb):
    rgba = np.dstack([rgb.astype(np.uint8), np.full(rgb.shape[:2], 255, np.uint8)])
    return skia.Image.fromarray(np.ascontiguousarray(rgba), colorType=skia.ColorType.kRGBA_8888_ColorType)


NEAREST = skia.SamplingOptions(skia.FilterMode.kNearest)
LINEAR = skia.SamplingOptions(skia.FilterMode.kLinear)


def draw_image(c, img, x, y, w, h, a=1.0, radius=0, sampling=NEAREST):
    if a <= 0.003:
        return
    p = skia.Paint(AntiAlias=True)
    p.setAlphaf(clamp(a))
    if radius:
        c.save()
        c.clipRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, y, w, h), radius, radius), True)
    c.drawImageRect(img, skia.Rect.MakeXYWH(x, y, w, h), sampling, p)
    if radius:
        c.restore()


TILE_BG = (20, 27, 45)


def fashion_image(img28, tint=INK, base=TILE_BG):
    v = (img28.astype(float) / 255.0)[..., None]
    rgb = np.array(base)[None, None] * (1 - v) + np.array(tint)[None, None] * v
    return image_from_rgb(rgb)


def weight_image(w784, scale=None):
    w = w784.reshape(28, 28)
    s = scale or np.abs(w).max()
    v = np.clip(w / s, -1, 1)
    base = np.array((14, 19, 33), float)
    pos = np.array(ORANGE, float)
    neg = np.array(BLUE, float)
    m = np.abs(v)[..., None] ** 0.85
    tgt = np.where(v[..., None] >= 0, pos, neg)
    return image_from_rgb(base * (1 - m) + tgt * m)


# ----------------------------------------------------------------------------
# Data used across scenes
# ----------------------------------------------------------------------------
W1, W2 = A['W1'], A['W2']
W1_INIT, W2_INIT = A['W1_init'], A['W2_init']
_rng = np.random.default_rng(7)
_bright = np.flatnonzero(A['grid_images'].reshape(-1, 784).mean(0) / 255 > 0.18)
INPUT_IDX = np.sort(_rng.choice(_bright, 20, replace=False))
W1S, W1S_INIT = W1[:, INPUT_IDX], W1_INIT[:, INPUT_IDX]
W1S_FRESH = _rng.uniform(-1 / 28, 1 / 28, W1S.shape)
W2_FRESH = _rng.uniform(-0.25, 0.25, W2.shape)
SCALE1 = float(np.percentile(np.abs(W1S), 92))
SCALE2 = float(np.percentile(np.abs(W2), 92))
INIT_S1 = float(np.percentile(np.abs(W1S_INIT), 92))
INIT_S2 = float(np.percentile(np.abs(W2_INIT), 92))
ABS1_ALL = np.abs(W1).ravel()
ABS2_ALL = np.abs(W2).ravel()


def magnitude_masks(sparsity):
    if sparsity <= 0:
        return np.ones_like(W1S), np.ones_like(W2)
    t1 = np.quantile(ABS1_ALL, sparsity)
    t2 = np.quantile(ABS2_ALL, sparsity)
    return (np.abs(W1S) >= t1).astype(float), (np.abs(W2) >= t2).astype(float)


M80_1, M80_2 = magnitude_masks(0.8)
EDGE_ORDER1 = _rng.permutation(W1S.size).reshape(W1S.shape) / W1S.size
EDGE_ORDER2 = _rng.permutation(W2.size).reshape(W2.shape) / W2.size

GALLERY = [fashion_image(im) for im in A['gallery_images']]
GRID = [fashion_image(im) for im in A['grid_images']]
NEURON_IMGS = [weight_image(W1[j]) for j in range(16)]
NEURON_INIT_IMGS = [weight_image(W1_INIT[j]) for j in range(16)]


class Net:
    """784-16-10 diagram with 20 representative input pixels."""

    def __init__(self, x0, y0, x1, y1, in_r=7, h_r=12, o_r=11):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.xin, self.xh, self.xo = x0, (x0 + x1) / 2, x1
        h = y1 - y0
        self.yin = np.r_[np.linspace(y0, y0 + h * 0.43, 10), np.linspace(y1 - h * 0.43, y1, 10)]
        self.yell = y0 + h * 0.5
        self.yh = np.linspace(y0 + h * 0.02, y1 - h * 0.02, 16)
        self.yo = np.linspace(y0 + h * 0.13, y1 - h * 0.13, 10)
        self.in_r, self.h_r, self.o_r = in_r, h_r, o_r

    def draw(self, c, w1=W1S, w2=W2, alpha=1.0, e1=1.0, e2=1.0, v1=None, v2=None,
             hid_vis=None, hid_act=None, out_act=None, labels=0.0, in_vis=1.0, hid_show=1.0,
             out_show=1.0, out_highlight=None, edge_w=1.0, in_act=None, s1=None, s2=None):
        if alpha <= 0.003:
            return
        segs = []
        if e1 > 0:
            mag = np.clip(np.abs(w1) / (s1 or SCALE1), 0, 1)
            vis = np.ones_like(mag) if v1 is None else v1
            if hid_vis is not None:
                vis = vis * np.asarray(hid_vis)[:, None]
            for j, i in zip(*np.nonzero(vis > 0.003)):
                a = alpha * e1 * vis[j, i] * (0.07 + 0.93 * mag[j, i] ** 1.1)
                segs.append((mag[j, i], self.xin, self.yin[i], self.xh, self.yh[j],
                             ORANGE if w1[j, i] >= 0 else BLUE, a, (0.8 + 2.0 * mag[j, i]) * edge_w))
        if e2 > 0:
            mag = np.clip(np.abs(w2) / (s2 or SCALE2), 0, 1)
            vis = np.ones_like(mag) if v2 is None else v2
            if hid_vis is not None:
                vis = vis * np.asarray(hid_vis)[None, :]
            for k, j in zip(*np.nonzero(vis > 0.003)):
                a = alpha * e2 * vis[k, j] * (0.07 + 0.93 * mag[k, j] ** 1.1)
                segs.append((mag[k, j], self.xh, self.yh[j], self.xo, self.yo[k],
                             ORANGE if w2[k, j] >= 0 else BLUE, a, (0.8 + 2.0 * mag[k, j]) * edge_w))
        segs.sort(key=lambda s: s[0])
        for _, xa, ya, xb, yb, rgb, a, wd in segs:
            c.drawLine(xa, ya, xb, yb, stroke(rgb, a, wd))
        # input nodes
        if in_vis > 0:
            for i, y in enumerate(self.yin):
                act = 0 if in_act is None else in_act[i]
                c.drawCircle(self.xin, y, self.in_r, fill(mix(PANEL, INK, act), alpha * in_vis))
                c.drawCircle(self.xin, y, self.in_r, stroke(MUTED, alpha * in_vis * 0.9, 1.5))
            for d in (-14, 0, 14):
                c.drawCircle(self.xin, self.yell + d, 2.6, fill(MUTED, alpha * in_vis))
        # hidden nodes
        if hid_show > 0:
            for j, y in enumerate(self.yh):
                hv = 1.0 if hid_vis is None else hid_vis[j]
                act = 0 if hid_act is None else hid_act[j]
                a = alpha * hid_show * (0.18 + 0.82 * hv)
                if act > 0.02:
                    c.drawCircle(self.xh, y, self.h_r + 7, fill(ORANGE, a * act * 0.55, blur=8))
                c.drawCircle(self.xh, y, self.h_r, fill(mix(PANEL, ORANGE, act), a))
                c.drawCircle(self.xh, y, self.h_r, stroke(INK if hv > 0.5 else FAINT, a, 2))
        # output nodes
        if out_show > 0:
            for k, y in enumerate(self.yo):
                act = 0 if out_act is None else out_act[k]
                hl = out_highlight is not None and out_highlight[0] == k
                g = out_highlight[1] if hl else 0
                if g > 0:
                    c.drawCircle(self.xo, y, self.o_r + 10, fill(GREEN, alpha * out_show * g * 0.6, blur=10))
                c.drawCircle(self.xo, y, self.o_r, fill(mix(mix(PANEL, INK, act), GREEN, g), alpha * out_show))
                c.drawCircle(self.xo, y, self.o_r, stroke(INK, alpha * out_show, 2))
                if labels > 0:
                    text(c, CLASS_SHORT[k], self.xo + self.o_r + 14, y, 21, mix(MUTED, INK, g),
                         alpha * labels * out_show, 500, vcenter=True)


def section(c, num, title, a):
    text(c, num, 96, 82, 24, ORANGE, a, 700)
    text(c, title.upper(), 96 + text_width(num, 24, 700) + 16, 82, 24, MUTED, a, 600)


def mini_net_glyph(c, x, y, w, h, edges, rgb, a, node_rgb=None, lw=2.2):
    """Small 4-5-3 network glyph used on tickets and recap cards."""
    cols = [np.linspace(y + h * 0.18, y + h * 0.82, n) for n in (4, 5, 3)]
    xs = [x + w * 0.12, x + w * 0.5, x + w * 0.88]
    for (l, i, j) in edges:
        c.drawLine(xs[l], cols[l][i], xs[l + 1], cols[l + 1][j], stroke(rgb, a, lw))
    for l in range(3):
        for yy in cols[l]:
            c.drawCircle(xs[l], yy, 5, fill(node_rgb or INK, a))


ALL_GLYPH_EDGES = [(0, i, j) for i in range(4) for j in range(5)] + [(1, i, j) for i in range(5) for j in range(3)]


# ----------------------------------------------------------------------------
# Scene context
# ----------------------------------------------------------------------------
class Ctx:
    def __init__(self, t, dur, cues, ends):
        self.t, self.dur, self.cues, self.ends = t, dur, cues, ends

    def at(self, i, d=0.8, delay=0.0, ease=eio):
        """Eased 0->1 progress starting at narration line i (+delay)."""
        return ease((self.t - self.cues[i] - delay) / d)

    def since(self, i, delay=0.0):
        return self.t - self.cues[i] - delay

    def span(self, i, extra=0.0):
        """Linear 0->1 progress through narration line i."""
        return clamp((self.t - self.cues[i]) / max(1e-6, self.ends[i] - self.cues[i] + extra))

    def word(self, i, frac, d=0.6):
        """Progress starting at a fraction of the way through line i."""
        start = lerp(self.cues[i], self.ends[i], frac)
        return eio((self.t - start) / d)

    def fade_out(self, d=0.6):
        return smooth((self.dur - self.t) / d)


# ----------------------------------------------------------------------------
# Scenes
# ----------------------------------------------------------------------------
def scene_title(c, x):
    net = Net(1010, 210, 1700, 870, in_r=6, h_r=11, o_r=10)
    m1, m2 = magnitude_masks(0.85)
    p = x.span(1, extra=1.0)
    prune = smooth((p - 0.15) / 0.7)
    v1 = 1 - prune * (1 - m1)
    v2 = 1 - prune * (1 - m2)
    a_net = x.at(0, 1.6, 0.0) * 0.9
    net.draw(c, alpha=a_net, v1=v1, v2=v2, e1=0.9, e2=0.9, edge_w=1.1)
    a = x.at(0, 1.0, 0.2)
    rise = (1 - eout(x.since(0, 0.2) / 1.2)) * 30
    text(c, 'Small networks,', 150, 430 + rise, 92, INK, a, 700)
    a2 = x.at(0, 1.0, 0.6)
    text(c, 'lucky tickets.', 150, 540 + (1 - eout(x.since(0, 0.6) / 1.2)) * 30, 92, GOLD, a2, 700)
    a3 = x.at(0, 1.0, 1.2)
    text(c, 'Neural networks  ·  Pruning  ·  The Lottery Ticket Hypothesis', 154, 618, 30, MUTED, a3, 500)
    c.drawLine(154, 660, 154 + 120 * x.at(0, 1.0, 1.4), 660, stroke(ORANGE, a3, 4))
    text(c, 'The machine learning behind Qismet', 154, 708, 26, MUTED, a3 * 0.8, 500)
    if prune > 0.02:
        text(c, f'{int(round(85 * prune))}% of connections removed', 1355, 935, 24, GREEN, prune * a_net, 600, 'c')


def scene_task(c, x):
    section(c, '01', 'The task', x.at(0, 0.8))
    # Phase A: grid of real training images
    out_a = 1 - x.at(1, 0.7)
    if out_a > 0:
        cols, rows, tile, gap = 16, 6, 92, 12
        gx = (W - (cols * (tile + gap) - gap)) / 2
        gy = 210
        for r in range(rows):
            for q in range(cols):
                i = r * cols + q
                pop = eback(x.since(0, 0.15 + (r + q) * 0.045) / 0.55)
                if pop <= 0:
                    continue
                s = tile * pop
                cx, cy = gx + q * (tile + gap) + tile / 2, gy + r * (tile + gap) + tile / 2
                draw_image(c, GRID[i], cx - s / 2, cy - s / 2, s, s, out_a * clamp(pop), 10)
        text(c, '70,000 labelled photos of clothing', W / 2, 900, 34, INK, x.at(0, 0.8, 1.6) * out_a, 600, 'c')
    # Phase B: ten classes
    a_b = x.at(1, 0.8, 0.3)
    focus = x.at(2, 1.0)
    if a_b > 0:
        tile, gap = 148, 24
        gx = (W - (10 * (tile + gap) - gap)) / 2
        for k in range(10):
            hl = x.word(1, 0.30 + 0.62 * k / 10, 0.35)
            pop = eout(x.since(1, 0.3 + k * 0.06) / 0.5)
            tx = gx + k * (tile + gap)
            ty = 360 + (1 - pop) * 40
            dim = 1 - focus * (0.0 if k == 7 else 1.0)
            if k == 7:
                # the chosen sneaker travels to the left
                tx = lerp(tx, 300, focus)
                ty = lerp(ty, 380, focus)
            size = tile if k != 7 else lerp(tile, 300, focus)
            a = a_b * pop * dim
            if a <= 0.003:
                continue
            rrect(c, tx - 6, ty - 6, size + 12, size + 12, 16, stroke(GOLD, a * hl * (1 - focus), 3))
            draw_image(c, GALLERY[k], tx, ty, size, size, a, 12)
            if focus < 0.5:
                text(c, CLASS_SHORT[k], tx + size / 2, ty + size + 42, 24, mix(MUTED, INK, hl), a * (1 - focus * 2), 600, 'c')
    # Phase C: image -> network -> answer
    if focus > 0:
        a = focus
        arrow(c, 640, 530, 800, 530, MUTED, a, 4, progress=x.at(2, 0.6, 0.5))
        bx = x.at(2, 0.6, 0.8)
        rrect(c, 830, 430, 340, 200, 26, fill(PANEL, a * bx))
        rrect(c, 830, 430, 340, 200, 26, stroke(VIOLET, a * bx, 2.5))
        text(c, 'Neural network', 1000, 530, 34, INK, a * bx, 600, 'c', vcenter=True)
        arrow(c, 1200, 530, 1360, 530, MUTED, a, 4, progress=x.at(2, 0.6, 1.3))
        ans = x.at(2, 0.6, 2.4)
        text(c, '?', 1480, 530, 90, MUTED, a * x.at(2, 0.4, 1.6) * (1 - ans), 700, 'c', vcenter=True)
        text(c, 'Sneaker', 1395, 530, 64, GREEN, a * ans, 700, vcenter=True)
        text(c, 'never seen during training', 300 + 150, 760, 24, MUTED, a * x.at(2, 0.6, 0.8), 500, 'c')


def scene_pixels(c, x):
    section(c, '02', 'Images are numbers', x.at(0, 0.8))
    img = A['gallery_images'][9].astype(float) / 255
    S = 20
    ix, iy = 150, 190
    a0 = x.at(0, 0.8, 0.2)
    draw_image(c, GALLERY[9], ix, iy, 28 * S, 28 * S, a0, 8)
    grid_a = x.at(0, 1.0, 1.4) * 0.45
    if grid_a > 0:
        p = stroke((0, 0, 0), grid_a, 1)
        for k in range(1, 28):
            c.drawLine(ix + k * S, iy, ix + k * S, iy + 28 * S, p)
            c.drawLine(ix, iy + k * S, ix + 28 * S, iy + k * S, p)
    text(c, '28 × 28 = 784 pixels', ix + 14 * S, iy + 28 * S + 56, 32, INK, x.at(1, 0.8, 0.8), 600, 'c')
    tape_phase = x.at(2, 0.6)
    # magnified patch with numbers
    a1 = x.at(1, 0.8) * (1 - tape_phase)
    if a1 > 0:
        r0, c0, n = 9, 10, 6
        c.drawRect(skia.Rect.MakeXYWH(ix + c0 * S, iy + r0 * S, n * S, n * S), stroke(GOLD, a1, 3))
        cell = 96
        px, py = 900, 200
        c.drawLine(ix + (c0 + n) * S, iy + r0 * S, px, py, stroke(GOLD, a1 * 0.5, 1.5))
        c.drawLine(ix + (c0 + n) * S, iy + (r0 + n) * S, px, py + n * cell, stroke(GOLD, a1 * 0.5, 1.5))
        for r in range(n):
            for q in range(n):
                v = img[r0 + r, c0 + q]
                pop = eout(x.since(1, 0.3 + (r * n + q) * 0.025) / 0.4)
                bg = mix(TILE_BG, INK, v)
                rrect(c, px + q * cell + 3, py + r * cell + 3, cell - 6, cell - 6, 8, fill(bg, a1 * pop))
                text(c, f'{v:.2f}', px + q * cell + cell / 2, py + r * cell + cell / 2, 26,
                     (12, 16, 28) if v > 0.55 else INK, a1 * pop, 600, 'c', vcenter=True)
        # brightness legend
        lx, ly, lw = 900, 820, n * cell
        sh = skia.GradientShader.MakeLinear([skia.Point(lx, 0), skia.Point(lx + lw, 0)], [col(TILE_BG, a1), col(INK, a1)])
        p = skia.Paint(AntiAlias=True)
        p.setShader(sh)
        rrect(c, lx, ly, lw, 22, 11, p)
        text(c, '0  black', lx, ly + 58, 24, MUTED, a1, 500)
        text(c, 'white  1', lx + lw, ly + 58, 24, MUTED, a1, 500, 'r')
    # unrolling into a tape of 784 numbers
    if tape_phase > 0:
        a = tape_phase
        prog = smooth(clamp(x.since(2, 0.6) / max(2.0, x.ends[2] - x.cues[2] - 0.4)))
        k = int(prog * 784)
        cell, ncell = 42, 36
        tx, ty = 900 - 0 * cell, 470
        rr = min(27, k // 28)
        c.drawRect(skia.Rect.MakeXYWH(ix, iy + rr * S, 28 * S, S), stroke(GOLD, a * (1 if k < 784 else 0.0), 3))
        start = max(0, k - ncell)
        text(c, 'The input: one list of 784 numbers', 900, 400, 32, INK, a, 600)
        for slot in range(ncell):
            idx = start + slot
            if idx >= k:
                break
            v = img.ravel()[idx]
            cx = tx + slot * (cell - 20) if False else tx + (slot % 18) * cell
            cy = ty + (slot // 18) * 100
            rrect(c, cx, cy, cell - 6, 56, 6, fill(mix(TILE_BG, INK, v), a))
            text(c, f'{v:.1f}', cx + (cell - 6) / 2, cy + 28, 15, (12, 16, 28) if v > 0.55 else MUTED, a, 600, 'c', vcenter=True)
            if (idx + 1) % 28 == 1 or slot in (0,):
                text_sub(c, 'x', str(idx + 1), cx + 4, cy + 84, 18, MUTED, a * 0.9, 500)
        text(c, f'{k} / 784', 900, 740, 44, GOLD, a, 700)
        text(c, 'read row by row', 900 + text_width(f'{k} / 784', 44, 700) + 24, 740, 26, MUTED, a, 500)


def scene_network(c, x):
    section(c, '03', 'The network', x.at(0, 0.8))
    net = Net(470, 170, 1300, 880)
    panel = x.at(2, 0.8) * (1 - x.at(3, 0.8))
    dim = 1 - 0.85 * panel
    in_vis = x.at(0, 0.8, 0.3)
    hid = x.at(0, 0.8, 1.8)
    out = x.at(0, 0.8, 3.3)
    sweep1 = clamp(x.since(1, 0.0) / 2.4)
    sweep2 = clamp(x.since(1, 2.2) / 1.6)
    v1 = np.clip((sweep1 * 1.25 - EDGE_ORDER1) * 6, 0, 1)
    v2 = np.clip((sweep2 * 1.25 - EDGE_ORDER2) * 6, 0, 1)
    final = x.at(3, 1.0, 0.3)
    hl = (7, final) if final > 0 else None
    oact = None
    if final > 0:
        oact = A['gallery_probs'][7] * final
    net.draw(c, alpha=dim, v1=v1, v2=v2, in_vis=in_vis, hid_show=hid, out_show=out, labels=out,
             out_highlight=hl, out_act=oact)
    text(c, '784 inputs', net.xin, 140, 26, INK, in_vis * dim, 600, 'c')
    text(c, '(20 shown)', net.xin, 916, 20, MUTED, in_vis * dim, 500, 'c')
    text(c, '16 hidden neurons', net.xh, 140, 26, INK, hid * dim, 600, 'c')
    text(c, '10 outputs', net.xo, 140, 26, INK, out * dim, 600, 'c')
    if x.at(1, 0.6, 1.0) > 0:
        a = x.at(1, 0.6, 3.6) * dim
        text(c, 'each line = one weight', 1580, 470, 26, INK, a, 600, 'c')
        c.drawLine(1500, 500, 1560, 500, stroke(ORANGE, a, 4))
        text(c, 'positive', 1575, 508, 22, MUTED, a, 500)
        c.drawLine(1500, 540, 1560, 540, stroke(BLUE, a, 4))
        text(c, 'negative', 1575, 548, 22, MUTED, a, 500)
    # single neuron panel
    if panel > 0:
        a = panel
        rrect(c, 160, 190, 1600, 700, 32, fill(PANEL, a * 0.96))
        rrect(c, 160, 190, 1600, 700, 32, stroke(FAINT, a, 1.5))
        text(c, 'Inside one hidden neuron', 220, 260, 32, INK, a, 700)
        ys = [370, 450, 530, 690]
        xs_in = 320
        ws = ['0.42', '−0.17', '0.08', '0.31']
        for i, (yy, wv) in enumerate(zip(ys, ws)):
            ap = a * x.at(2, 0.4, 0.2 + i * 0.12)
            c.drawCircle(xs_in, yy, 22, fill(mix(TILE_BG, INK, [0.8, 0.3, 0.6, 0.9][i]), ap))
            sub = ['1', '2', '3', '784'][i]
            text_sub(c, 'x', sub, xs_in - 70 - (14 if i == 3 else 0), yy + 10, 28, MUTED, ap)
            wc = BLUE if wv.startswith('−') else ORANGE
            c.drawLine(xs_in + 24, yy, 760, 530, stroke(wc, ap * 0.9, 3))
            text_sub(c, 'w', sub, lerp(xs_in, 760, 0.42) - 20, lerp(yy, 530, 0.42) - 14, 24, wc, ap, 600)
        for d in (-14, 0, 14):
            c.drawCircle(xs_in, 610 + d, 3, fill(MUTED, a))
        sa = a * x.at(2, 0.5, 1.2)
        c.drawCircle(820, 530, 64, fill((30, 40, 66), sa))
        c.drawCircle(820, 530, 64, stroke(VIOLET, sa, 3))
        text(c, 'Σ', 820, 520, 52, INK, sa, 600, 'c', vcenter=True)
        text(c, '+ b', 820, 572, 22, MUTED, sa, 600, 'c')
        text(c, 'add up', 820, 640, 22, MUTED, sa, 500, 'c')
        arrow(c, 890, 530, 1040, 530, MUTED, sa, 3, progress=x.at(2, 0.5, 1.8))
        # ReLU plot
        ra = a * x.at(2, 0.6, 2.4)
        gx, gy, gw, gh = 1070, 380, 320, 300
        rrect(c, gx, gy, gw, gh, 18, fill((16, 22, 38), ra))
        ox, oy = gx + gw / 2, gy + gh * 0.72
        c.drawLine(gx + 20, oy, gx + gw - 20, oy, stroke(FAINT, ra, 2))
        c.drawLine(ox, gy + 20, ox, gy + gh - 20, stroke(FAINT, ra, 2))
        path = skia.Path()
        path.moveTo(gx + 24, oy)
        path.lineTo(ox, oy)
        path.lineTo(gx + gw - 30, gy + 40)
        c.drawPath(path, stroke(ORANGE, ra, 5))
        text(c, 'ReLU', gx + 24, gy + 44, 28, INK, ra, 700)
        text(c, 'negative total becomes 0', gx + gw / 2, gy + gh + 44, 22, MUTED, ra, 500, 'c', family='Helvetica Neue')
        text(c, 'positive total passes through', gx + gw / 2, gy + gh + 76, 22, MUTED, ra, 500, 'c', family='Helvetica Neue')
        z = math.sin(x.t * 1.6) * 0.9
        px = ox + z * (gw / 2 - 40)
        py = oy - max(0, z) * (gw / 2 - 40) * (oy - gy - 40) / (gw / 2 - 30)
        c.drawCircle(px, py, 9, fill(GOLD, ra))
        arrow(c, 1410, 530, 1560, 530, MUTED, ra, 3, progress=x.at(2, 0.5, 3.0))
        oa = a * x.at(2, 0.5, 3.4)
        act = max(0, z)
        c.drawCircle(1620, 530, 40, fill(ORANGE, oa * act * 0.6, blur=12))
        c.drawCircle(1620, 530, 34, fill(mix(PANEL, ORANGE, act), oa))
        c.drawCircle(1620, 530, 34, stroke(INK, oa, 2.5))
        text(c, 'output', 1620, 610, 22, MUTED, oa, 500, 'c')
        # formula
        fa = a * x.at(2, 0.6, 1.0)
        fx, fy = 300, 820
        fx += text(c, 'output = ReLU(', fx, fy, 30, INK, fa, 600)
        for i, sub in enumerate(['1', '2']):
            fx += text_sub(c, 'w', sub, fx + 4, fy, 30, ORANGE, fa, 600)
            fx += text_sub(c, 'x', sub, fx, fy, 30, INK, fa, 600)
            fx += text(c, ' + ', fx, fy, 30, INK, fa, 600)
        fx += text(c, '…', fx, fy, 30, INK, fa, 600)
        fx += text(c, ' + ', fx, fy, 30, INK, fa, 600)
        fx += text_sub(c, 'w', '784', fx, fy, 30, ORANGE, fa, 600)
        fx += text_sub(c, 'x', '784', fx, fy, 30, INK, fa, 600)
        fx += text(c, ' + b )', fx, fy, 30, INK, fa, 600)
    # parameter counter
    ca = x.at(3, 0.6, 1.5)
    if ca > 0:
        n = int(12730 * eout(x.since(3, 1.5) / 2.2))
        text(c, f'{n:,}', 1650, 860, 64, GOLD, ca, 700, 'c')
        text(c, 'adjustable numbers', 1650, 900, 24, MUTED, ca, 500, 'c')
        text(c, '784×16 + 16 + 16×10 + 10', 1650, 935, 20, FAINT, ca, 500, 'c')


def scene_training(c, x):
    section(c, '04', 'Learning', x.at(0, 0.8))
    net = Net(230, 200, 860, 860, in_r=6, h_r=11, o_r=10)
    curve = [M['init_val_accuracy']] + M['dense_val_curve']
    span_dur = max(3.0, x.ends[1] - x.cues[1] + 0.4)
    ep = 20 * smooth(clamp((x.t - x.cues[1] - 0.5) / span_dur))
    # weight morph follows the real learning curve shape
    acc_now = np.interp(ep, np.arange(21), curve)
    f = clamp((acc_now - curve[0]) / (curve[-1] - curve[0]))
    w1 = W1S_INIT + (W1S - W1S_INIT) * f
    w2 = W2_INIT + (W2 - W2_INIT) * f
    rnd = np.random.default_rng(int(x.t * 6))
    noisy = rnd.dirichlet(np.ones(10) * 1.5)
    out = noisy * (1 - f) + A['gallery_probs'][7] * f
    a = x.at(0, 0.8, 0.2)
    net.draw(c, w1=w1, w2=w2, alpha=a, labels=1.0, out_act=out * 0.9,
             out_highlight=(7, x.at(2, 0.6, 0.3)) if x.t > x.cues[2] else None)
    # stream of training images into the inputs
    if 0 < ep < 20:
        for k in range(6):
            ph = (x.t * 0.9 + k / 6) % 1.0
            img = GRID[(int(x.t * 0.9 + k / 6) * 6 + k) % 96]
            xx = lerp(40, 150, ph) - 30
            yy = 240 + k * 105
            draw_image(c, img, xx, yy, 60, 60, a * smooth(ph * 4) * smooth((1 - ph) * 4), 6)
    # chart
    gx, gy, gw, gh = 1080, 230, 700, 520
    ca = x.at(0, 0.8, 0.6)
    c.drawLine(gx, gy + gh, gx + gw, gy + gh, stroke(FAINT, ca, 2))
    for v in (0, 25, 50, 75, 100):
        yy = gy + gh - gh * v / 100
        c.drawLine(gx, yy, gx + gw, yy, stroke(FAINT, ca * 0.35, 1))
        text(c, f'{v}%', gx - 16, yy, 20, MUTED, ca, 500, 'r', vcenter=True)
    for e in (0, 5, 10, 15, 20):
        text(c, str(e), gx + gw * e / 20, gy + gh + 34, 20, MUTED, ca, 500, 'c')
    text(c, 'passes through the training data (epochs)', gx + gw / 2, gy + gh + 74, 22, MUTED, ca, 500, 'c')
    text(c, 'Accuracy on held-out images', gx, gy - 30, 26, INK, ca, 600)
    # chance line
    yy = gy + gh - gh * 0.10
    c.drawLine(gx, yy, gx + gw, yy, stroke(MUTED, ca * 0.6, 1.5, dash=[6, 8]))
    text(c, 'random guessing (10%)', gx + gw - 4, yy - 12, 18, MUTED, ca * 0.8, 500, 'r')
    path = skia.Path()
    n_full = int(ep)
    pts = [(gx + gw * e / 20, gy + gh - gh * curve[e]) for e in range(n_full + 1)]
    if ep > n_full and n_full < 20:
        fr = ep - n_full
        pts.append((gx + gw * ep / 20, gy + gh - gh * lerp(curve[n_full], curve[n_full + 1], fr)))
    if len(pts) > 1:
        path.moveTo(*pts[0])
        for p in pts[1:]:
            path.lineTo(*p)
        c.drawPath(path, stroke(GREEN, ca, 8, blur=6))
        c.drawPath(path, stroke(GREEN, ca, 4))
    hx, hy = pts[-1]
    c.drawCircle(hx, hy, 8, fill(GREEN, ca))
    text(c, f'epoch {int(round(ep))}', gx + gw, gy - 30, 26, MUTED, ca * clamp(ep), 600, 'r')
    text(c, f'{100 * acc_now:.0f}%', hx + 14, hy - 16, 26, GREEN, ca * clamp(ep), 700)
    ba = x.at(2, 0.7, 0.8)
    if ba > 0:
        rrect(c, gx + 200, gy + 250, 460, 130, 22, fill((20, 52, 46), ba))
        rrect(c, gx + 200, gy + 250, 460, 130, 22, stroke(GREEN, ba, 2))
        text(c, '≈ 82% correct', gx + 430, gy + 300, 38, GREEN, ba, 700, 'c', vcenter=True)
        text(c, 'on 10,000 unseen test images', gx + 430, gy + 350, 22, MUTED, ba, 500, 'c', vcenter=True)
    if x.t < x.cues[1] + 0.5:
        text(c, 'random weights, random guesses', net.xh, 920, 24, CORAL, x.at(0, 0.6, 1.0), 600, 'c')


def scene_neurons(c, x):
    section(c, '05', 'What the neurons learned', x.at(0, 0.8))
    tile, gap = 182, 24
    gx = (W - (8 * (tile + gap) - gap)) / 2
    gy = 250
    flip = x.word(1, 0.55, 0.5) * (1 - x.word(1, 0.85, 0.5))
    for j in range(16):
        r, q = divmod(j, 8)
        pop = eback(x.since(0, 0.3 + j * 0.07) / 0.6)
        s = tile * pop
        cx = gx + q * (tile + gap) + tile / 2
        cy = gy + r * (tile + gap + 40) + tile / 2
        if pop <= 0:
            continue
        draw_image(c, NEURON_IMGS[j], cx - s / 2, cy - s / 2, s, s, clamp(pop), 14)
        if flip > 0:
            draw_image(c, NEURON_INIT_IMGS[j], cx - s / 2, cy - s / 2, s, s, flip, 14)
        text(c, f'neuron {j + 1}', cx, cy + tile / 2 + 30, 20, MUTED, clamp(pop), 500, 'c')
    if flip > 0.02:
        text(c, 'before training: random noise', W / 2, 205, 28, CORAL, flip, 600, 'c')
    la = x.at(1, 0.8, 0.0)
    lx, ly, lw = W / 2 - 300, 790, 600
    sh = skia.GradientShader.MakeLinear([skia.Point(lx, 0), skia.Point(lx + lw, 0)],
                                        [col(BLUE, la), col((14, 19, 33), la), col(ORANGE, la)], [0.0, 0.5, 1.0])
    p = skia.Paint(AntiAlias=True)
    p.setShader(sh)
    rrect(c, lx, ly, lw, 26, 13, p)
    text(c, 'pushes the total down', lx, ly + 66, 24, BLUE, la, 600)
    text(c, 'pushes the total up', lx + lw, ly + 66, 24, ORANGE, la, 600, 'r')
    text(c, 'weight ≈ 0', lx + lw / 2, ly + 100, 20, MUTED, la, 500, 'c')


def scene_forward(c, x):
    section(c, '06', 'A prediction', x.at(0, 0.8))
    k_ex = 7
    hid = A['gallery_hidden'][k_ex]
    hid_n = hid / hid.max()
    probs = A['gallery_probs'][k_ex]
    a0 = x.at(0, 0.8, 0.2)
    draw_image(c, GALLERY[k_ex], 110, 380, 320, 320, a0, 16)
    text(c, 'test image', 270, 750, 24, MUTED, a0, 500, 'c')
    hx, hy0, ht = 760, 150, 44
    hgap = 2.4
    yy = [hy0 + j * (ht + hgap) for j in range(16)]
    flow = x.at(1, 0.8)
    for j in range(16):
        a = x.at(0, 0.5, 0.6 + j * 0.04)
        draw_image(c, NEURON_IMGS[j], hx, yy[j], ht, ht, a, 6)
        grow = eout(x.since(1, 0.8 + j * 0.08) / 0.8)
        bl = 230 * hid_n[j] * grow
        if hid_n[j] < 0.01 and grow > 0:
            text(c, '0', hx + ht + 16, yy[j] + ht / 2, 18, FAINT, grow, 600, vcenter=True)
        rrect(c, hx + ht + 12, yy[j] + 10, max(0.1, bl), ht - 20, 6, fill(ORANGE, a * (0.35 + 0.65 * hid_n[j])))
    text(c, 'hidden neurons', hx + 140, 120, 24, MUTED, x.at(0, 0.6, 0.6), 600, 'c')
    # particles from image to hidden
    if 0 < flow:
        for p_i in range(48):
            ph = ((x.t - x.cues[1]) * 0.8 + p_i / 48) % 1.0
            j = p_i % 16
            sx, sy = 430, 540
            ex, ey = hx - 8, yy[j] + ht / 2
            px = lerp(sx, ex, ph)
            py = lerp(sy, ey, smooth(ph))
            fa = flow * (1 - x.at(2, 0.6)) * math.sin(math.pi * ph) * (0.3 + 0.7 * hid_n[j])
            c.drawCircle(px, py, 3.5, fill(GOLD, fa))
    # outputs
    ox, oy0, oh = 1230, 230, 60
    oa = x.at(2, 0.6, 0.0)
    text(c, 'probability of each class', ox + 330, 190, 24, MUTED, oa, 600, 'c')
    win = int(np.argmax(probs))
    for k in range(10):
        y = oy0 + k * oh
        g = eout(x.since(2, 0.5 + k * 0.05) / 0.9)
        is_win = k == win
        hl = x.at(2, 0.6, 2.0) if is_win else 0
        text(c, CLASS_SHORT[k], ox + 150, y + 30, 24, mix(MUTED, INK, hl), oa, 600, 'r', vcenter=True)
        rrect(c, ox + 170, y + 14, 360, 32, 8, fill(PANEL, oa))
        bw = 360 * probs[k] * g
        if bw > 0.5:
            rrect(c, ox + 170, y + 14, bw, 32, 8, fill(GREEN if is_win else BLUE, oa))
        text(c, f'{100 * probs[k] * g:.1f}%', ox + 550, y + 30, 22, mix(MUTED, INK, hl), oa, 600, vcenter=True)
        # connection lines from hidden to output
        if g > 0 and k == win:
            for j in range(16):
                if hid_n[j] > 0.05:
                    c.drawLine(hx + ht + 250, yy[j] + ht / 2, ox + 20, y + 30,
                               stroke(ORANGE if W2[k, j] > 0 else BLUE, oa * 0.25 * hid_n[j], 1.5))
    ca = x.at(2, 0.6, 2.6)
    if ca > 0:
        check_mark(c, ox + 190, 900, 48, GREEN, ca, progress=x.at(2, 0.6, 2.6))
        text(c, 'Correct: it is a sneaker', ox + 230, 912, 30, GREEN, ca, 700)


def histogram(c, x0, y0, w, h, thr_frac, a, grow):
    vals = ABS1_ALL
    bins = np.linspace(0, np.percentile(vals, 99.5), 56)
    counts, _ = np.histogram(vals, bins)
    cmax = counts.max()
    bw = w / len(counts)
    thr = np.quantile(vals, thr_frac) if thr_frac > 0 else -1
    for i, n in enumerate(counts):
        bh = h * n / cmax * grow
        cut = bins[i + 1] <= thr
        rgb = CORAL if cut else VIOLET
        rrect(c, x0 + i * bw + 1, y0 + h - bh, bw - 2, bh, 2, fill(rgb, a * (0.35 if cut else 0.95)))
    c.drawLine(x0, y0 + h, x0 + w, y0 + h, stroke(FAINT, a, 2))
    text(c, '0', x0, y0 + h + 32, 20, MUTED, a, 500, 'c')
    text(c, 'size of weight  |w|', x0 + w / 2, y0 + h + 66, 22, MUTED, a, 500, 'c')
    text(c, 'larger', x0 + w - 18, y0 + h + 32, 20, MUTED, a, 500, 'r')
    arrow(c, x0 + w - 14, y0 + h + 25, x0 + w, y0 + h + 25, MUTED, a, 2, 7)
    text(c, 'number of weights', x0, y0 - 22, 22, MUTED, a, 500)
    if thr_frac > 0:
        tx = x0 + w * thr / bins[-1]
        c.drawLine(tx, y0 - 10, tx, y0 + h, stroke(GOLD, a, 3))
        text(c, f'cut below here: {100 * thr_frac:.0f}% removed', tx + 12, y0 + 20, 22, GOLD, a, 600)


def scene_pruning(c, x):
    section(c, '07', 'Pruning', x.at(0, 0.8))
    net = Net(170, 200, 880, 860, in_r=6, h_r=11, o_r=10)
    span_dur = max(2.5, x.ends[2] - x.cues[2] - 1.0)
    cut = smooth(clamp((x.t - x.cues[2] - 1.0) / span_dur)) * 0.8
    m1, m2 = magnitude_masks(cut)
    a = x.at(0, 0.8, 0.2)
    net.draw(c, alpha=a, v1=m1, v2=m2, labels=1.0)
    ha = x.at(0, 0.8, 1.5)
    histogram(c, 1090, 300, 690, 400, cut, ha, eout(x.since(0, 1.5) / 1.2))
    if x.at(0, 0.6, 3.5) > 0:
        na = x.at(0, 0.6, 3.5) * (1 - x.at(2, 0.5))
        text(c, 'most weights are close to zero', 1250, 380, 26, INK, na, 600)
        arrow(c, 1244, 372, 1135, 330, INK, na, 2.5, 10)
    pa = x.at(1, 0.8)
    if pa > 0:
        rrect(c, 1090, 790, 690, 120, 20, fill(PANEL, pa))
        text(c, 'Pruning', 1120, 838, 30, GOLD, pa, 700)
        text(c, 'remove parts of a trained network', 1120, 880, 24, INK, pa, 500)
        text(c, 'smaller  ·  faster  ·  cheaper to run', 1760, 838, 22, MUTED, pa, 500, 'r')


def scene_neuron_pruning(c, x):
    section(c, '08', 'Pruning whole neurons', x.at(0, 0.8))
    phase2 = x.at(2, 0.8)
    a = x.at(0, 0.8, 0.2) * (1 - phase2)
    if a > 0:
        rng = np.random.default_rng(4)
        keep = rng.random(W1S.shape) > 0.6
        keep2 = rng.random(W2.shape) > 0.6
        left = Net(170, 260, 770, 800, in_r=5, h_r=10, o_r=9)
        cut = x.at(0, 1.2, 1.0)
        left.draw(c, alpha=a, v1=1 - cut * (1 - keep), v2=1 - cut * (1 - keep2))
        text(c, 'Prune connections', 470, 200, 30, INK, a, 700, 'c')
        right = Net(1120, 260, 1720, 800, in_r=5, h_r=10, o_r=9)
        removed = np.array([1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1], float)
        hv = 1 - cut * removed
        right.draw(c, alpha=a, hid_vis=hv)
        for j in np.flatnonzero(removed):
            cross_mark(c, right.xh, right.yh[j], 14, CORAL, a * cut, progress=cut, w=3)
        text(c, 'Prune neurons', 1420, 200, 30, INK, a, 700, 'c')
        ba = x.at(1, 0.8, 0.3)
        text(c, 'zeros scattered inside', 470, 870, 24, MUTED, a * ba, 500, 'c')
        text(c, 'the same-size weight table', 470, 902, 24, MUTED, a * ba, 500, 'c')
        text(c, 'the weight table itself shrinks', 1420, 870, 24, GREEN, a * ba, 600, 'c')
        text(c, 'multiply-adds per image: 12,704 down to 6,352', 1420, 906, 24, GREEN, a * ba, 600, 'c')
    if phase2 > 0:
        net = Net(360, 180, 1240, 860)
        g4 = x.word(2, 0.55, 0.8)
        g2 = x.word(2, 0.78, 0.8)
        keep4 = np.array([(60 >> g) & 1 for g in range(8)])
        keep2 = np.array([(24 >> g) & 1 for g in range(8)])
        kept = lerp(np.ones(8), keep4.astype(float), g4)
        kept = lerp(kept, keep2.astype(float), g2)
        hv = np.repeat(kept, 2)
        net.draw(c, alpha=phase2, hid_vis=hv, labels=1.0)
        ga = x.at(2, 0.6, 1.0)
        for g in range(8):
            y0, y1 = net.yh[2 * g] - 20, net.yh[2 * g + 1] + 20
            rrect(c, net.xh - 30, y0, 60, y1 - y0, 14, stroke(GOLD, phase2 * ga * (0.35 + 0.65 * kept[g]), 2.5))
            text(c, f'G{g + 1}', net.xh - 48, (y0 + y1) / 2, 20, GOLD, phase2 * ga * (0.35 + 0.65 * kept[g]), 700, 'r', vcenter=True)
        text(c, '16 neurons  =  8 pairs', net.xh, 140, 28, INK, phase2 * ga, 700, 'c')
        # mask readout
        bx, by = 1560, 300
        ra = x.word(2, 0.5, 0.6)
        text(c, 'keep / drop', bx + 70, by - 40, 22, MUTED, ra, 600, 'c')
        for g in range(8):
            on = kept[g]
            rrect(c, bx, by + g * 52, 140, 42, 10, fill(mix((60, 30, 36), (24, 66, 52), on), ra))
            text(c, f'G{g + 1}  {"keep" if on > 0.5 else "drop"}', bx + 70, by + g * 52 + 21, 20,
                 GREEN if on > 0.5 else CORAL, ra, 700, 'c', vcenter=True)
        lbl = '50% of neurons removed' if g2 < 0.5 else '75% of neurons removed'
        la = g4 * (1 - abs(g2 - 0.5) * 0 if True else 1)
        text(c, lbl, W / 2, 930, 32, GOLD, la * (1 if g2 < 0.3 or g2 > 0.7 else abs(g2 - 0.5) * 2), 700, 'c')


def scene_which(c, x):
    section(c, '09', 'Which neurons to keep?', x.at(0, 0.8))
    r = M['results']
    sd = M['results_sd']
    x0, x1, vmin, vmax = 560, 1760, 66.0, 84.0

    def X(v):
        return x0 + (x1 - x0) * (v - vmin) / (vmax - vmin)

    a = x.at(0, 0.8, 0.3)
    for v in range(66, 85, 2):
        c.drawLine(X(v), 230, X(v), 870, stroke(FAINT, a * 0.3, 1))
        text(c, f'{v}%', X(v), 910, 20, MUTED, a, 500, 'c')
    text(c, 'test accuracy (mean of 5 runs, bars = ±1 SD)', (x0 + x1) / 2, 950, 20, MUTED, a, 500, 'c')
    dense = r['50.0|Dense']
    c.drawLine(X(dense), 210, X(dense), 870, stroke(INK, a, 2, dash=[8, 8]))
    text(c, f'full network {dense:.1f}%', X(dense), 196, 22, INK, a, 600, 'c')
    rows = [('Random pick', 'Random (10-mask mean)', CORAL),
            ('Largest weights', 'Magnitude', BLUE),
            ('Best choice found', 'Best of 32 random', GREEN)]
    for gi, (p, title, cue) in enumerate([(50.0, 'Half the neurons removed', 1), (75.0, 'Three quarters removed', 2)]):
        ga = x.at(cue, 0.8)
        y0 = 290 + gi * 330
        text(c, title, 120, y0 - 20, 28, INK, ga, 700)
        for ri, (label, key, rgb) in enumerate(rows):
            y = y0 + 40 + ri * 78
            m = r[f'{p}|{key}']
            s = sd[f'{p}|{key}']
            g = eout(x.since(cue, 0.5 + ri * 0.35) / 1.0)
            text(c, label, 520, y, 24, MUTED, ga, 600, 'r', vcenter=True)
            c.drawLine(X(vmin), y, X(vmin + (m - vmin) * g), y, stroke(rgb, ga * 0.25, 2))
            if g > 0.9:
                q = (g - 0.9) / 0.1
                c.drawLine(X(m - s), y, X(m + s), y, stroke(rgb, ga * q * 0.8, 3))
                c.drawLine(X(m - s), y - 10, X(m - s), y + 10, stroke(rgb, ga * q * 0.8, 3))
                c.drawLine(X(m + s), y - 10, X(m + s), y + 10, stroke(rgb, ga * q * 0.8, 3))
            cx = X(vmin + (m - vmin) * g)
            c.drawCircle(cx, y, 16, fill(rgb, ga * g * 0.4, blur=6))
            c.drawCircle(cx, y, 11, fill(rgb, ga * clamp(g * 2)))
            text(c, f'{m:.1f}%', cx + (-30 if p == 75.0 and ri == 0 and False else 26), y - 22, 22, rgb, ga * clamp(g * 3 - 2), 700, 'c')


def scene_retrain(c, x):
    section(c, '10', 'Retraining after pruning', x.at(0, 0.8))
    split = x.at(1, 1.0)
    a = x.at(0, 0.8, 0.2)
    cxs = (lerp(660, 230, split), lerp(660, 1100, split))
    # left/right networks share the 80%-pruned mask
    left = Net(cxs[0], 260, cxs[0] + 600, 820, in_r=6, h_r=11, o_r=10)
    shuffle = x.at(1, 1.2, 1.0)
    if 0 < shuffle < 1:
        rnd = np.random.default_rng(int(x.t * 14))
        w1l = rnd.uniform(-1 / 28, 1 / 28, W1S.shape)
        w2l = rnd.uniform(-0.25, 0.25, W2.shape)
    elif shuffle >= 1:
        w1l, w2l = W1S_FRESH, W2_FRESH
    else:
        w1l, w2l = W1S, W2
    if split < 0.999:
        left.draw(c, alpha=a, v1=M80_1, v2=M80_2)
        text(c, 'pruned network (80% of weights removed)', cxs[0] + 300, 220, 28, INK, a * (1 - split), 600, 'c')
    if split > 0:
        left.draw(c, w1=w1l, w2=w2l, alpha=split, v1=M80_1, v2=M80_2,
                  s1=lerp(SCALE1, INIT_S1, shuffle), s2=lerp(SCALE2, INIT_S2, shuffle))
        text(c, 'A · Fresh random start', cxs[0] + 300, 220, 30, CORAL, split, 700, 'c')
        text(c, 'brand-new random weights', cxs[0] + 300, 880, 24, MUTED, split * shuffle, 500, 'c')
    rew = x.at(2, 1.6, 0.8)
    ra = x.at(2, 0.8)
    if ra > 0:
        right = Net(1100, 260, 1700, 820, in_r=6, h_r=11, o_r=10)
        w1r = W1S + (W1S_INIT - W1S) * rew
        w2r = W2 + (W2_INIT - W2) * rew
        right.draw(c, w1=w1r, w2=w2r, alpha=ra, v1=M80_1, v2=M80_2,
                   s1=lerp(SCALE1, INIT_S1, rew), s2=lerp(SCALE2, INIT_S2, rew))
        text(c, 'B · Rewind', 1400, 220, 30, GREEN, ra, 700, 'c')
        text(c, 'the same starting values as before training', 1400, 880, 24, MUTED, ra, 500, 'c')
        # rewind icon
        ix, iy = 1400, 950
        ang = -rew * 720
        p = skia.Path()
        p.addArc(skia.Rect.MakeXYWH(ix - 22, iy - 22, 44, 44), ang - 40, -280)
        c.drawPath(p, stroke(GREEN, ra, 4))
        hx = ix + 22 * math.cos(math.radians(ang - 40))
        hy = iy + 22 * math.sin(math.radians(ang - 40))
        c.drawCircle(hx, hy, 5, fill(GREEN, ra))
        text(c, 'back to the start', ix + 40, iy + 8, 22, GREEN, ra, 600)
    elif split > 0.5:
        # placeholder slot for option B
        rrect(c, 1100, 260, 600, 560, 24, stroke(FAINT, split - 0.5, 2, dash=[10, 10]))
        text(c, 'B · ?', 1400, 220, 30, MUTED, split - 0.5, 700, 'c')


def scene_lth(c, x):
    tA = x.at(0, 0.9, 0.2)
    to_grid = x.at(1, 0.9)
    title_y = lerp(470, 150, to_grid)
    size = lerp(84, 44, to_grid)
    text(c, 'The Lottery Ticket Hypothesis', W / 2, title_y, size, GOLD, tA, 700, 'c')
    text(c, 'Jonathan Frankle & Michael Carbin  ·  ICLR 2019', W / 2, title_y + lerp(70, 46, to_grid),
         lerp(30, 22, to_grid), MUTED, tA, 500, 'c')
    if to_grid <= 0:
        q = x.at(0, 0.8, 4.5)
        text(c, '“The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks”', W / 2, 640, 26, INK, q * 0.8, 500, 'c')
        return
    cols, rows = 6, 4
    tw, th, gx_, gy_ = 236, 132, 34, 32
    ox = (W - (cols * (tw + gx_) - gx_)) / 2
    oy = 250
    winners = {3, 13, 20}
    rng = np.random.default_rng(11)
    zoom = x.at(3, 1.0)
    scratch_t = x.since(2, 0.6)
    for i in range(cols * rows):
        r, q = divmod(i, cols)
        pop = eback(x.since(1, 0.4 + (r + q) * 0.06) / 0.6)
        if pop <= 0:
            continue
        edges = [e for e in ALL_GLYPH_EDGES if rng.random() < 0.38]
        tx = ox + q * (tw + gx_)
        ty = oy + r * (th + gy_)
        is_win = i in winners
        sc = clamp((scratch_t - (r * cols + q) * 0.07) / 0.6)
        a = clamp(pop) * (1 - zoom * (0 if (is_win and i == 13) else 1))
        if i == 13:
            tx = lerp(tx, 210, zoom)
            ty = lerp(ty, 360, zoom)
        s = 1 + zoom * 1.3 if i == 13 else 1
        w_, h_ = tw * s, th * s
        if a <= 0.003:
            continue
        base = mix(PANEL, (70, 56, 20), sc) if is_win else mix(PANEL, (26, 30, 42), sc)
        rrect(c, tx, ty, w_, h_, 14, fill(base, a))
        rrect(c, tx, ty, w_, h_, 14, stroke(GOLD if (is_win and sc > 0.5) else FAINT, a, 2.5 if is_win and sc > 0.5 else 1.5))
        # ticket notches + perforation
        for nx in (tx, tx + w_):
            c.drawCircle(nx, ty + h_ / 2, 11 * s, fill(BG_BOT, a))
        c.drawLine(tx + w_ * 0.78, ty + 12, tx + w_ * 0.78, ty + h_ - 12, stroke(FAINT, a, 1.5, dash=[4, 6]))
        glyph_rgb = GOLD if (is_win and sc > 0.5) else (MUTED if sc < 0.5 else FAINT)
        mini_net_glyph(c, tx + 14 * s, ty + 6 * s, w_ * 0.7, h_ - 12 * s, edges, glyph_rgb, a * (0.9 if sc > 0.5 else 0.7),
                       node_rgb=glyph_rgb, lw=1.8 * s)
        label = 'WIN' if is_win else '—'
        text(c, label if sc > 0.5 else '#' + str(1000 + i * 37), tx + w_ * 0.89, ty + h_ / 2, 17 * s,
             GOLD if (is_win and sc > 0.5) else MUTED, a * (1 if sc > 0.5 or sc == 0 else 0.4), 700, 'c', vcenter=True)
        # silver scratch coating sweeping away
        if sc < 1:
            cover = skia.Rect.MakeXYWH(tx + 10 + (w_ * 0.7) * sc, ty + 10, (w_ * 0.7) * (1 - sc), h_ - 20)
            c.drawRoundRect(cover, 8, 8, fill((150, 158, 176), a * 0.85 * x.at(1, 1.0, 3.0)))
    if zoom > 0:
        za = x.at(3, 0.8, 0.8)
        bx = 840
        rrect(c, bx, 330, 900, 150, 22, fill((20, 52, 46), za))
        check_mark(c, bx + 60, 405, 40, GREEN, za, x.at(3, 0.6, 1.2))
        text(c, 'Original starting weights', bx + 110, 392, 30, INK, za, 700)
        text(c, 'trains alone to match the full network', bx + 110, 434, 24, GREEN, za, 600)
        zb = x.at(3, 0.8, 4.2)
        rrect(c, bx, 520, 900, 150, 22, fill((60, 28, 34), zb))
        cross_mark(c, bx + 60, 595, 36, CORAL, zb, x.at(3, 0.6, 4.6))
        text(c, 'New random starting weights', bx + 110, 582, 30, INK, zb, 700)
        text(c, 'usually loses its edge', bx + 110, 624, 24, CORAL, zb, 600)


def scene_recipe(c, x):
    section(c, '12', 'Finding a winning ticket', x.at(0, 0.8))
    steps = ['Train', 'Prune smallest', 'Rewind', 'Train again']
    fr = [0.12, 0.30, 0.48, 0.66]
    cx, cy, R = 600, 540, 300
    a = x.at(0, 0.8, 0.2)
    c.drawCircle(cx, cy, R, stroke(FAINT, a, 3, dash=[10, 12]))
    after = x.t - x.ends[0]
    lap_period = 2.2
    laps = int(max(0, after) / lap_period) if after > 0 else 0
    if x.t < x.ends[0]:
        sp = x.span(0)
        cur = sum(sp >= f for f in fr) - 1
        phase = clamp(sp / fr[-1]) * 0.75 if sp < fr[-1] else 0.75 + clamp((sp - fr[-1]) / (1 - fr[-1])) * 0.25
    else:
        ph = (after % lap_period) / lap_period
        cur = int(ph * 4) % 4
        phase = ph
    ang = -90 + 360 * phase
    hx = cx + R * math.cos(math.radians(ang))
    hy = cy + R * math.sin(math.radians(ang))
    c.drawCircle(hx, hy, 18, fill(GOLD, a * 0.5, blur=8))
    c.drawCircle(hx, hy, 9, fill(GOLD, a))
    for i, s in enumerate(steps):
        an = math.radians(-90 + 90 * i)
        sx, sy = cx + R * math.cos(an), cy + R * math.sin(an)
        on = 1.0 if i == cur else 0.0
        ap = a * x.word(0, fr[i] - 0.06, 0.5) if x.t < x.ends[0] else a
        bw = 236
        rrect(c, sx - bw / 2, sy - 40, bw, 80, 20, fill(mix(PANEL, (60, 50, 22), on), ap))
        rrect(c, sx - bw / 2, sy - 40, bw, 80, 20, stroke(GOLD if on else FAINT, ap, 2.5))
        text(c, f'{i + 1}', sx - bw / 2 + 28, sy, 24, GOLD, ap, 700, 'c', vcenter=True)
        text(c, s, sx + 14, sy, 24, INK, ap, 600, 'c', vcenter=True)
    ra = x.word(0, 0.8, 0.6)
    text(c, 'repeat', cx, cy - 10, 40, GOLD, ra, 700, 'c', vcenter=True)
    text(c, 'cut 20% more each round', cx, cy + 34, 22, MUTED, ra, 500, 'c')
    rounds = min(12, laps + (1 if x.t > lerp(x.cues[0], x.ends[0], fr[1]) else 0))
    target = 1 - 0.8 ** rounds
    sp = 1 - 0.8 ** max(0, rounds - 1)
    sp = lerp(sp, target, smooth((x.t - (x.cues[0] if laps == 0 else x.ends[0] + laps * lap_period)) / 0.6)) if rounds else 0
    m1, m2 = magnitude_masks(sp)
    net = Net(1080, 220, 1720, 860, in_r=6, h_r=11, o_r=10)
    net.draw(c, alpha=a, v1=m1, v2=m2)
    text(c, f'round {rounds}', 1400, 180, 28, INK, a, 700, 'c')
    text(c, f'{100 * sp:.0f}% of weights removed', 1400, 920, 28, GOLD, a, 700, 'c')


def scene_evidence(c, x):
    section(c, '13', 'Does it work here?', x.at(0, 0.8))
    rows = M['lottery_iterative']
    data = {}
    for rr in rows:
        data.setdefault(rr['variant'], []).append((rr['sparsity'], rr['mean'], rr['sd']))
    gx, gy, gw, gh = 300, 210, 1180, 600
    vmin, vmax = 74.0, 84.0
    rem_max, rem_min = 1.0, 0.065

    def X(s):
        rem = max(1 - s, rem_min)
        return gx + gw * (math.log(rem_max) - math.log(rem)) / (math.log(rem_max) - math.log(rem_min))

    def Y(v):
        return gy + gh - gh * (v - vmin) / (vmax - vmin)

    a = x.at(0, 0.8, 0.3)
    for v in range(74, 85, 2):
        c.drawLine(gx, Y(v), gx + gw, Y(v), stroke(FAINT, a * 0.35, 1))
        text(c, f'{v}%', gx - 16, Y(v), 20, MUTED, a, 500, 'r', vcenter=True)
    for rem in (1.0, 0.5, 0.25, 0.1):
        text(c, f'{int(rem * 100)}%', X(1 - rem), gy + gh + 36, 20, MUTED, a, 500, 'c')
    text(c, 'weights remaining  (log scale)', gx + gw / 2, gy + gh + 76, 22, MUTED, a, 500, 'c')
    text(c, 'test accuracy · 5 runs each · bands = ±1 SD', gx, gy - 28, 22, MUTED, a, 500)
    dense = M['results']['50.0|Dense']
    c.drawLine(gx, Y(dense), gx + gw, Y(dense), stroke(INK, a * 0.7, 2, dash=[8, 8]))
    text(c, 'full network', gx + gw + 14, Y(dense), 20, INK, a * 0.8, 600, vcenter=True)
    draw_p = clamp(x.since(1, 0.4) / max(2.0, x.ends[1] - x.cues[1] - 0.2))
    for variant, rgb, cue_frac in [('winning ticket', GREEN, 0.0), ('random reinit', CORAL, 0.45)]:
        pts = sorted(data[variant])
        pr = clamp((draw_p - cue_frac) / 0.55)
        pr = smooth(pr)
        if pr <= 0:
            continue
        n = len(pts)
        upto = pr * (n - 1)
        k = int(upto)
        band = skia.Path()
        upper = [(X(s), Y(m + d)) for s, m, d in pts]
        lower = [(X(s), Y(m - d)) for s, m, d in pts]
        line = [(X(s), Y(m)) for s, m, d in pts]

        def cut(seq):
            out = seq[:k + 1]
            if k < n - 1:
                f = upto - k
                out = out + [(lerp(seq[k][0], seq[k + 1][0], f), lerp(seq[k][1], seq[k + 1][1], f))]
            return out
        U, L, Ln = cut(upper), cut(lower), cut(line)
        band.moveTo(*U[0])
        for p in U[1:]:
            band.lineTo(*p)
        for p in reversed(L):
            band.lineTo(*p)
        band.close()
        c.drawPath(band, fill(rgb, a * 0.16))
        path = skia.Path()
        path.moveTo(*Ln[0])
        for p in Ln[1:]:
            path.lineTo(*p)
        c.drawPath(path, stroke(rgb, a * 0.6, 9, blur=6))
        c.drawPath(path, stroke(rgb, a, 4))
        for i, (px, py) in enumerate(Ln[:k + 1]):
            c.drawCircle(px, py, 6, fill(rgb, a))
        ex, ey = Ln[-1]
        lab = 'rewound to original start' if variant == 'winning ticket' else 'fresh random start'
        text(c, lab, ex + 18, ey + (-18 if variant == 'winning ticket' else 30), 22, rgb, a * pr, 700)
    # callouts with exact numbers
    win = {round(s, 3): m for s, m, _ in data['winning ticket']}
    rnd = {round(s, 3): m for s, m, _ in data['random reinit']}
    last = max(win)
    ca = x.at(2, 0.8, 0.5)
    if ca > 0:
        px = X(last)
        c.drawLine(px, Y(win[last]) - 8, px, Y(rnd[last]) + 8, stroke(GOLD, ca, 3))
        gap = win[last] - rnd[last]
        rrect(c, px - 300, Y(vmin) - 160, 280, 120, 18, fill(PANEL, ca))
        text(c, f'{100 * last:.0f}% removed', px - 160, Y(vmin) - 122, 22, MUTED, ca, 600, 'c')
        text(c, f'{win[last]:.1f}%  vs  {rnd[last]:.1f}%', px - 160, Y(vmin) - 84, 28, INK, ca, 700, 'c')
        text(c, f'+{gap:.1f} points for the ticket', px - 160, Y(vmin) - 52, 20, GOLD, ca, 600, 'c')
    na = x.at(4, 0.8) if len(x.cues) > 4 else 0
    if na > 0:
        text(c, 'One small network, one dataset:', gx + 30, Y(76.2), 24, MUTED, na, 500)
        text(c, 'a demonstration, not a proof.', gx + 30, Y(76.2) + 34, 24, MUTED, na, 500)


def scene_outro(c, x):
    section(c, '14', 'Back to Qismet', x.at(0, 0.8) * (1 - x.at(1, 0.6)))
    pa = x.at(0, 0.8) * (1 - x.at(1, 0.8))
    if pa > 0:
        steps = [('Train the full', 'network'), ('Choose which', 'neuron groups stay'), ('Rewind survivors', 'to original start'),
                 ('Retrain', '20 epochs'), ('Measure', 'test accuracy')]
        bw, gap = 300, 50
        ox = (W - (5 * bw + 4 * gap)) / 2
        for i, (l1, l2) in enumerate(steps):
            ap = pa * x.at(0, 0.5, 0.3 + i * 0.35)
            bx = ox + i * (bw + gap)
            hl = i == 2
            rrect(c, bx, 420, bw, 170, 24, fill(mix(PANEL, (20, 52, 46), 1.0 if hl else 0), ap))
            rrect(c, bx, 420, bw, 170, 24, stroke(GREEN if hl else FAINT, ap, 3 if hl else 1.5))
            text(c, l1, bx + bw / 2, 492, 26, INK, ap, 700, 'c')
            text(c, l2, bx + bw / 2, 530, 22, GREEN if hl else MUTED, ap, 500, 'c')
            if i < 4:
                arrow(c, bx + bw + 8, 505, bx + bw + gap - 8, 505, MUTED, ap, 3, 10)
        text(c, 'Every pruned network in the experiment starts from its original weights',
             W / 2, 700, 26, MUTED, pa * x.at(0, 0.6, 2.4), 500, 'c')
    ra = x.at(1, 0.8, 0.3)
    if ra > 0:
        fr = [0.12, 0.35, 0.58]
        cards = [('A network learns', 'by adjusting its weights', VIOLET),
                 ('Pruning', 'removes the parts it does not need', CORAL),
                 ('The lottery ticket', 'a small network that trains well was\nhidden inside the big one from the start', GOLD)]
        cw, gap = 500, 50
        ox = (W - (3 * cw + 2 * gap)) / 2
        for i, (t1, t2, rgb) in enumerate(cards):
            ap = ra * x.word(1, fr[i], 0.7)
            bx = ox + i * (cw + gap)
            by = 300 + (1 - ap) * 30
            rrect(c, bx, by, cw, 380, 28, fill(PANEL, ap))
            rrect(c, bx, by, cw, 380, 28, stroke(rgb, ap, 2.5))
            if i == 0:
                mini_net_glyph(c, bx + 130, by + 30, 240, 150, ALL_GLYPH_EDGES, rgb, ap * 0.8)
            elif i == 1:
                mini_net_glyph(c, bx + 130, by + 30, 240, 150, ALL_GLYPH_EDGES[::3], rgb, ap * 0.8)
            else:
                rrect(c, bx + 140, by + 50, 220, 120, 14, stroke(rgb, ap, 2.5))
                for nx in (bx + 140, bx + 360):
                    c.drawCircle(nx, by + 110, 12, fill(PANEL, ap))
                text(c, 'WIN', bx + 250, by + 110, 40, rgb, ap, 700, 'c', vcenter=True)
            text(c, t1, bx + cw / 2, by + 250, 32, rgb, ap, 700, 'c')
            for li, ln in enumerate(t2.split('\n')):
                text(c, ln, bx + cw / 2, by + 296 + li * 32, 23, INK, ap, 500, 'c')
    ea = x.at(1, 1.0, max(0.5, x.ends[1] - x.cues[1] + 0.6))
    if ea > 0:
        c.drawRect(skia.Rect.MakeXYWH(0, 0, W, H), fill(BG_TOP, ea))
        text(c, 'Qismet', W / 2, 520, 110, INK, ea, 700, 'c', vcenter=True)
        text(c, 'Grouped neuron pruning on Fashion-MNIST', W / 2, 620, 30, MUTED, ea, 500, 'c')


# ----------------------------------------------------------------------------
# Script: narration lines drive timing. (text, pause after in seconds)
# ----------------------------------------------------------------------------
LT = {r['variant']: r for r in M['lottery_iterative'] if r['key'] == 12}

SCENES = [
    dict(draw=scene_title, lead=0.8, tail=1.8, lines=[
        ("How can a neural network get smaller, without getting worse?", 0.8),
        "In this video, we'll see how a simple network learns to recognize clothing, what it means to prune it, "
        "and a surprising idea called the Lottery Ticket Hypothesis.",
    ]),
    dict(draw=scene_task, tail=2.8, lines=[
        ("Here's the task. Fashion-MNIST is a collection of seventy thousand small grayscale photos of clothing.", 0.6),
        ("Each photo belongs to one of ten classes: T-shirts, trousers, pullovers, dresses, coats, sandals, shirts, "
         "sneakers, bags, and ankle boots.", 0.6),
        "We want a program that looks at an image it has never seen before, and names the class.",
    ]),
    dict(draw=scene_pixels, tail=1.6, lines=[
        ("To a computer, an image is just a grid of numbers.", 0.8),
        ("Each photo is twenty-eight by twenty-eight pixels. Every pixel holds a brightness, from zero for black, "
         "up to one for white.", 0.8),
        "Read the grid row by row, and you get one long list of seven hundred and eighty-four numbers. "
        "That list is what the network sees.",
    ]),
    dict(draw=scene_network, tail=2.2, lines=[
        ("The network in this project is called a multilayer perceptron. It has seven hundred and eighty-four inputs, "
         "sixteen hidden neurons, and ten outputs, one for each class.", 0.8),
        ("Every input connects to every hidden neuron, and every hidden neuron connects to every output. "
         "Each connection has its own number, called a weight.", 1.2),
        ("A neuron multiplies each input by its weight, adds them all up along with a bias, and then applies a rule "
         "called ReLU: negative totals become zero, and positive totals pass straight through.", 1.4),
        "The output with the highest score is the network's answer. All together, there are twelve thousand, "
        "seven hundred and thirty numbers to adjust.",
    ]),
    dict(draw=scene_training, tail=2.2, lines=[
        ("At first, every weight is a random number, so the network's guesses are random too.", 0.6),
        ("Training shows the network thousands of labelled images. After each small batch, it measures how wrong it was, "
         "and nudges every weight slightly in the direction that reduces the error. This is called gradient descent.", 1.0),
        "After twenty passes through five thousand training images, it correctly classifies about eighty-two percent "
        "of ten thousand test images it has never seen.",
    ]),
    dict(draw=scene_neurons, tail=1.6, lines=[
        ("Here is what each of the sixteen hidden neurons learned. Each square shows one neuron's seven hundred and "
         "eighty-four weights, laid back out as a twenty-eight by twenty-eight image.", 0.8),
        "Orange pixels push the neuron's total up, and blue pixels push it down. Training turned random noise "
        "into templates that respond to particular shapes.",
    ]),
    dict(draw=scene_forward, tail=2.6, lines=[
        ("Now let's send a real test image through the trained network.", 0.6),
        ("Each hidden neuron compares the image with its template. The better the match, the more strongly it fires. "
         "Some don't fire at all.", 0.8),
        "The ten output scores become probabilities. Here, the network is ninety-nine point seven percent sure this is "
        "a sneaker, and it's right.",
    ]),
    dict(draw=scene_pruning, tail=2.4, lines=[
        ("But does the network really need all twelve thousand weights? Usually not. Many end up very close to zero, "
         "and barely affect the answer.", 0.6),
        ("Pruning means removing parts of a trained network, to make it smaller and cheaper to run.", 0.6),
        "The simplest rule is magnitude pruning: the weights closest to zero matter least, so cut those first.",
    ]),
    dict(draw=scene_neuron_pruning, tail=2.8, lines=[
        ("You can prune single connections, or whole neurons.", 0.8),
        ("Removing a neuron deletes every connection attached to it at once, so the network really does less work.", 1.0),
        "This project removes neurons in pairs. The sixteen hidden neurons form eight groups. Keeping four groups "
        "removes half of the neurons. Keeping only two removes three quarters.",
    ]),
    dict(draw=scene_which, tail=1.8, lines=[
        ("Which groups you keep matters, especially when you cut deep.", 0.6),
        ("With half the neurons removed, almost any choice works. Random picks reach about eighty percent, close to "
         "the full network's eighty-two.", 0.8),
        ("With three quarters removed, random picks fall to about seventy-one percent, while the best choices found "
         "reach about seventy-six and a half.", 0.8),
        "Finding a good choice is a search problem, and it's the focus of the rest of the Qismet project.",
    ]),
    dict(draw=scene_retrain, tail=2.4, lines=[
        ("Cutting a network usually costs some accuracy, so the smaller network is trained again.", 0.6),
        ("But starting from where? One option is a fresh start: give the surviving weights brand-new random values.", 0.8),
        "The other is to rewind: reset every surviving weight to the exact random value it had before the original "
        "training began.",
    ]),
    dict(draw=scene_lth, tail=2.6, lines=[
        ("In twenty nineteen, Jonathan Frankle and Michael Carbin showed that rewinding can work remarkably well. "
         "They called their idea the Lottery Ticket Hypothesis.", 0.8),
        ("Think of a big, randomly started network as a huge bundle of lottery tickets. Each ticket is a small "
         "subnetwork hidden inside it, with its own random starting weights.", 0.6),
        ("Training the big network is like scratching every ticket at once. Pruning keeps the winners: the small set "
         "of connections that turned out to matter.", 1.0),
        "The claim is that a winning ticket, trained by itself from its original starting weights, can match the "
        "accuracy of the full network. But give it new random starting weights, and it usually loses that edge.",
    ]),
    dict(draw=scene_recipe, tail=9.0, lines=[
        "The recipe is simple. Train the full network. Remove the smallest weights. Rewind the survivors to their "
        "original values. Then train again, and repeat, cutting a little more each round.",
    ]),
    dict(draw=scene_evidence, tail=2.4, lines=[
        ("We tested this on the exact network and data used in Qismet, repeating everything with five different "
         "random starts.", 0.6),
        ("Green is the winning ticket, rewound to its original weights. Red is the same pruned network, trained from "
         "new random weights.", 1.4),
        ("With its original starting weights, the ticket stays at about eighty-two percent until roughly eighty "
         "percent of the weights are gone. Even with ninety-three percent removed, it still reaches about eighty percent.", 0.6),
        ("Trained from new random weights, the same connections fall steadily behind, ending near seventy-seven percent. "
         "The starting values really matter.", 0.6),
        "This is one small network on one dataset, so treat it as a demonstration, not a proof.",
    ]),
    dict(draw=scene_outro, tail=4.5, lines=[
        ("That's why, in Qismet, every pruned network is rewound to its original starting weights before it is "
         "retrained.", 1.0),
        "To recap: a neural network learns by adjusting its weights. Pruning removes the parts it doesn't need. "
        "And the Lottery Ticket Hypothesis says that a small network that trains just as well may have been hidden "
        "inside the big one, from the very start.",
    ]),
]

SECTION_NAMES = {scene_lth: ('11', 'The big idea')}


# ----------------------------------------------------------------------------
# Narration + timeline
# ----------------------------------------------------------------------------
def tts(textline, voice, rate):
    BUILD.mkdir(exist_ok=True)
    (BUILD / 'tts').mkdir(exist_ok=True)
    key = hashlib.sha1(f'{voice}|{rate}|{textline}'.encode()).hexdigest()[:16]
    path = BUILD / 'tts' / f'{key}.wav'
    if not path.exists():
        subprocess.run(['say', '-v', voice, '-r', str(rate), '-o', str(path), '--file-format=WAVE',
                        f'--data-format=LEI16@{SR}', textline], check=True)
    with wave.open(str(path)) as w:
        n = w.getnframes()
        audio = np.frombuffer(w.readframes(n), dtype=np.int16)
    # trim leading/trailing near-silence so timings are tight
    nz = np.flatnonzero(np.abs(audio) > 300)
    if len(nz):
        audio = audio[max(0, nz[0] - 400): nz[-1] + 2000]
    return audio


def build_timeline(voice, rate):
    timeline, audio_clips, t = [], [], 0.0
    for si, sc in enumerate(SCENES):
        start = t
        cur = sc.get('lead', 0.6)
        cues, ends = [], []
        for item in sc['lines']:
            line, pause = (item, 0.45) if isinstance(item, str) else item
            clip = tts(line, voice, rate)
            dur = len(clip) / SR
            cues.append(cur)
            ends.append(cur + dur)
            audio_clips.append((start + cur, clip, line, dur))
            cur += dur + pause
        dur = ends[-1] + sc.get('tail', 1.2)
        timeline.append(dict(index=si, start=start, dur=dur, cues=cues, ends=ends, draw=sc['draw']))
        t += dur
    return timeline, audio_clips, t


def caption_chunks(audio_clips):
    """Split each narration line into short caption chunks timed by length."""
    out = []
    for start, _, line, dur in audio_clips:
        parts = re.split(r'(?<=[.?!:])\s+|(?<=,)\s+(?=\S)', line)
        chunks, cur = [], ''
        for p in parts:
            if cur and len(cur) + len(p) + 1 > 96:
                chunks.append(cur)
                cur = p
            else:
                cur = (cur + ' ' + p).strip()
        if cur:
            chunks.append(cur)
        total = sum(len(ch) for ch in chunks)
        t = start
        for ch in chunks:
            d = dur * len(ch) / total
            out.append((t, t + d, ch))
            t += d
    return out


def wrap(s, size, maxw, weight=500):
    words, lines, cur = s.split(), [], ''
    for w in words:
        trial = (cur + ' ' + w).strip()
        if text_width(trial, size, weight) > maxw and cur:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    lines.append(cur)
    return lines


def draw_caption(c, t, captions):
    for (s, e, ch) in captions:
        if s - 0.1 <= t <= e + 0.25:
            a = smooth((t - s + 0.1) / 0.18) * smooth((e + 0.25 - t) / 0.2)
            lines = wrap(ch, 30, 1500)
            lh = 42
            hh = lh * len(lines) + 26
            wmax = max(text_width(ln, 30) for ln in lines) + 56
            y0 = 1040 - hh
            rrect(c, W / 2 - wmax / 2, y0, wmax, hh, 16, fill((5, 8, 16), a * 0.72))
            for i, ln in enumerate(lines):
                text(c, ln, W / 2, y0 + 13 + lh * i + lh * 0.5, 30, INK, a, 500, 'c', vcenter=True)
            break


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------
_BG = None


def background(scale):
    global _BG
    if _BG is None:
        surf = skia.Surface.MakeRaster(skia.ImageInfo.Make(W * scale, H * scale, skia.ColorType.kRGBA_8888_ColorType,
                                                           skia.AlphaType.kPremul_AlphaType))
        c = surf.getCanvas()
        c.scale(scale, scale)
        p = skia.Paint()
        p.setShader(skia.GradientShader.MakeLinear([skia.Point(0, 0), skia.Point(0, H)], [col(BG_TOP), col(BG_BOT)]))
        c.drawRect(skia.Rect.MakeWH(W, H), p)
        for gx in range(0, W + 1, 48):
            for gy in range(0, H + 1, 48):
                c.drawCircle(gx, gy, 1.1, fill(INK, 0.045))
        v = skia.Paint()
        v.setShader(skia.GradientShader.MakeRadial(skia.Point(W / 2, H * 0.45), W * 0.75,
                                                   [col((0, 0, 0), 0), col((0, 0, 0), 0.45)]))
        c.drawRect(skia.Rect.MakeWH(W, H), v)
        _BG = surf.makeImageSnapshot()
    return _BG


def render_frame(frame, timeline, captions, total, scale, show_captions=True):
    t = frame / FPS
    sc = next((s for s in timeline if s['start'] <= t < s['start'] + s['dur']), timeline[-1])
    lt = t - sc['start']
    surf = skia.Surface.MakeRaster(skia.ImageInfo.Make(W * scale, H * scale, skia.ColorType.kRGBA_8888_ColorType,
                                                       skia.AlphaType.kPremul_AlphaType))
    c = surf.getCanvas()
    bg = background(scale)
    c.drawImage(bg, 0, 0)
    c.scale(scale, scale)
    ctx = Ctx(lt, sc['dur'], sc['cues'], sc['ends'])
    sc['draw'](c, ctx)
    if sc['draw'] in SECTION_NAMES:
        num, title = SECTION_NAMES[sc['draw']]
        section(c, num, title, ctx.at(0, 0.8) * x_section_alpha(ctx))
    # scene transitions: fade through the background
    fade = max(1 - smooth(lt / 0.45), 1 - smooth((sc['dur'] - lt) / 0.45))
    if sc['index'] == 0:
        fade = max(1 - smooth(lt / 0.9), 1 - smooth((sc['dur'] - lt) / 0.45))
    if sc['index'] == len(timeline) - 1:
        fade = 1 - smooth(lt / 0.45) if lt < 1 else 1 - smooth((sc['dur'] - lt) / 1.2)
    if fade > 0.002:
        p = skia.Paint()
        p.setAlphaf(fade)
        c.save()
        c.resetMatrix()
        c.drawImage(bg, 0, 0, skia.SamplingOptions(), p)
        c.restore()
    if show_captions:
        draw_caption(c, t, captions)
    # thin progress bar
    c.drawRect(skia.Rect.MakeXYWH(0, 0, W * t / total, 4), fill(ORANGE, 0.7))
    return surf.makeImageSnapshot().toarray()


def x_section_alpha(ctx):
    return 1 - ctx.at(1, 0.6) * 0 if len(ctx.cues) > 1 else 1


_STATE = {}


def _worker_init(voice, rate, scale, captions_on):
    timeline, clips, total = build_timeline(voice, rate)
    _STATE.update(timeline=timeline, captions=caption_chunks(clips), total=total, scale=scale, cap=captions_on)


def _render_chunk(args):
    idx, f0, f1, out = args
    s = _STATE
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    w, h = W * s['scale'], H * s['scale']
    cmd = [ff, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{w}x{h}', '-r', str(FPS),
           '-i', '-', '-vf', 'scale=out_color_matrix=bt709:out_range=tv', '-c:v', 'libx264', '-preset', 'slow',
           '-crf', '15', '-tune', 'animation', '-pix_fmt', 'yuv420p', '-profile:v', 'high',
           '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv',
           '-g', str(FPS * 2), str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(f0, f1):
        arr = render_frame(f, s['timeline'], s['captions'], s['total'], s['scale'], s['cap'])
        proc.stdin.write(arr.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f'ffmpeg failed on chunk {idx}')
    return idx, f1 - f0


def write_audio(clips, total, path):
    n = int(math.ceil(total * SR)) + SR
    mixbuf = np.zeros(n, np.float32)
    for start, clip, _, _ in clips:
        i = int(start * SR)
        mixbuf[i:i + len(clip)] += clip.astype(np.float32)
    # gentle fade in/out and normalize to -1 dBFS peak
    peak = np.abs(mixbuf).max() or 1
    mixbuf *= (0.89 * 32767) / peak
    with wave.open(str(path), 'w') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.clip(mixbuf, -32768, 32767).astype(np.int16).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--voice', default='Samantha')
    ap.add_argument('--rate', type=int, default=172)
    ap.add_argument('--scale', type=int, default=1, help='1 = 1080p, 2 = 4K')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--no-captions', action='store_true')
    ap.add_argument('--still', help='scene:seconds, writes build/still.png')
    ap.add_argument('--out', default=str(HERE / 'qismet_ml_explained.mp4'))
    args = ap.parse_args()

    timeline, clips, total = build_timeline(args.voice, args.rate)
    captions = caption_chunks(clips)
    print(f'{len(timeline)} scenes, {total:.1f} s, {int(total * FPS)} frames')
    for s in timeline:
        print(f"  scene {s['index']:2d}  start {s['start']:6.1f}s  length {s['dur']:5.1f}s")

    if args.still:
        si, sec = args.still.split(':')
        sc = timeline[int(si)]
        frame = int((sc['start'] + float(sec)) * FPS)
        arr = render_frame(frame, timeline, captions, total, args.scale, not args.no_captions)
        from PIL import Image
        BUILD.mkdir(exist_ok=True)
        out = BUILD / f'still_{si}_{sec}.png'
        Image.fromarray(arr[..., :3]).save(out)
        print(out)
        return

    BUILD.mkdir(exist_ok=True)
    nframes = int(math.ceil(total * FPS))
    nchunks = args.workers * 3
    bounds = np.linspace(0, nframes, nchunks + 1).astype(int)
    jobs = [(i, int(bounds[i]), int(bounds[i + 1]), BUILD / f'chunk_{i:03d}.mp4') for i in range(nchunks)]
    done = 0
    with Pool(args.workers, initializer=_worker_init,
              initargs=(args.voice, args.rate, args.scale, not args.no_captions)) as pool:
        for idx, n in pool.imap_unordered(_render_chunk, jobs):
            done += n
            print(f'  rendered {done}/{nframes} frames', flush=True)
    write_audio(clips, total, BUILD / 'narration.wav')
    listing = BUILD / 'chunks.txt'
    listing.write_text(''.join(f"file '{j[3].name}'\n" for j in jobs))
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', str(listing),
                    '-i', str(BUILD / 'narration.wav'), '-c:v', 'copy', '-c:a', 'aac', '-b:a', '256k',
                    '-ar', '48000', '-movflags', '+faststart', '-shortest', args.out], check=True)
    for j in jobs:
        j[3].unlink()
    print('wrote', args.out)


if __name__ == '__main__':
    main()
