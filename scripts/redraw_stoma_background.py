# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy", "pillow", "scipy", "scikit-image", "skia-python"]
# ///
"""Re-draw the low-res pencil stoma background at print resolution.

The draft (leaf-stoma-pencil.png) is 1448 x 1086 px, ~30 ppi on the 48 x 36 in
poster. Rather than upscaling pixels, this reads three fields off the draft --
local stroke direction, pencil density, and pencil colour -- and grows fresh
hatching strokes along that direction field at the target resolution. The pore
is re-drawn as a vector shape and the paper grain is generated at full
resolution, so nothing in the output is an enlarged pixel.

    uv run scripts/redraw_stoma_background.py            # 300 ppi, 14400 x 10800
    uv run scripts/redraw_stoma_background.py --ppi 100  # quick preview
"""
import argparse
from pathlib import Path

import numpy as np
import skia
from PIL import Image
from scipy import ndimage as ndi
from scipy.interpolate import UnivariateSpline

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'outputs/harlan_2026/assets'
SOURCE = ASSETS / 'leaf-stoma-pencil.png'
PAGE_IN = 48.0
PAPER = np.array([236., 243., 230.])
LUMA = np.array([.299, .587, .114])
Image.MAX_IMAGE_PIXELS = None


def load():
    src = np.asarray(Image.open(SOURCE).convert('RGB')).astype(np.float64)
    ink = np.clip(PAPER @ LUMA - src @ LUMA, 0, None)
    chroma = np.linalg.norm(src - PAPER, axis=2)
    return src, np.maximum(ink, chroma * .6)


def pore_mask(src):
    dark = src @ LUMA < 125
    lab, n = ndi.label(dark)
    sizes = ndi.sum(dark, lab, range(1, n + 1))
    return lab == np.argmax(sizes) + 1


class Fields:
    """Direction, density and colour of the draft's pencil work, in source px."""

    def __init__(self, src, ink, pore):
        near_pore = ndi.binary_dilation(pore, iterations=2)
        ink = np.where(near_pore, 0, ink)
        self.ink = ink
        g = ndi.gaussian_filter(ink, 1.0)
        gy, gx = np.gradient(g)
        jxx = ndi.gaussian_filter(gx * gx, 3.0)
        jyy = ndi.gaussian_filter(gy * gy, 3.0)
        jxy = ndi.gaussian_filter(gx * gy, 3.0)
        # Doubled-angle representation of the stroke (not gradient) direction.
        self.c2 = -(jxx - jyy)
        self.s2 = -2 * jxy
        mag = np.hypot(self.c2, self.s2)
        self.coherence = mag / (jxx + jyy + 1e-6)
        self.density = ndi.gaussian_filter(ink, 1.2)
        self.sharp = ndi.gaussian_filter(ink, .7)
        self.ridge = ndi.gaussian_filter(ink, 1.6)
        self.ry, self.rx = np.gradient(self.ridge)
        w = ink ** 2 + 1e-3
        self.col = np.stack([ndi.gaussian_filter(src[..., k] * w, 1.5) for k in range(3)], -1) \
            / ndi.gaussian_filter(w, 1.5)[..., None]

    def at(self, arr, pts):
        return ndi.map_coordinates(arr, [pts[:, 1] - .5, pts[:, 0] - .5], order=1, mode='nearest')

    def tangent(self, pts):
        ang = .5 * np.arctan2(self.at(self.s2, pts), self.at(self.c2, pts))
        return np.c_[np.cos(ang), np.sin(ang)]


def seeds(f, n, rng):
    # Seed on the draft's darkest pixels so strands land on its strand centres.
    p = f.ink.ravel() ** 1.6
    p /= p.sum()
    idx = rng.choice(p.size, size=n, p=p)
    y, x = np.divmod(idx, f.density.shape[1])
    return np.c_[x, y] + rng.uniform(0, 1, (n, 2))


def grow(f, start, direction, max_len, step=.4):
    """Integrate streamlines of the direction field from start points."""
    n = len(start)
    pts = [start.copy()]
    pos = start.copy()
    prev = direction.copy()
    alive = np.ones(n, bool)
    travelled = np.zeros(n)
    d0 = f.at(f.density, start)
    for _ in range(int(max_len.max() / step)):
        t = f.tangent(pos)
        flip = np.sum(t * prev, axis=1) < 0
        t[flip] *= -1
        # Pull gently onto the nearest pencil ridge so strands cling to lines.
        nrm = np.c_[-t[:, 1], t[:, 0]]
        grad = np.c_[f.at(f.rx, pos), f.at(f.ry, pos)]
        pull = np.sum(grad * nrm, axis=1, keepdims=True) * nrm
        pull /= np.maximum(f.at(f.ridge, pos)[:, None], 8)
        move = t * step + np.clip(pull * .35, -.035, .035)
        turn = np.sum(t * prev, axis=1)
        new = pos + move * alive[:, None]
        dens = f.at(f.density, new)
        alive &= (turn > .9) & (dens > d0 * .22) & (travelled < max_len) \
            & (new[:, 0] > -2) & (new[:, 0] < f.ink.shape[1] + 2) \
            & (new[:, 1] > -2) & (new[:, 1] < f.ink.shape[0] + 2)
        pos = np.where(alive[:, None], new, pos)
        travelled += step * alive
        prev = np.where(alive[:, None], t, prev)
        pts.append(pos.copy())
        if not alive.any():
            break
    return np.stack(pts, 1), travelled


def make_strokes(f, n, rng):
    start = seeds(f, n, rng)
    coh = np.clip(f.at(f.coherence, start), 0, 1)
    dens = f.at(f.density, start)
    # Long strands on coherent contour lines, short hatches in scribbly zones.
    length = rng.uniform(.6, 1.4, n) * (10 + 55 * coh ** 1.5) * (1 + .6 * (dens > 60))
    t0 = f.tangent(start)
    fwd, lf = grow(f, start, t0, length * rng.uniform(.3, .7, n))
    bwd, lb = grow(f, start, -t0, length - lf)
    strokes = []
    for i in range(n):
        a = fwd[i, :int(np.ceil(lf[i] / .4)) + 1]
        b = bwd[i, 1:int(np.ceil(lb[i] / .4)) + 1][::-1]
        line = np.r_[b, a]
        if len(line) < 6:
            continue
        strokes.append(snap(f, line))
    return strokes


def snap(f, line):
    """Slide a stroke sideways onto the draft's strand centre (sub-pixel peak
    of the ink profile across the stroke), smoothed along its length."""
    d = np.gradient(line, axis=0)
    nrm = np.c_[-d[:, 1], d[:, 0]]
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    v0, vm, vp = (f.at(f.sharp, line + o * nrm) for o in (0, -1, 1))
    curv = vm - 2 * v0 + vp
    off = np.where(curv < -1e-3, .5 * (vm - vp) / np.minimum(curv, -1e-3), 0)
    off = np.clip(off, -1, 1)
    k = np.hanning(23)[1:-1]
    k /= k.sum()
    off = np.convolve(np.r_[off[:1].repeat(10), off, off[-1:].repeat(10)], k, 'valid')
    return line + nrm * off[:, None] * .85


def handify(line, rng):
    """Make a streamline look drawn by hand at arm's length: slightly faceted
    (the hand moves in short straight-ish pushes), a slow tremor across the
    stroke, and the odd overshoot past the line's end."""
    line = line[np.r_[True, np.any(np.abs(np.diff(line, axis=0)) > 1e-6, axis=1)]]
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(line, axis=0).T))]
    length = d[-1]
    if length < 2:
        return line
    s = np.linspace(0, length, max(8, int(length / .5)))
    ctrl_s = np.linspace(0, length, max(2, int(round(length / rng.uniform(4, 8))) + 1))
    ctrl = np.c_[np.interp(ctrl_s, d, line[:, 0]), np.interp(ctrl_s, d, line[:, 1])]
    pts = np.c_[np.interp(s, ctrl_s, ctrl[:, 0]), np.interp(s, ctrl_s, ctrl[:, 1])]
    pts = ndi.uniform_filter1d(pts, 3, axis=0, mode='nearest')
    tang = np.gradient(pts, axis=0)
    nrm = np.c_[-tang[:, 1], tang[:, 0]]
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    tremor = ndi.gaussian_filter1d(rng.standard_normal(len(s)), 14, mode='nearest')
    pts = pts + nrm * (tremor / (tremor.std() + 1e-9) * .3)[:, None]
    for end in (0, -1):
        if rng.random() < .35:
            j = 1 if end == 0 else -2
            u = pts[end] - pts[j]
            u /= max(np.linalg.norm(u), 1e-9)
            extra = pts[end] + u * np.linspace(.5, rng.uniform(1, 3), 5)[:, None]
            pts = np.r_[extra[::-1], pts] if end == 0 else np.r_[pts, extra]
    return pts


def stroke_style(f, line, rng):
    mid = line[len(line) // 2][None]
    obs = np.array([ndi.map_coordinates(f.col[..., k], [mid[:, 1] - .5, mid[:, 0] - .5], order=1)[0]
                    for k in range(3)])
    dens = f.at(f.density, line).mean()
    # The draft's strands are blurred into the paper; restore pencil strength.
    k = np.clip(dens / 90, .15, 1)
    col = PAPER + (obs - PAPER) * rng.uniform(1.0, 1.35)
    col += rng.normal(0, 3) * np.array([1., 1., .4]) + rng.normal(0, 1.5, 3)
    alpha = .35 + .6 * k ** .8
    return np.clip(col, 0, 255) / 255, alpha, k


def ribbon(pts, widths):
    d = np.gradient(pts, axis=0)
    nrm = np.c_[-d[:, 1], d[:, 0]]
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    poly = np.r_[pts + nrm * widths[:, None] / 2, (pts - nrm * widths[:, None] / 2)[::-1]]
    return skia.Path.Polygon([skia.Point(*p) for p in poly], True)


def smooth(line):
    line = line[np.r_[True, np.any(np.abs(np.diff(line, axis=0)) > 1e-6, axis=1)]]
    if len(line) < 5:
        return line
    k = np.ones(5) / 5
    pad = np.r_[line[:1].repeat(2, 0), line, line[-1:].repeat(2, 0)]
    return np.c_[np.convolve(pad[:, 0], k, 'valid'), np.convolve(pad[:, 1], k, 'valid')]


def graphite_texture(ppi, rng, size=1024):
    """Tileable alpha texture of graphite on coarse paper: streaks along the
    stroke (x axis) where the lead drags, flecks where it skips the tooth."""
    mm = ppi / 25.4
    streak = ndi.gaussian_filter(rng.standard_normal((size, size)), (.07 * mm, 2.4 * mm), mode='wrap')
    fleck = ndi.gaussian_filter(rng.standard_normal((size, size)), .07 * mm, mode='wrap')
    g = .8 * streak / streak.std() + .45 * fleck / fleck.std()
    g /= g.std()
    alpha = np.clip((g + .35) / 1.5, 0, 1) ** .9
    return skia.Image.fromarray(np.ascontiguousarray((alpha * 255).astype(np.uint8)),
                                colorType=skia.kAlpha_8_ColorType)


def graphite(texture, pts, rng):
    """Shader laying the texture along the stroke's overall direction."""
    d = pts[-1] - pts[0]
    m = skia.Matrix()
    m.setRotate(float(np.degrees(np.arctan2(d[1], d[0]))))
    m.postTranslate(*rng.uniform(0, 4096, 2))
    return texture.makeShader(skia.TileMode.kRepeat, skia.TileMode.kRepeat,
                              skia.SamplingOptions(skia.FilterMode.kLinear), m)


def draw_strokes(canvas, f, strokes, scale, texture, rng):
    for line in strokes:
        col, alpha, k = stroke_style(f, line, rng)
        pts = handify(line, rng)
        if rng.random() < .5:
            pts = pts[::-1]
        m = len(pts)
        t = np.linspace(0, 1, m)
        # Pencil pressure: lands firm, lifts off in a tapering flick.
        press = np.clip(t / .05, 0, 1) ** .5 * np.clip((1 - t) / rng.uniform(.2, .45), 0, 1) ** .8
        press = .12 + .88 * press
        press *= 1 + .15 * np.sin(t * rng.uniform(2, 5) + rng.uniform(0, 6.3))
        # A big blunt pencil: ~1-1.7 mm at print, heavier on dense contour lines.
        w = scale * (.9 + 1.0 * k) * press * rng.uniform(.85, 1.15)
        pts = pts * scale
        shader = graphite(texture, pts, rng)
        canvas.drawPath(ribbon(pts, w), skia.Paint(
            AntiAlias=True, Color4f=skia.Color4f(*col, alpha * .75), Shader=shader))
        # Graphite builds up in streaks along the lead's edges.
        dark = np.clip(col * .88, 0, 1)
        d = np.gradient(pts, axis=0)
        nrm = np.c_[-d[:, 1], d[:, 0]]
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
        for side in rng.choice([-1, 1], size=rng.integers(1, 3), replace=False):
            off = side * rng.uniform(.15, .32)
            canvas.drawPath(ribbon(pts + nrm * (w * off)[:, None], w * rng.uniform(.18, .3)),
                            skia.Paint(AntiAlias=True, Color4f=skia.Color4f(*dark, alpha * .35),
                                       Shader=shader))


def pore_path(pore, scale):
    """Smooth vector pore: sub-pixel top/bottom edges per column of the draft,
    refit as a smooth centre line plus half-thickness so the tips stay sharp."""
    soft = ndi.gaussian_filter(pore.astype(float), .8)
    cols = np.nonzero(pore.any(0))[0]
    xs, top, bot = [], [], []
    for x in cols:
        v = soft[:, x]
        inside = np.nonzero(v >= .5)[0]
        if len(inside) == 0:
            continue
        a, b = inside[0], inside[-1]
        # Linear interpolation of the 0.5 crossing on each side.
        ya = a - (v[a] - .5) / max(v[a] - v[a - 1], 1e-6)
        yb = b + (v[b] - .5) / max(v[b] - v[b + 1], 1e-6)
        xs.append(x + .5); top.append(ya + .5); bot.append(yb + .5)
    xs, top, bot = map(np.array, (xs, top, bot))
    x0, x1 = xs[0] - .5, xs[-1] + .5
    u = (xs - x0) / (x1 - x0)
    centre = UnivariateSpline(xs, (top + bot) / 2, s=len(xs) * .15)
    # Half-thickness modelled as sqrt-free profile h = u^a (1-u)^b * poly(u).
    half = np.maximum((bot - top) / 2, 0)
    hs = UnivariateSpline(xs, half, s=len(xs) * .12)
    t = np.linspace(0, 1, 3000)
    xx = x0 + t * (x1 - x0)
    env = np.clip(np.minimum(t, 1 - t) / .015, 0, 1) ** .5
    h = np.maximum(hs(xx), 0) * env
    c = centre(xx)
    poly = np.r_[np.c_[xx, c - h], np.c_[xx, c + h][::-1]] * scale
    return skia.Path.Polygon([skia.Point(*p) for p in poly], True)


def base_wash(src, ink, pore):
    """Soft tint under the pencil work: the draft with strands lifted out,
    plus a faint graphite smudge so dense zones keep their tone."""
    strands = ink > 14
    keep = (~strands & ~pore).astype(float)
    num = np.stack([ndi.gaussian_filter(src[..., k] * keep, 2.5) for k in range(3)], -1)
    den = ndi.gaussian_filter(keep, 2.5)[..., None]
    lifted = np.where(den > 1e-3, num / np.maximum(den, 1e-3), PAPER)
    srcp = np.where(pore[..., None], PAPER, src)
    smudge = np.stack([ndi.gaussian_filter(srcp[..., k], 3.0) for k in range(3)], -1)
    return PAPER + (lifted - PAPER) * .8 + (smudge - PAPER) * .22


def noise(shape, sigma, rng):
    n = ndi.gaussian_filter(rng.standard_normal(shape).astype(np.float32), sigma)
    return n / (n.std() + 1e-9)


def paper_noise(white, y0, y1, sigma):
    """Band of a seamless blurred-noise field (unit std), filtered with margins."""
    m = int(4 * sigma) + 2
    a, b = max(0, y0 - m), min(len(white), y1 + m)
    n = ndi.gaussian_filter(white[a:b], sigma)[y0 - a:y0 - a + (y1 - y0)]
    return n * (2 * np.sqrt(np.pi) * max(sigma, .5))


def upsample(field, out_hw):
    h, w = out_hw
    chans = field if field.ndim == 3 else field[..., None]
    out = np.empty((h, w, chans.shape[2]), np.float32)
    for k in range(chans.shape[2]):
        img = Image.fromarray(chans[..., k].astype(np.float32), mode='F')
        out[..., k] = np.asarray(img.resize((w, h), Image.BICUBIC))
    return out if field.ndim == 3 else out[..., 0]


def render(ppi, out_path, density=1.0, seed=7):
    rng = np.random.default_rng(seed)
    src, ink = load()
    sh, sw = ink.shape
    W, H = int(round(PAGE_IN * ppi)), int(round(PAGE_IN * ppi * sh / sw))
    scale = W / sw
    pore = pore_mask(src)
    f = Fields(src, ink, pore)
    n = int(f.density.sum() / 620 * density)
    strokes = make_strokes(f, n, rng)
    print(f'{W}x{H} px, scale {scale:.2f}, {len(strokes)} strokes')

    base = upsample(base_wash(src, ink, pore).astype(np.float32), (H, W))

    surf = skia.Surface(W, H)
    c = surf.getCanvas()
    c.clear(skia.ColorTRANSPARENT)
    c.drawPath(pore_path(pore, scale), skia.Paint(AntiAlias=True, Color=skia.ColorBLACK))
    pore_a = surf.makeImageSnapshot().toarray()[..., 3].copy()
    c.clear(skia.ColorTRANSPARENT)
    draw_strokes(c, f, strokes, scale, graphite_texture(ppi, rng), rng)
    # Thin graphite edge around the pore.
    edge = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=scale * .55,
                      Color4f=skia.Color4f(.22, .30, .22, .8))
    c.drawPath(pore_path(pore, scale), edge)
    strands = surf.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType,
                                               alphaType=skia.kUnpremul_AlphaType)
    del surf

    pore_col = np.median(src[pore], axis=0)
    out = np.empty((H, W, 3), np.uint8)
    mottle = upsample(noise((sh // 2, sw // 2), 6.0, rng), (H, W))
    white = rng.standard_normal((H, W), dtype=np.float32)
    band = 1200
    for y0 in range(0, H, band):
        y1 = min(H, y0 + band)
        # Paper grain sized physically: ~0.05 mm fibres, ~0.12 mm tooth.
        fine = paper_noise(white, y0, y1, .05 / 25.4 * ppi)
        tooth = paper_noise(white, y0, y1, .12 / 25.4 * ppi)
        img = base[y0:y1].copy()
        img += (fine * 1.0 + tooth * .8)[..., None]
        pa = pore_a[y0:y1, :, None].astype(np.float32) / 255
        pcol = pore_col + mottle[y0:y1, :, None] * 1.2 + (tooth * 1.8 + fine * 1.2)[..., None]
        img = img * (1 - pa) + pcol * pa
        s = strands[y0:y1].astype(np.float32)
        sa = s[..., 3] / 255
        # Paper tooth is shared by every stroke: light pencil skips its valleys,
        # heavy build-up gradually fills them.
        valley = paper_noise(white, y0, y1, .1 / 25.4 * ppi)
        sa = (sa * np.clip((valley + .7 + 1.1 * sa) / 1.2, 0, 1))[..., None]
        img = img * (1 - sa) + s[..., :3] * sa
        out[y0:y1] = np.clip(img + .5, 0, 255).astype(np.uint8)
    im = Image.fromarray(out)
    if out_path.suffix.lower() in ('.jpg', '.jpeg'):
        im.save(out_path, quality=94, subsampling=0, dpi=(ppi, ppi))
    else:
        im.save(out_path, dpi=(ppi, ppi))
    print(out_path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ppi', type=int, default=300)
    ap.add_argument('--density', type=float, default=1.0)
    ap.add_argument('--out', type=Path, default=ASSETS / 'leaf-stoma-pencil-300ppi.jpg')
    args = ap.parse_args()
    render(args.ppi, args.out, args.density)
