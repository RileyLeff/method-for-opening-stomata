"""Vertical stretch of the leaf drawing on the 48 x 36 in Harlan poster.

The original layout kept a ~5 in logo strip under the leaf. With the logos moved
to the header, the leaf grows down into that strip: the epidermal cells stretch
~25 % while the stomatal pore band stretches only ~5 %, so the guard cells keep
their shape. Positions are in poster inches from the top edge.
"""
import numpy as np

PAGE_H = 36.0
# (source-inch breakpoint, local stretch factor below it). Steps fall in blank
# paper or are eased, so drawn lines never see a kink.
_TOP, _PORE0, _PORE1, _BOTTOM = 5.5, 14.5, 21.0, 30.6
_CELL, _PORE = 1.254, 1.05
_EASE = .7

_ys = np.linspace(0, PAGE_H, 36001)


def _stretch(y):
    def step(edge, width=0.0):
        if width == 0:
            return (y >= edge).astype(float)
        return 1 / (1 + np.exp(-(y - edge) / (width / 4)))
    inside = step(_TOP) * (1 - step(_BOTTOM))
    pore = step(_PORE0, _EASE) * (1 - step(_PORE1, _EASE))
    s = 1 + inside * ((_CELL - 1) + pore * (_PORE - _CELL))
    # Blank paper below the drawing absorbs the extra height.
    below = y >= _BOTTOM
    used = np.trapezoid(s[~below], y[~below])
    s[below] = (PAGE_H - used) / (PAGE_H - _BOTTOM)
    return s


_out = np.concatenate([[0], np.cumsum((_stretch(_ys)[1:] + _stretch(_ys)[:-1]) / 2 * np.diff(_ys))])


def to_out(y_src):
    """Original-layout inches -> stretched-layout inches."""
    return np.interp(y_src, _ys, _out)


def to_src(y_out):
    """Stretched-layout inches -> original-layout inches."""
    return np.interp(y_out, _out, _ys)
