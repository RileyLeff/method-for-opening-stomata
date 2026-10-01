from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import font_manager
from matplotlib.lines import Line2D
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph
from scipy.stats import gaussian_kde

from plot_phrag_brownsville_capacity_envelopes_literature import build_draws_and_summary


ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path('/Users/rileyleff/Documents/dev/riley-blog-content/content/forcing-plants-to-participate/assets')
TMP = ROOT / 'tmp/harlan_2026'
OUT = ROOT / 'outputs/harlan_2026'
PDF_OUT = ROOT / 'output/pdf'
FONTS = ROOT / 'tmp/pdfs/fonts'
INK = '#172C31'
SECONDARY = '#52646A'
BLUE = '#2676A8'
ORANGE = '#D36B2A'
CYAN = '#268B89'
RULE = '#DDE4E5'
PAGE_W, PAGE_H = 48 * 72, 36 * 72


def fonts() -> None:
    for path in FONTS.glob('*.ttf'):
        font_manager.fontManager.addfont(path)
    for name, filename in {
        'PlexSans': 'IBMPlexSans-Regular.ttf',
        'PlexSansBold': 'IBMPlexSans-SemiBold.ttf',
        'PlexSansItalic': 'IBMPlexSans-Italic.ttf',
        'PlexSansBoldItalic': 'IBMPlexSans-SemiBoldItalic.ttf',
        'PlexSerif': 'IBMPlexSerif-Regular.ttf',
        'PlexSerifBold': 'IBMPlexSerif-Bold.ttf',
    }.items():
        pdfmetrics.registerFont(TTFont(name, str(FONTS / filename)))
    pdfmetrics.registerFontFamily('PlexSans', normal='PlexSans', bold='PlexSansBold',
                                 italic='PlexSansItalic', boldItalic='PlexSansBoldItalic')
    mpl.rcParams.update({
        'font.family': 'IBM Plex Sans', 'font.size': 28, 'text.color': INK,
        'axes.labelcolor': INK, 'axes.edgecolor': SECONDARY,
        'xtick.color': SECONDARY, 'ytick.color': SECONDARY,
        'axes.linewidth': 1.4, 'pdf.fonttype': 42,
        'mathtext.fontset': 'dejavusans', 'savefig.facecolor': 'white',
    })


def save_plot(fig: plt.Figure, stem: str) -> None:
    fig.savefig(TMP / f'{stem}.pdf')
    fig.savefig(TMP / f'{stem}.png', dpi=190)
    plt.close(fig)


def method_plot() -> None:
    data = json.loads((ASSETS / 'low-co2-opening.json').read_text())
    df = pd.DataFrame(data['records'])
    fig, axes = plt.subplots(2, 1, figsize=(20.3, 12.8), sharex=True)
    fig.subplots_adjust(left=.135, right=.985, bottom=.115, top=.89, hspace=.23)
    for ax in axes:
        for stage, color in zip(data['stages'], [BLUE, BLUE, ORANGE, BLUE]):
            ax.axvspan(stage['start'], stage['end'], color=color,
                       alpha=.055 if stage['co2'] == 420 else .10, linewidth=0)
        for minute in [30, 60, 90]:
            ax.axvline(minute, color=SECONDARY, lw=1.3, ls=(0, (5, 5)), alpha=.5)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color=RULE, lw=1)
        ax.tick_params(labelsize=27, length=7, pad=9)
        ax.set_axisbelow(True)
    axes[0].plot(df.minute, df.gsw, color=BLUE, lw=4.5, marker='o', ms=5)
    axes[1].plot(df.minute, df.a, color=CYAN, lw=4.5, marker='o', ms=5)
    axes[0].set_ylim(-.015, .25)
    axes[0].set_yticks([0, .1, .2])
    axes[0].set_ylabel('Stomatal openness, $g_{sw}$\n(mol H$_2$O m$^{-2}$ s$^{-1}$)', fontsize=29, labelpad=19)
    axes[1].set_ylim(0, 17.3)
    axes[1].set_yticks([0, 5, 10, 15])
    axes[1].set_ylabel('Carbon assimilation, $A$\n' + r'($\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$)', fontsize=29, labelpad=19)
    axes[1].set_xlim(0, 120)
    axes[1].set_xticks([0, 30, 60, 90, 120])
    axes[1].set_xlabel('Elapsed time (min)', fontsize=30, labelpad=13)
    for stage in data['stages']:
        axes[0].text((stage['start'] + stage['end']) / 2, 1.065,
                     f"{stage['co2']} ppm", transform=axes[0].get_xaxis_transform(),
                     ha='center', va='bottom', fontsize=31, fontweight='bold')
    save_plot(fig, 'method')


def surface_plot() -> None:
    data = json.loads((ASSETS / 'phrag-p10-surface.json').read_text())
    ci, q = np.meshgrid(data['ci'], data['q'])
    a = np.asarray(data['assimilation'])
    fig = plt.figure(figsize=(23.8, 19.7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(ci, q, a, cmap='viridis', vmin=0, vmax=26,
                    rcount=len(data['q']), ccount=len(data['ci']),
                    linewidth=0, antialiased=True, alpha=.94, rasterized=True)
    points = pd.DataFrame(data['points'])
    ax.scatter(points.ci, points.q, points.a, c=points.a,
               cmap='viridis', vmin=0, vmax=26, s=42, alpha=.60,
               edgecolors='white', linewidths=.4, depthshade=False)
    ax.set_xlim(0, max(data['ci']))
    ax.set_ylim(0, 2000)
    ax.set_zlim(0, 27)
    ax.set_xticks([0, 400, 800, 1200])
    ax.set_yticks([0, 500, 1000, 1500, 2000])
    ax.set_zticks([0, 10, 20])
    ax.set_xlabel(r'$C_i$ ($\mu$mol mol$^{-1}$)', fontsize=32, labelpad=38)
    ax.set_ylabel(r'$Q$ ($\mu$mol photons m$^{-2}$ s$^{-1}$)', fontsize=32, labelpad=44)
    ax.set_zlabel(r'$A$ ($\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$)', fontsize=32, labelpad=34)
    ax.tick_params(labelsize=28, pad=12)
    ax.view_init(elev=24, azim=-135)
    ax.set_box_aspect((1.1, 1, .82))
    ax.set_proj_type('persp', focal_length=.98)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((.94, .96, .96, .36))
        axis._axinfo['grid']['color'] = mpl.colors.to_rgba(RULE, .8)
        axis._axinfo['grid']['linewidth'] = 1.0
    fig.subplots_adjust(left=.025, right=.97, bottom=.06, top=1.0)
    save_plot(fig, 'surface')


def capacity_plot() -> None:
    tree = xr.open_datatree(ROOT / 'outputs/phrag_brownsville_tier_c_posterior.nc')
    draws, summary = build_draws_and_summary(tree)
    summary.to_csv(OUT / 'capacity_summary.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(20.3, 7.6))
    fig.subplots_adjust(left=.035, right=.985, bottom=.29, top=.72, wspace=.24)
    for ax, parameter, limit, name in zip(
        axes, ['Vcmax25', 'Jmax25'], [120, 220],
        ['Carboxylation capacity', 'Electron-transport capacity'],
    ):
        grid = np.linspace(0, limit, 650)
        for zone, color in [('High', BLUE), ('Low', ORANGE)]:
            d = gaussian_kde(draws[parameter][zone])(grid)
            ax.fill_between(grid, 0, d, color=color, alpha=.22, linewidth=0)
            ax.plot(grid, d, color=color, lw=4)
        ax.set_xlim(0, limit)
        ax.set_ylim(bottom=0)
        ax.set_yticks([])
        ax.set_title(name, fontsize=31, fontweight='bold', pad=22)
        label = ('$V_{cmax,25}$' if parameter == 'Vcmax25' else '$J_{max,25}$')
        units = r'$\mu$mol m$^{-2}$ s$^{-1}$' if parameter == 'Vcmax25' else r'$\mu$mol e$^{-}$ m$^{-2}$ s$^{-1}$'
        ax.set_xlabel(f'{label} ({units})', fontsize=29, labelpad=16)
        ax.tick_params(labelsize=26, pad=10, length=7)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.grid(axis='x', color=RULE, lw=1)
    legend = [Line2D([0], [0], color=BLUE, lw=6, label='Low salinity (n = 3)'),
              Line2D([0], [0], color=ORANGE, lw=6, label='High salinity (n = 5)')]
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, 1.01),
               ncol=2, frameon=False, fontsize=27, handlelength=1.8, columnspacing=1.5)
    fig.text(.5, .03, 'Posterior distributions of group means, adjusted for measurement order.',
             ha='center', va='bottom', fontsize=23, color=SECONDARY)
    save_plot(fig, 'capacity')
    tree.close()


def layout() -> dict:
    return {
        'widthInches': 48, 'heightInches': 36,
        'background': '#FFFFFF',
        'text': [
            {'text': 'Forcing Plants to Participate', 'x': 1.05, 'y': .78, 'w': 46, 'h': 1.8,
             'size': 103, 'font': 'PlexSerifBold', 'bold': True},
            {'text': 'Riley Leff, Acadia Conboy, Keryn Gedan', 'x': 1.08, 'y': 2.65, 'w': 45, 'h': .75,
             'size': 39, 'font': 'PlexSans'},
            {'text': 'Department of Biological Sciences, George Washington University',
             'x': 1.08, 'y': 3.55, 'w': 45, 'h': .62, 'size': 29, 'font': 'PlexSans', 'color': SECONDARY},
            {'text': "If a leaf isn't photosynthesizing, how can we tell if it's incapable or unwilling?",
             'x': 1.08, 'y': 4.85, 'w': 45, 'h': .85, 'size': 39, 'font': 'PlexSans'},
            {'text': 'Lower CO2 keeps stomata opening.', 'x': 1.08, 'y': 7.1, 'w': 20.5, 'h': .8,
             'size': 38, 'font': 'PlexSansBold', 'bold': True},
            {'text': 'Morella cerifera', 'x': 1.08, 'y': 8.06, 'w': 20, 'h': .65,
             'size': 30, 'font': 'PlexSansBoldItalic', 'bold': True, 'italic': True},
            {'text': 'Illustrative run at constant Q = 800. Returning to 420 ppm restores A.',
             'x': 1.08, 'y': 21.97, 'w': 20.5, 'h': .95, 'size': 26, 'font': 'PlexSans', 'color': SECONDARY},
            {'text': 'No detectable salinity difference.', 'x': 1.08, 'y': 23.9, 'w': 20.5, 'h': .8,
             'size': 38, 'font': 'PlexSansBold', 'bold': True},
            {'text': 'The capacity underneath is still there.', 'x': 23.2, 'y': 7.1, 'w': 23.7, 'h': .8,
             'size': 38, 'font': 'PlexSansBold', 'bold': True},
            {'text': 'Phragmites australis', 'x': 23.2, 'y': 8.06, 'w': 23, 'h': .65,
             'size': 30, 'font': 'PlexSansBoldItalic', 'bold': True, 'italic': True},
            {'text': 'CO2 ramps across light levels, with chlorophyll fluorescence.',
             'x': 23.2, 'y': 29.1, 'w': 23.7, 'h': .65, 'size': 28, 'font': 'PlexSans'},
            {'text': '12 leaves. ~60,000 gas-exchange observations. ~600 flashes.',
             'x': 23.2, 'y': 30.1, 'w': 23.7, 'h': 1.1, 'size': 30, 'font': 'PlexSansBold', 'bold': True},
            {'text': 'Brownsville Preserve, Eastern Shore of Virginia.\nEight sun leaves in the salinity comparison; four additional shade pilots.',
             'x': 23.2, 'y': 31.63, 'w': 23.7, 'h': 1.15, 'size': 25, 'font': 'PlexSans', 'color': SECONDARY},
            {'text': 'Methods: Stinziano et al. 2017 (10.1111/pce.12911); Saathoff & Welles 2021 (10.1111/pce.14178).',
             'x': 1.08, 'y': 34.42, 'w': 45.8, 'h': .5, 'size': 20, 'font': 'PlexSans', 'color': SECONDARY},
            {'text': 'rileyleff.com/blog/forcing-plants-to-participate',
             'x': 1.08, 'y': 35.01, 'w': 45.8, 'h': .45, 'size': 20, 'font': 'PlexSans', 'color': SECONDARY},
        ],
        'figures': [
            {'stem': 'method', 'x': 1.08, 'y': 8.9, 'w': 20.3, 'h': 12.8},
            {'stem': 'capacity', 'x': 1.08, 'y': 25.07, 'w': 20.3, 'h': 7.6},
            {'stem': 'surface', 'x': 23.2, 'y': 8.9, 'w': 23.8, 'h': 19.7},
        ],
    }


def poster_pdf(spec: dict) -> None:
    base = TMP / 'poster_base.pdf'
    c = Canvas(str(base), pagesize=(PAGE_W, PAGE_H))
    c.setTitle('Forcing Plants to Participate')
    c.setAuthor('Riley Leff, Acadia Conboy, Keryn Gedan')
    for item in spec['text']:
        style = ParagraphStyle('Poster', fontName=item['font'], fontSize=item['size'],
                               leading=item['size'] * 1.18, textColor=HexColor(item.get('color', INK)))
        text = html.escape(item['text']).replace('\n', '<br/>')
        text = text.replace('CO2', 'CO<sub>2</sub>')
        p = Paragraph(text, style)
        _, height = p.wrap(item['w'] * 72, item['h'] * 72)
        if height > item['h'] * 72 + 1:
            raise RuntimeError(f"Text overflow: {item['text']} ({height:.1f})")
        p.drawOn(c, item['x'] * 72, PAGE_H - item['y'] * 72 - height)
    c.showPage()
    c.save()
    writer = PdfWriter()
    page = writer.add_page(PdfReader(base).pages[0])
    for item in spec['figures']:
        src = PdfReader(TMP / f"{item['stem']}.pdf").pages[0]
        scale = item['w'] * 72 / float(src.mediabox.width)
        transform = Transformation().scale(scale).translate(
            item['x'] * 72, PAGE_H - (item['y'] + item['h']) * 72)
        page.merge_transformed_page(src, transform)
    writer.add_metadata({'/Title': 'Forcing Plants to Participate',
                         '/Author': 'Riley Leff, Acadia Conboy, Keryn Gedan'})
    path = PDF_OUT / 'forcing-plants-to-participate-harlan-poster.pdf'
    with path.open('wb') as stream:
        writer.write(stream)
    print(path)


def abstract_pdf() -> None:
    source = (OUT / 'abstract.md').read_text()
    sections = source.strip().split('\n\n')
    paragraphs = sections[3:]
    word_count = len(' '.join(paragraphs).split())
    if word_count > 250:
        raise RuntimeError(f'Abstract exceeds 250 words: {word_count}')
    path = PDF_OUT / 'forcing-plants-to-participate-harlan-abstract.pdf'
    c = Canvas(str(path), pagesize=(612, 792))
    c.setTitle('Forcing Plants to Participate - Abstract')
    c.setAuthor('Riley Leff, Acadia Conboy, Keryn Gedan')
    c.setFillColor(HexColor(INK))
    c.setFont('PlexSerifBold', 25)
    c.drawString(48, 733, 'Forcing Plants to Participate')
    c.setFont('PlexSans', 12)
    c.drawString(48, 704, sections[1])
    c.setFont('PlexSans', 10.5)
    c.setFillColor(HexColor(SECONDARY))
    c.drawString(48, 683, sections[2])
    style = ParagraphStyle('Abstract', fontName='PlexSans', fontSize=12,
                           leading=18.6, textColor=HexColor(INK), spaceAfter=14)
    y = 644
    for paragraph in paragraphs:
        markup = html.escape(paragraph).replace('CO2', 'CO<sub>2</sub>')
        markup = markup.replace('Phragmites australis', '<i>Phragmites australis</i>')
        p = Paragraph(markup, style)
        _, height = p.wrap(516, 600)
        y -= height
        p.drawOn(c, 48, y)
        y -= 14
    assert y > 48, f'Abstract overflowed: {y}'
    c.showPage()
    c.save()
    (OUT / 'abstract.txt').write_text('\n\n'.join(paragraphs) + '\n')
    (OUT / 'layout.json').write_text(json.dumps(layout(), indent=2) + '\n')
    print(f'{path}\nAbstract: {word_count} words')


def main() -> None:
    for path in [TMP, OUT, PDF_OUT]:
        path.mkdir(parents=True, exist_ok=True)
    fonts()
    method_plot()
    surface_plot()
    capacity_plot()
    poster_pdf(layout())
    abstract_pdf()


if __name__ == '__main__':
    main()
