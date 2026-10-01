from __future__ import annotations

import html
import json
import re
import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image
from pypdf import PdfReader
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import LETTER, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph


ROOT = Path(__file__).resolve().parents[1]
CONTENT_DIR = Path(
    "/Users/rileyleff/Documents/dev/riley-blog-content/content/forcing-plants-to-participate"
)
CONTENT_PATH = CONTENT_DIR / "content.mdx"
ASSET_DIR = CONTENT_DIR / "assets"
TMP_DIR = ROOT / "tmp" / "pdfs"
FONT_DIR = TMP_DIR / "fonts"
OUTPUT_DIR = ROOT / "output" / "pdf"
OUTPUT_PATH = OUTPUT_DIR / "forcing-plants-to-participate-hidden-science.pdf"

PAPER = "#FDF6E3"
PAPER_ALT = "#EEE8D5"
INK = "#073642"
INK_SECONDARY = "#586E75"
INK_MUTED = "#93A1A1"
BLUE = "#268BD2"
CYAN = "#2AA198"
ORANGE = "#CB4B16"
RULE = "#D3CBB7"

LINKS = {
    "Nguyen": "https://doi.org/10.1093/aobpla/plt016",
    "coastal": "https://doi.org/10.1111/gcb.16297",
    "death": "https://doi.org/10.1038/s43017-022-00272-1",
    "Stinziano": "https://doi.org/10.1111/pce.12911",
    "Saathoff": "https://doi.org/10.1111/pce.14178",
}

CAPTION_ONE_LEAD = "Demoing our method on a Morella cerifera leaf."
CAPTION_ONE = (
    "g_sw measures stomatal openness, while A measures carbon assimilation. "
    "Although g_sw is still rising during the initial 420 ppm CO2 period, in repeated "
    "observations, leaves held there generally level off rather than continue to open. "
    "Lowering CO2 keeps g_sw climbing. Returning to 420 ppm then gives us a brief window "
    "to measure A and its response to manipulated conditions before stomata become limiting again."
)
CAPTION_TWO_LEAD = (
    "A full characterization of a Phragmites australis leaf's photosynthetic capacity."
)
CAPTION_TWO = (
    "C_i is intercellular CO2 concentration, and Q is incident light. As Q and C_i increase, "
    "carbon assimilation increases and plateaus. The height and curvature of that response "
    "let us calculate the leaf's photochemical limits."
)

REFERENCES = [
    (
        "Nguyen et al. (2013). Photosynthesis of co-existing Phragmites haplotypes in their "
        "non-native range. AoB PLANTS 5:plt016.",
        LINKS["Nguyen"],
    ),
    (
        "McDowell et al. (2022). Processes and mechanisms of coastal woody-plant mortality. "
        "Global Change Biology 28:5881-5900.",
        LINKS["coastal"],
    ),
    (
        "McDowell et al. (2022). Mechanisms of woody-plant mortality under rising drought, "
        "CO2 and vapour pressure deficit. Nature Reviews Earth & Environment 3:294-308.",
        LINKS["death"],
    ),
    (
        "Stinziano et al. (2017). The rapid A-Ci response: photosynthesis in the phenomic era. "
        "Plant, Cell & Environment 40:1256-1262.",
        LINKS["Stinziano"],
    ),
    (
        "Saathoff & Welles (2021). Gas exchange measurements in the unsteady state. "
        "Plant, Cell & Environment 44:3509-3523.",
        LINKS["Saathoff"],
    ),
]


def register_fonts() -> None:
    font_files = {
        "PlexSans": "IBMPlexSans-Regular.ttf",
        "PlexSansItalic": "IBMPlexSans-Italic.ttf",
        "PlexSansSemibold": "IBMPlexSans-SemiBold.ttf",
        "PlexSansSemiboldItalic": "IBMPlexSans-SemiBoldItalic.ttf",
        "PlexSerif": "IBMPlexSerif-Regular.ttf",
        "PlexSerifBold": "IBMPlexSerif-Bold.ttf",
        "PlexMono": "IBMPlexMono-Regular.ttf",
        "PlexMonoSemibold": "IBMPlexMono-SemiBold.ttf",
    }
    for family, filename in font_files.items():
        path = FONT_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing font: {path}")
        pdfmetrics.registerFont(TTFont(family, str(path)))

    pdfmetrics.registerFontFamily(
        "PlexSans",
        normal="PlexSans",
        bold="PlexSansSemibold",
        italic="PlexSansItalic",
        boldItalic="PlexSansSemiboldItalic",
    )


def extract_essay_markdown() -> list[str]:
    source = CONTENT_PATH.read_text(encoding="utf-8")
    source = re.sub(r"<LowCo2Experiment\b.*?/>\s*", "", source, flags=re.DOTALL)
    source = re.sub(r"<PhotosynthesisSurface\b.*?/>\s*", "", source, flags=re.DOTALL)
    return [paragraph.strip() for paragraph in re.split(r"\n\s*\n", source) if paragraph.strip()]


def markdown_to_reportlab(markdown: str) -> str:
    tokens: list[str] = []

    def stash(value: str) -> str:
        tokens.append(value)
        return f"@@TOKEN{len(tokens) - 1}@@"

    linked = re.sub(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        lambda match: stash(
            f'<link href="{html.escape(match.group(2), quote=True)}" color="{BLUE}">'
            f"<u>{html.escape(match.group(1))}</u></link>"
        ),
        markdown,
    )
    escaped = html.escape(linked)
    escaped = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", escaped)
    for index, token in enumerate(tokens):
        escaped = escaped.replace(f"@@TOKEN{index}@@", token)
    return escaped


def draw_page_background(canvas: Canvas, width: float, height: float) -> None:
    canvas.setFillColor(HexColor(PAPER))
    canvas.rect(0, 0, width, height, stroke=0, fill=1)
    canvas.saveState()
    canvas.setStrokeColor(HexColor(PAPER_ALT))
    canvas.setLineWidth(0.22)
    canvas.setStrokeAlpha(0.45)
    for index in range(24):
        y = 8 + index * (height - 16) / 23
        inset = 4 + (index * 17) % 19
        canvas.line(inset, y, width - inset * 0.72, y + ((index % 3) - 1) * 0.15)
    canvas.restoreState()


def draw_rule(canvas: Canvas, x: float, y: float, width: float) -> None:
    canvas.setStrokeColor(HexColor(RULE))
    canvas.setLineWidth(0.8)
    canvas.line(x, y, x + width * 0.42, y)
    canvas.setStrokeColor(HexColor(PAPER_ALT))
    canvas.line(x + width * 0.42, y, x + width, y)


def fit_essay(paragraphs: list[str], width: float, available_height: float):
    for size in np.arange(9.7, 8.79, -0.05):
        style = ParagraphStyle(
            "Essay",
            fontName="PlexSans",
            fontSize=float(size),
            leading=float(size * 1.42),
            textColor=HexColor(INK),
            alignment=TA_LEFT,
            spaceAfter=7.4,
            allowWidows=0,
            allowOrphans=0,
        )
        rendered = [Paragraph(markdown_to_reportlab(item), style) for item in paragraphs]
        heights = [paragraph.wrap(width, available_height)[1] for paragraph in rendered]
        total = sum(heights) + style.spaceAfter * (len(rendered) - 1)
        if total <= available_height:
            return rendered, heights, style, total
    raise RuntimeError("Essay did not fit on one page")


def draw_page_one(canvas: Canvas, paragraphs: list[str]) -> None:
    width, height = LETTER
    canvas.setPageSize(LETTER)
    draw_page_background(canvas, width, height)

    left = 48
    right = width - 48
    canvas.setFillColor(HexColor(INK))
    canvas.setFont("PlexSerifBold", 26)
    canvas.drawString(left, 742, "Forcing Plants to Participate")
    canvas.setFont("PlexSerif", 11.8)
    canvas.setFillColor(HexColor(INK_SECONDARY))
    canvas.drawString(
        left,
        719,
        "If a leaf isn't photosynthesizing, how can we tell if it's incapable or unwilling?",
    )

    canvas.setFont("PlexSerif", 11.8)
    canvas.setFillColor(HexColor(INK_SECONDARY))
    canvas.drawString(left, 696, "Riley Leff, August 2026")
    draw_rule(canvas, left, 682, right - left)

    body_top = 663
    body_bottom = 39
    body_width = right - left
    available_height = body_top - body_bottom
    rendered, heights, _, total_height = fit_essay(paragraphs, body_width, available_height)
    y = body_top - max(0, (available_height - total_height) * 0.08)
    for index, (paragraph, paragraph_height) in enumerate(zip(rendered, heights, strict=True)):
        y -= paragraph_height
        paragraph.drawOn(canvas, left, y)
        if index != len(rendered) - 1:
            y -= 7.4

    if y < body_bottom - 0.5:
        raise RuntimeError(f"Page-one copy overflowed by {body_bottom - y:.1f} points")
    canvas.showPage()


def add_matplotlib_fonts() -> None:
    for path in FONT_DIR.glob("*.ttf"):
        font_manager.fontManager.addfont(path)
    mpl.rcParams.update(
        {
            "font.family": "IBM Plex Sans",
            "font.size": 8.5,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK_MUTED,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
        }
    )


def render_surface() -> tuple[Path, dict]:
    add_matplotlib_fonts()
    data = json.loads((ASSET_DIR / "phrag-p10-surface.json").read_text(encoding="utf-8"))
    ci = np.asarray(data["ci"], dtype=float)
    q = np.asarray(data["q"], dtype=float)
    assimilation = np.asarray(data["assimilation"], dtype=float)
    ci_grid, q_grid = np.meshgrid(ci, q)

    figure = plt.figure(figsize=(6.6, 4.7), facecolor=PAPER)
    axis = figure.add_subplot(111, projection="3d")
    axis.set_facecolor(PAPER)
    surface = axis.plot_surface(
        ci_grid,
        q_grid,
        assimilation,
        cmap="viridis",
        vmin=-1,
        vmax=26,
        linewidth=0,
        antialiased=True,
        alpha=0.92,
        rcount=len(q),
        ccount=min(len(ci), 100),
    )
    points = data["points"]
    axis.scatter(
        [point["ci"] for point in points],
        [point["q"] for point in points],
        [point["a"] for point in points],
        c=[point["a"] for point in points],
        cmap="viridis",
        vmin=-1,
        vmax=26,
        s=5,
        alpha=0.62,
        depthshade=False,
        edgecolors=PAPER,
        linewidths=0.15,
    )

    axis.set_xlabel(r"$C_i$ (µmol mol$^{-1}$)", labelpad=7)
    axis.set_ylabel(r"$Q$ (µmol photons m$^{-2}$ s$^{-1}$)", labelpad=8)
    axis.set_zlabel(r"$A$ (µmol CO$_2$ m$^{-2}$ s$^{-1}$)", labelpad=7)
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    axis.set_zlim(-1.5, 27)
    axis.view_init(elev=22, azim=-135)
    axis.set_box_aspect((1.1, 1, 0.72))
    axis.set_proj_type("persp", focal_length=0.92)

    pane = mpl.colors.to_rgba(PAPER_ALT, 0.42)
    grid = mpl.colors.to_rgba(INK_MUTED, 0.28)
    for item in (axis.xaxis, axis.yaxis, axis.zaxis):
        item.set_pane_color(pane)
        item._axinfo["grid"]["color"] = grid
        item._axinfo["grid"]["linewidth"] = 0.55
        item._axinfo["axisline"]["color"] = mpl.colors.to_rgba(INK_MUTED, 0.7)
    axis.tick_params(labelsize=7.2, pad=0)

    colorbar = figure.colorbar(surface, ax=axis, pad=0.035, shrink=0.55, aspect=18)
    colorbar.set_label(r"$A$", rotation=0, labelpad=7, color=INK)
    colorbar.ax.tick_params(labelsize=7, colors=INK_SECONDARY, length=2)
    colorbar.outline.set_edgecolor(RULE)
    colorbar.outline.set_linewidth(0.6)

    figure.subplots_adjust(left=0.0, right=0.94, bottom=0.02, top=0.99)
    output = TMP_DIR / "phrag-p10-surface-static.png"
    figure.savefig(output, dpi=330, facecolor=PAPER, transparent=False)
    plt.close(figure)
    return output, data


def render_low_co2_figure() -> Path:
    source = ASSET_DIR / "low-co2-opening.svg"
    output = TMP_DIR / "low-co2-opening-static.png"
    subprocess.run(
        [
            "rsvg-convert",
            "--width",
            "2200",
            "--background-color",
            PAPER,
            "--output",
            str(output),
            str(source),
        ],
        check=True,
    )
    return output


def caption_markup(label: str, lead: str, caption: str, species: str) -> str:
    lead_markup = html.escape(lead).replace(species, f"<i>{species}</i>")
    caption_markup_text = html.escape(caption)
    caption_markup_text = caption_markup_text.replace("g_sw", "<i>g</i><sub>sw</sub>")
    caption_markup_text = caption_markup_text.replace("C_i", "<i>C</i><sub>i</sub>")
    caption_markup_text = caption_markup_text.replace("CO2", "CO<sub>2</sub>")
    return (
        f'<font name="PlexMonoSemibold" color="{BLUE}">{label}</font>  '
        f'<b>{lead_markup}</b> {caption_markup_text}'
    )


def draw_caption(
    canvas: Canvas,
    markup: str,
    x: float,
    top: float,
    width: float,
) -> float:
    style = ParagraphStyle(
        "Caption",
        fontName="PlexSans",
        fontSize=8.8,
        leading=12.8,
        textColor=HexColor(INK_SECONDARY),
        alignment=TA_LEFT,
    )
    paragraph = Paragraph(markup, style)
    _, height = paragraph.wrap(width, 110)
    paragraph.drawOn(canvas, x, top - height)
    return top - height


def draw_reference(
    canvas: Canvas,
    number: int,
    text: str,
    url: str,
    x: float,
    top: float,
    width: float,
) -> float:
    style = ParagraphStyle(
        "Reference",
        fontName="PlexSans",
        fontSize=7.05,
        leading=9.5,
        textColor=HexColor(INK_SECONDARY),
        leftIndent=12,
        firstLineIndent=-12,
        alignment=TA_LEFT,
    )
    markup = (
        f'<font name="PlexMonoSemibold" color="{BLUE}">{number:02d}</font>  '
        f'<link href="{url}" color="{INK_SECONDARY}">{html.escape(text)}</link>'
    )
    paragraph = Paragraph(markup, style)
    _, height = paragraph.wrap(width, 70)
    paragraph.drawOn(canvas, x, top - height)
    return top - height - 4.5


def draw_page_two(canvas: Canvas, low_co2_path: Path, surface_path: Path, surface_data: dict) -> None:
    page_size = landscape(LETTER)
    width, height = page_size
    canvas.setPageSize(page_size)
    draw_page_background(canvas, width, height)

    left = 36
    right = width - 36
    gutter = 22
    column_width = (right - left - gutter) / 2
    second_x = left + column_width + gutter

    canvas.setFont("PlexSerifBold", 22)
    canvas.setFillColor(HexColor(INK))
    canvas.drawString(left, 570, "Figures")
    draw_rule(canvas, left, 550, right - left)

    low_image = Image.open(low_co2_path)
    low_ratio = low_image.height / low_image.width
    low_width = column_width
    low_height = low_width * low_ratio
    low_y = 535 - low_height
    canvas.drawImage(
        ImageReader(low_co2_path),
        left,
        low_y,
        low_width,
        low_height,
        preserveAspectRatio=True,
        mask="auto",
    )

    canvas.drawImage(
        ImageReader(surface_path),
        second_x,
        250,
        column_width,
        285,
        preserveAspectRatio=True,
        anchor="c",
        mask="auto",
    )

    left_caption = caption_markup(
        "FIGURE 1",
        CAPTION_ONE_LEAD,
        CAPTION_ONE,
        "Morella cerifera",
    )
    right_caption = caption_markup(
        "FIGURE 2",
        CAPTION_TWO_LEAD,
        CAPTION_TWO,
        "Phragmites australis",
    )
    left_caption_bottom = draw_caption(canvas, left_caption, left, low_y - 14, column_width)
    right_caption_bottom = draw_caption(canvas, right_caption, second_x, 236, column_width)

    references_top = min(left_caption_bottom, right_caption_bottom) - 20
    canvas.setFont("PlexMonoSemibold", 7.4)
    canvas.setFillColor(HexColor(INK_MUTED))
    canvas.drawString(left, references_top, "SOURCES & METHODS")
    canvas.setStrokeColor(HexColor(RULE))
    canvas.setLineWidth(0.6)
    canvas.line(left + 92, references_top + 2, right, references_top + 2)

    ref_top = references_top - 13
    y_left = ref_top
    for index, (text, url) in enumerate(REFERENCES[:3], start=1):
        y_left = draw_reference(canvas, index, text, url, left, y_left, column_width)
    y_right = ref_top
    for index, (text, url) in enumerate(REFERENCES[3:], start=4):
        y_right = draw_reference(canvas, index, text, url, second_x, y_right, column_width)
    if min(y_left, y_right) < 34:
        raise RuntimeError("Page-two references overflowed")

    canvas.showPage()


def markdown_to_plain_text(markdown: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1", markdown)
    text = text.replace("*", "")
    return " ".join(text.split())


def verify_pdf(essay_paragraphs: list[str]) -> None:
    reader = PdfReader(OUTPUT_PATH)
    if len(reader.pages) != 2:
        raise RuntimeError(f"Expected 2 pages, found {len(reader.pages)}")
    page_one = " ".join((reader.pages[0].extract_text() or "").split())
    for paragraph in essay_paragraphs:
        expected = markdown_to_plain_text(paragraph)
        if expected not in page_one:
            raise RuntimeError(f"Page-one prose changed or disappeared: {expected[:80]}...")


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    register_fonts()
    essay_paragraphs = extract_essay_markdown()
    if len(essay_paragraphs) != 5:
        raise RuntimeError(f"Expected five essay paragraphs, found {len(essay_paragraphs)}")

    low_co2_path = render_low_co2_figure()
    surface_path, surface_data = render_surface()

    canvas = Canvas(str(OUTPUT_PATH), pagesize=LETTER, pageCompression=1)
    canvas.setTitle("Forcing Plants to Participate")
    canvas.setAuthor("Riley Leff")
    canvas.setSubject("Photosynthetic capacity and stomatal limitation")
    draw_page_one(canvas, essay_paragraphs)
    draw_page_two(canvas, low_co2_path, surface_path, surface_data)
    canvas.save()
    verify_pdf(essay_paragraphs)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
