"""
merge_images.py
---------------
Combine individual analysis figures into composite panels.

Two composites are produced:

1. ``DYAMOND1_TELL_BA_Hourly_Loads_2016_with_BA_extent.png``
   The TELL hourly-load time-series figure (left) next to the three per-BA
   service-territory maps stacked vertically (right).

2. ``heatwave_anomaly_panel_{tag}_{year}.png``
   A 2 × 5 grid of heatwave anomaly spatial maps:

   Row 0 (ctrl)  — EAMxx_ctrl   | EAMxx_ctrl_p12   | EAMxx_ctrl_p24   | EAMxx_ctrl_n12   | EAMxx_ctrl_n24
   Row 1 (PGW)   — EAMxx_SSP245 | EAMxx_SSP245_p12 | EAMxx_SSP245_p24 | EAMxx_SSP245_n12 | EAMxx_SSP245_n24

Usage
-----
    conda activate py313
    python merge_images.py

Authors
-------
Eva Sinha  <eva.sinha@pnnl.gov>
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------
def merge_ba_and_timeseries(
    img_left: Image.Image,
    img_top: Image.Image,
    img_mid: Image.Image,
    img_bot: Image.Image,
    ba_panel_height: int = 370,
) -> Image.Image:
    """Stack three BA maps vertically and paste them to the right of a time-series figure.

    The three BA maps are uniformly scaled to *ba_panel_height* pixels tall
    (aspect ratio preserved) before compositing.

    Parameters
    ----------
    img_left : PIL Image
        Wide time-series figure placed on the left.
    img_top, img_mid, img_bot : PIL Image
        BA service-territory maps stacked top-to-bottom on the right.
    ba_panel_height : int
        Target height in pixels for each BA panel (default 370).

    Returns
    -------
    PIL Image
        Composite RGBA image.
    """
    # Scale all three BA panels to the same height (preserve aspect ratio)
    aspect = img_top.size[0] / img_top.size[1]
    ba_w   = int(round(ba_panel_height * aspect))
    ba_size = (ba_w, ba_panel_height)

    img_top = img_top.resize(ba_size)
    img_mid = img_mid.resize(ba_size)
    img_bot = img_bot.resize(ba_size)

    # Canvas: left panel width + BA panel width, left panel height
    canvas_w = img_left.size[0] + ba_w
    canvas_h = img_left.size[1]
    composite = Image.new("RGBA", (canvas_w, canvas_h))

    composite.paste(img_left, (0, 0))
    composite.paste(img_top,  (img_left.size[0], 0))
    composite.paste(img_mid,  (img_left.size[0], ba_panel_height))
    composite.paste(img_bot,  (img_left.size[0], ba_panel_height * 2))

    return composite

# ---------------------------------------------------------------------------
# Heatwave anomaly 2 × 5 panel
# ---------------------------------------------------------------------------

# Row definitions: (row label, list of data_source strings left-to-right)
_PANEL_ROWS: list[tuple[str, list[str]]] = [
    ("Control", ["EAMxx_ctrl",   "EAMxx_ctrl_p12",   "EAMxx_ctrl_p24",   "EAMxx_ctrl_n12",   "EAMxx_ctrl_n24"]),
    ("PGW",     ["EAMxx_SSP245", "EAMxx_SSP245_p12", "EAMxx_SSP245_p24", "EAMxx_SSP245_n12", "EAMxx_SSP245_n24"]),
]

# Column header labels (one per member in the order above)
_COL_LABELS: list[str] = [
    "Initial",
    "Initial +12 hr",
    "Initial +24 hr",
    "Initial -12 hr",
    "Initial -24 hr",
]


def _fig_path(
    ensemble_output_dir: Path,
    data_source: str,
    tag: str,
    year: int,
) -> Path:
    """Return the expected path for one per-member spatial figure."""
    fname = f"{data_source}_t2m_{tag}_{year}.png"
    return ensemble_output_dir / data_source / fname


def merge_heatwave_panel(
    ensemble_output_dir: Path,
    tag: str,
    year: int,
    cell_width: int = 480,
    cell_height: int = 275,
    label_height: int = 50,
    row_label_width: int = 80,
) -> Image.Image:
    """Build a 2 × 5 composite of heatwave spatial maps.

    Layout (W = cell_width, H = cell_height, L = label_height,
            R = row_label_width):

    ┌───┬────────┬────────┬────────┬────────┬────────┐   ← column headers
    │   │Intial  │ +12 hr │ +24 hr │ -12 hr │ −24 hr │
    ├───┼────────┼────────┼────────┼────────┼────────┤
    │ c │ image  │ image  │ image  │ image  │ image  │  ctrl row
    │ t │        │        │        │        │        │
    ├───┼────────┼────────┼────────┼────────┼────────┤
    │ P │ image  │ image  │ image  │ image  │ image  │  PGW row
    │ G │        │        │        │        │        │
    └───┴────────┴────────┴────────┴────────┴────────┘

    Missing figures are replaced with a grey placeholder with a warning.

    Parameters
    ----------
    ensemble_output_dir : Path
        Root directory containing one sub-folder per ensemble member.
    tag : str
        Figure type tag, e.g. ``'max_heatwave_anomaly'``.
    year : int
        Analysis year, e.g. ``2011``.
    cell_width, cell_height : int
        Width/height in pixels to which each panel image is rescaled.
    label_height : int
        Height in pixels of the column-header strip at the top.
    row_label_width : int
        Width in pixels of the row-label strip on the left.

    Returns
    -------
    PIL Image (RGBA)
    """
    n_cols = len(_COL_LABELS)
    n_rows = len(_PANEL_ROWS)

    canvas_w = row_label_width + n_cols * cell_width
    canvas_h = label_height    + n_rows * cell_height
    composite = Image.new("RGBA", (canvas_w, canvas_h), color=(255, 255, 255, 255))
    draw = ImageDraw.Draw(composite)
    font_header = ImageFont.load_default(size=24)   # column headers
    font_row    = ImageFont.load_default(size=22)   # row labels

    # ── Column headers ───────────────────────────────────────────────────────
    for col_idx, label in enumerate(_COL_LABELS):
        x = row_label_width + col_idx * cell_width + cell_width // 2
        draw.text((x, label_height // 2), label,
                  fill=(0, 0, 0, 255), font=font_header, anchor="mm")

    # ── Row labels (rotated 90°) + cells ─────────────────────────────────────
    for row_idx, (row_label, members) in enumerate(_PANEL_ROWS):
        y_top = label_height + row_idx * cell_height

        # Render label on a temporary image, rotate, then paste
        # Temp image is tall enough to contain the text at the chosen font size
        tmp = Image.new("RGBA", (cell_height, row_label_width), color=(255, 255, 255, 255))
        tmp_draw = ImageDraw.Draw(tmp)
        tmp_draw.text(
            (cell_height // 2, row_label_width // 2),
            row_label,
            fill=(0, 0, 0, 255), font=font_row, anchor="mm",
        )
        rotated = tmp.rotate(90, expand=True)   # expand=True resizes canvas to fit rotated text
        composite.paste(rotated, (0, y_top))

        for col_idx, member in enumerate(members):
            fpath = _fig_path(ensemble_output_dir, member, tag, year)
            x_left = row_label_width + col_idx * cell_width

            cell_img = Image.open(fpath).resize((cell_width, cell_height))

            composite.paste(cell_img, (x_left, y_top))

    return composite

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    SRC = Path(__file__).parent

    # ── Composite 1: TELL load time-series + BA service territory maps ────────
    IMG_TOP  = SRC / "outputs/BA_service_territory/BA_Service_Territory_NYIS.png"
    IMG_MID  = SRC / "outputs/BA_service_territory/BA_Service_Territory_PJM.png"
    IMG_BOT  = SRC / "outputs/BA_service_territory/BA_Service_Territory_MISO.png"
    IMG_LEFT = SRC / "outputs/DYAMOND1/DYAMOND1_TELL_BA_Hourly_Loads_2016.png"
    IMG_OUT  = SRC / "outputs/DYAMOND1/DYAMOND1_TELL_BA_Hourly_Loads_2016_with_BA_extent.png"

    composite_load = merge_ba_and_timeseries(
        img_left=Image.open(IMG_LEFT),
        img_top =Image.open(IMG_TOP),
        img_mid =Image.open(IMG_MID),
        img_bot =Image.open(IMG_BOT),
    )
    IMG_OUT.parent.mkdir(parents=True, exist_ok=True)
    composite_load.save(IMG_OUT)
    log.info("Saved → %s", IMG_OUT)

    # ── Composite 2: heatwave 2 × 4 panels (one per tag) ─────────────────────
    ENSEMBLE_OUTPUT_DIR = SRC / "outputs"
    YEAR = 2011
    TAGS = [
        "heatwave_frequency",
        "largest_heatwave_region",
        "longest_heatwave_duration",
        "max_heatwave_anomaly",
    ]

    for TAG in TAGS:
        panel = merge_heatwave_panel(
            ensemble_output_dir=ENSEMBLE_OUTPUT_DIR,
            tag=TAG,
            year=YEAR,
        )
        panel_out = SRC / f"outputs/heatwave_panel_{TAG}_{YEAR}.png"
        panel_out.parent.mkdir(parents=True, exist_ok=True)
        panel.save(panel_out)
        log.info("Saved → %s", panel_out)