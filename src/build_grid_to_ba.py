"""Build a parquet file that maps EAMxx grid-cell indices to BA codes.

Inputs
------
1. grid_cell_to_county_weight.EAMxx.parquet
       Columns: cell_index (int), FIPS (str, zero-padded), weight (float)
       Each row says grid cell *cell_index* overlaps county *FIPS* with
       area-fraction *weight*.

2. ba_service_territory_2019.csv
       Columns: County_FIPS (int), BA_Code (str), …
       Maps every US county to its Balancing Authority.

Output
------
grid_cell_to_ba.EAMxx.parquet
       Columns: cell_index (int), BA_Code (str), ba_weight (float)
       Aggregated area-weight of each grid cell belonging to each BA.
"""

import logging
from pathlib import Path

import pandas as pd

from util_constants import WRF_TO_TELL_DIR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_BASE = WRF_TO_TELL_DIR

GRID_COUNTY_PQ = _BASE / "grid_cell_to_county_weight.EAMxx.parquet"
BA_CSV = _BASE / "wrf_to_tell_ancillary_data" / "ba_service_territory_2019.csv"
OUTPUT_PQ = (
    Path(__file__).parent / "grid_cell_to_ba.EAMxx.parquet"
)  # Written to the scripts directory so analyze_tell_heatwave.py can find it

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def load_grid_county(path: Path) -> pd.DataFrame:
    """Load the grid-cell → county weight parquet."""
    log.info("Reading %s", path)
    df = pd.read_parquet(path)
    log.info("  shape %s  |  %d unique cells, %d unique FIPS",
             df.shape, df["cell_index"].nunique(), df["FIPS"].nunique())
    return df


def load_ba_territory(path: Path) -> pd.DataFrame:
    """Load the BA service-territory CSV and harmonise FIPS to 5-char str."""
    log.info("Reading %s", path)
    df = pd.read_csv(path)
    # Convert integer County_FIPS → zero-padded 5-character string
    df["FIPS"] = df["County_FIPS"].astype(str).str.zfill(5)
    log.info("  shape %s  |  %d unique FIPS, %d unique BA_Code",
             df.shape, df["FIPS"].nunique(), df["BA_Code"].nunique())
    return df


def build_grid_to_ba(grid_county: pd.DataFrame,
                     ba_territory: pd.DataFrame) -> pd.DataFrame:
    """Merge grid-cell→county with county→BA, then aggregate weights per
    (cell_index, BA_Code).

    Steps
    -----
    1. Left-join on FIPS so every grid-county row gets a BA_Code (or NaN).
    2. Drop rows with no BA match (counties not served by any BA).
    3. Group by (cell_index, BA_Code) and sum the weights — a single grid
       cell may overlap multiple counties in the *same* BA.
    """
    merged = grid_county.merge(
        ba_territory[["FIPS", "BA_Code"]].drop_duplicates(),
        on="FIPS",
        how="left",
    )
    n_before = len(merged)
    merged = merged.dropna(subset=["BA_Code"])
    n_after = len(merged)
    log.info("  Merge: %d rows → %d after dropping unmatched FIPS "
             "(%d dropped)", n_before, n_after, n_before - n_after)

    grid_to_ba = (
        merged
        .groupby(["cell_index", "BA_Code"], as_index=False)["weight"]
        .sum()
        .rename(columns={"weight": "ba_weight"})
    )
    log.info("  Aggregated: %d rows  |  %d unique cells, %d unique BAs",
             len(grid_to_ba),
             grid_to_ba["cell_index"].nunique(),
             grid_to_ba["BA_Code"].nunique())
    return grid_to_ba


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write the result to parquet."""
    df.to_parquet(path, index=False)
    size_mb = path.stat().st_size / 1024**2
    log.info("Saved %s  (%.2f MB, %d rows)", path, size_mb, len(df))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    grid_county = load_grid_county(GRID_COUNTY_PQ)
    ba_territory = load_ba_territory(BA_CSV)
    grid_to_ba = build_grid_to_ba(grid_county, ba_territory)

    # Quick sanity print
    print("\n=== OUTPUT PREVIEW ===")
    print(grid_to_ba.head(15).to_string(index=False))
    print(f"\nShape : {grid_to_ba.shape}")
    print(f"Unique cell_index: {grid_to_ba['cell_index'].nunique()}")
    print(f"Unique BA_Code   : {grid_to_ba['BA_Code'].nunique()}")

    save_parquet(grid_to_ba, OUTPUT_PQ)


if __name__ == "__main__":
    main()
