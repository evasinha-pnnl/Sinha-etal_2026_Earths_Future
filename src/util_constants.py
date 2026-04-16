"""
util_constants.py
-----------------
Shared configuration constants used across the TELL–heatwave analysis
scripts (analyze_tell_heatwave.py, decompose_load_anomaly.py,
interconnection_stress.py, mean_shift_decompose.py).

Centralising these here avoids copy-paste drift between files.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Ensemble member colours  (consistent across all figures)
# ---------------------------------------------------------------------------
MEMBER_COLORS: dict[str, str] = {
    "EAMxx_ctrl":        "#0000ff",   # blue
    "EAMxx_ctrl_p12":    "#ffa500",   # orange
    "EAMxx_ctrl_p24":    "#ff0000",   # red
    "EAMxx_ctrl_n12":    "#7f7f7f",   # gray
    "EAMxx_ctrl_n24":    "#008000",   # green
    "EAMxx_SSP245":      "#0000ff",   # blue
    "EAMxx_SSP245_p12":  "#ffa500",   # orange
    "EAMxx_SSP245_p24":  "#ff0000",   # red
    "EAMxx_SSP245_n12":  "#7f7f7f",   # gray
    "EAMxx_SSP245_n24":  "#008000",   # green
}

# ---------------------------------------------------------------------------
# Anomaly pairs: perturbation member → its matched control member
# ---------------------------------------------------------------------------
ANOMALY_PAIRS: dict[str, str] = {
    "EAMxx_SSP245":      "EAMxx_ctrl",
    "EAMxx_SSP245_p12":  "EAMxx_ctrl_p12",
    "EAMxx_SSP245_p24":  "EAMxx_ctrl_p24",
    "EAMxx_SSP245_n12":  "EAMxx_ctrl_n12",
    "EAMxx_SSP245_n24":  "EAMxx_ctrl_n24",
}

PERTURBATION_MEMBERS: list[str] = list(ANOMALY_PAIRS.keys())

# ---------------------------------------------------------------------------
# Default font size for figures
# ---------------------------------------------------------------------------
FONTSIZE: int = 14


# ---------------------------------------------------------------------------
# Shared helper: human-readable display name for a member label
# ---------------------------------------------------------------------------
_DISPLAY_NAMES: dict[str, str] = {
    "EAMxx_ctrl":        "Control",
    "EAMxx_ctrl_p12":    "Control +12h",
    "EAMxx_ctrl_p24":    "Control +24h",
    "EAMxx_ctrl_n12":    "Control -12h",
    "EAMxx_ctrl_n24":    "Control -24h",
    "EAMxx_SSP245":      "SSP245",
    "EAMxx_SSP245_p12":  "SSP245 +12h",
    "EAMxx_SSP245_p24":  "SSP245 +24h",
    "EAMxx_SSP245_n12":  "SSP245 -12h",
    "EAMxx_SSP245_n24":  "SSP245 -24h",
}


def display_name(member: str) -> str:
    """Return a short human-readable label for a member string.

    ``"EAMxx_ctrl"`` → ``"Control"``,
    ``"EAMxx_SSP245_p12"`` → ``"SSP245 +12h"``, etc.
    Members not in the lookup table are returned unchanged.
    """
    return _DISPLAY_NAMES.get(member, member)


# ---------------------------------------------------------------------------
# Shared helper: ctrl-then-PGW member ordering for consistent legends
# ---------------------------------------------------------------------------
def sorted_member_order(available: set[str]) -> list[str]:
    """Return members in ctrl₀, PGW₀, ctrl₁, PGW₁, … order.

    With ``ncol=4`` this produces a two-row legend where the top row is all
    control members (dashed lines / hatched bars) and the bottom row is all
    PGW members (solid lines / filled bars).

    Parameters
    ----------
    available : set of member labels present in the current dataset/filter.

    Returns
    -------
    Ordered list; members not covered by ANOMALY_PAIRS are appended at the
    end (e.g. ERA5_obs).
    """
    ordered: list[str] = []
    for pgw, ctrl in ANOMALY_PAIRS.items():
        if ctrl in available:
            ordered.append(ctrl)
        if pgw in available:
            ordered.append(pgw)
    for m in available:
        if m not in ordered:
            ordered.append(m)
    return ordered
