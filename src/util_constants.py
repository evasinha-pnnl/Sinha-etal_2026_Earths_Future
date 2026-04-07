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
    "EAMxx_ctrl":        "#1f77b4",   # blue
    "EAMxx_ctrl_p12":    "#ff7f0e",   # orange
    "EAMxx_ctrl_p24":    "#9467bd",   # purple
    "EAMxx_ctrl_n24":    "#8c564b",   # brown
    "EAMxx_SSP245":      "#2ca02c",   # green
    "EAMxx_SSP245_p12":  "#d62728",   # red
    "EAMxx_SSP245_p24":  "#9467bd",   # purple
    "EAMxx_SSP245_n24":  "#8c564b",   # brown
}

# ---------------------------------------------------------------------------
# Anomaly pairs: perturbation member → its matched control member
# ---------------------------------------------------------------------------
ANOMALY_PAIRS: dict[str, str] = {
    "EAMxx_SSP245":      "EAMxx_ctrl",
    "EAMxx_SSP245_p12":  "EAMxx_ctrl_p12",
    "EAMxx_SSP245_p24":  "EAMxx_ctrl_p24",
    "EAMxx_SSP245_n24":  "EAMxx_ctrl_n24",
}

PERTURBATION_MEMBERS: list[str] = list(ANOMALY_PAIRS.keys())

# ---------------------------------------------------------------------------
# Default font size for figures
# ---------------------------------------------------------------------------
FONTSIZE: int = 14
