import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# ─── Page Setup ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="RFDS QLD B200 Landing Distance Calculator", layout="wide")
st.title("🛬 RFDS QLD B200 King Air Landing Distance Calculator — NOT FOR OPERATIONAL USE")

# Landing distance factors selectable in the sidebar
factor_options = {
    "Standard Factor (1.43)": 1.43,
    "Approved Factor (1.20)": 1.20,
}

# ─── Step 0: Dataset selector ──────────────────────────────────────────────
with st.sidebar:
    dataset_choice = st.selectbox("Select dataset", ["set1", "set2", "set3"], index=0)
DATA_ROOT = Path("data") / dataset_choice

def datafile(name: str) -> Path:
    """Resolve a filename inside the selected dataset folder."""
    return DATA_ROOT / name

# ─── Step 1: User Inputs ───────────────────────────────────────────────────
with st.sidebar:
    press_alt = st.slider("Pressure Altitude (ft)", 0, 10000, 0, 250)
    oat = st.slider("Outside Air Temperature (°C)", -5, 45, 15, 1)
    weight = st.slider("Landing Weight (lb)", 9000, 12500, 11500, 100)
    wind = st.slider(
        "Wind Speed (kt)",
        -10,
        30,
        0,
        1,
        help="Negative = tailwind, Positive = headwind",
    )
    factor_label = st.selectbox("Select Landing Distance Factor", list(factor_options.keys()))

    # Runway surface condition (rollout only)
    surface_choice = st.radio(
        "Runway surface (rollout only)",
        options=["Dry (0%)", "Wet (+15%)", "Standing water (+30%)"],
        help="Applies only to ground roll."
    )
    if "Wet" in surface_choice:
        W = 1.15
    elif "Standing" in surface_choice:
        W = 1.30
    else:
        W = 1.00

    slope_deg = st.number_input(
        "Runway Slope (%)",
        min_value=-5.0,
        max_value=0.0,
        value=0.0,
        step=0.1,
        help="Slope factor need only applied when greater than 1%",
    )
    avail_m = st.number_input(
        "Landing Distance Available (m)",
        min_value=0.0,
        value=1150.0,
        step=5.0,
        help="Enter the runway length available in metres",
    )

# ─── Step 2: Table 1 – Pressure Altitude × OAT (Bilinear Interpolation) ───
raw1 = pd.read_csv(datafile("pressureheight_oat.csv"), skiprows=[0])
raw1 = raw1.rename(columns={raw1.columns[0]: "dummy", raw1.columns[1]: "PressAlt"})
tbl1 = raw1.drop(columns=["dummy"]).set_index("PressAlt")
tbl1.columns = tbl1.columns.astype(int)

def lookup_tbl1_bilinear(df, pa, t):
    pas = np.array(sorted(df.index))
    oats = np.array(sorted(df.columns))
    pa  = np.clip(pa, pas[0], pas[-1])
    t   = np.clip(t,  oats[0], oats[-1])
    x1 = pas[pas <= pa].max()
    x2 = pas[pas >= pa].min()
    y1 = oats[oats <= t].max()
    y2 = oats[oats >= t].min()
    Q11 = df.at[x1, y1]; Q21 = df.at[x2, y1]
    Q12 = df.at[x1, y2]; Q22 = df.at[x2, y2]
    if x1 == x2 and y1 == y2:
        return Q11
    if x1 == x2:
        return Q11 + (Q12 - Q11) * (t - y1) / (y2 - y1)
    if y1 == y2:
        return Q11 + (Q21 - Q11) * (pa - x1) / (x2 - x1)
    denom = (x2 - x1) * (y2 - y1)
    fxy1 = Q11 * (x2 - pa) + Q21 * (pa - x1)
    fxy2 = Q12 * (x2 - pa) + Q22 * (pa - x1)
    return (fxy1 * (y2 - t) + fxy2 * (t - y1)) / denom

baseline = lookup_tbl1_bilinear(tbl1, press_alt, oat)
st.markdown("### Step 1: Baseline Distance")
st.success(f"Baseline landing distance: **{baseline:.0f} ft**")

# ─── Step 3: Table 2 – Weight Adjustment (1D Interpolation) ────────────────
raw2    = pd.read_csv(datafile("weightadjustment.csv"), header=0)
wt_cols = [int(w) for w in raw2.columns]
df2     = raw2.astype(float)
df2.columns = wt_cols

def lookup_tbl2_interp(df, baseline, w, ref_weight=12500, _debug=False, _st=None):
    """
    Nearest-columns 2D interpolation on ABSOLUTE values (preferred).
    Returns the absolute weight-adjusted distance.
    """
    tbl = df.copy()
    tbl.columns = [int(c) for c in tbl.columns]

    # Choose a reference column as x-axis (e.g., 12,500 lb column)
    if ref_weight not in tbl.columns:
        raise ValueError(f"ref_weight {ref_weight} not found in columns")
    tbl = tbl.sort_values(by=ref_weight).reset_index(drop=True).astype(float)

    # X-axis tied to reference column (e.g., 12,500 lb)
    x_ref = tbl[ref_weight].values

    # Find nearest lower/upper weight columns
    weights = np.array(sorted(int(c) for c in tbl.columns))
    idx = int(np.searchsorted(weights, w, side="left"))
    if idx == 0:
        w1 = w2 = int(weights[0])
    elif idx >= len(weights):
        w1 = w2 = int(weights[-1])
    else:
        lower = int(weights[idx-1]); upper = int(weights[idx])
        w1, w2 = (upper, upper) if upper == w else (lower, upper)

    # Interpolate ABSOLUTE values in each bounding column at this baseline
    y1 = np.interp(baseline, x_ref, tbl[w1].values,
                   left=tbl[w1].values[0], right=tbl[w1].values[-1])
    y2 = np.interp(baseline, x_ref, tbl[w2].values,
                   left=tbl[w2].values[0], right=tbl[w2].values[-1])

    # Horizontal blend by proximity in weight
    if w1 == w2:
        y = y1
    else:
        alpha = (w - w1) / (w2 - w1)
        y = (1 - alpha) * y1 + alpha * y2

    return float(y)

weight_adj = lookup_tbl2_interp(df2, baseline, weight, _debug=True, _st=st)
st.markdown("### Step 2: Weight Adjustment")
st.success(f"Weight-adjusted distance: **{weight_adj:.0f} ft**")

# ─── Step 4: Table 3 – Wind Adjustment (1D Interpolation) ──────────────────
# (Filename has a space — leaving it as-is to match your repo)
raw3      = pd.read_csv(datafile("wind adjustment.csv"), header=None)
wind_cols = [int(w) for w in raw3.iloc[0]]
df3       = raw3.iloc[1:].reset_index(drop=True).apply(pd.to_numeric, errors="coerce")
df3.columns = wind_cols

def lookup_tbl3_interp(df, refd, ws):
    tbl        = df.sort_values(by=0).reset_index(drop=True)
    ref_rolls  = tbl[0].values
    wind_rolls = tbl[ws].values
    deltas     = wind_rolls - ref_rolls
    delta_wind = np.interp(refd,
                           ref_rolls,
                           deltas,
                           left=deltas[0],
                           right=deltas[-1])
    return float(delta_wind)

delta_wind = lookup_tbl3_interp(df3, weight_adj, wind)
wind_adj   = weight_adj + delta_wind
st.markdown("### Step 3: Wind Adjustment, final ground roll")
st.success(f"After wind adjustment, Ground Roll: **{wind_adj:.0f} ft**")

# ─── Step 5: Table 4 – 50 ft Obstacle Correction (1D Interpolation) ────────
raw4 = pd.read_csv(datafile("50ft.csv"), header=None)
df4  = raw4.iloc[:, :2].copy()
df4.columns = [0, 50]
df4 = df4.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)

def lookup_tbl4_interp(df, refd, h=50, ref_col=0, _debug=False, _st=None):
    """
    2D ABSOLUTE interpolation for the 50 ft obstacle table (or any height h):
      - x-axis: reference distances in column `ref_col` (e.g., 0 ft obstacle).
      - y-axis: absolute distances in the two nearest obstacle-height columns around `h`.
      - returns the absolute distance at obstacle height h.
    """
    tbl = df.copy()
    # Ensure numeric columns
    tbl.columns = pd.to_numeric(tbl.columns, errors="coerce")
    tbl = tbl.dropna(axis=1, how="all")

    # Build an x-axis from the reference column
    if ref_col not in tbl.columns:
        raise ValueError(f"ref_col {ref_col} not found in columns")
    tbl = tbl.sort_values(by=ref_col).reset_index(drop=True).astype(float)
    x_ref = tbl[ref_col].values

    # Map available obstacle heights to their columns
    colmap = {int(c): c for c in tbl.columns if pd.notna(c)}

    # Candidate obstacle columns (numeric only, excluding ref_col)
    obs_heights = sorted([k for k in colmap.keys() if k != ref_col])

    # Find nearest lower/upper heights around h
    import bisect
    idx = bisect.bisect_left(obs_heights, h)
    if idx == 0:
        h1 = h2 = obs_heights[0]
    elif idx >= len(obs_heights):
        h1 = h2 = obs_heights[-1]
    else:
        lower = obs_heights[idx-1]; upper = obs_heights[idx]
        h1, h2 = (upper, upper) if upper == h else (lower, upper)

    # Interpolate ABS values in each obstacle column at this refd
    y1 = np.interp(refd, x_ref, tbl[colmap[h1]].values,
                   left=tbl[colmap[h1]].values[0], right=tbl[colmap[h1]].values[-1])
    y2 = np.interp(refd, x_ref, tbl[colmap[h2]].values,
                   left=tbl[colmap[h2]].values[0], right=tbl[colmap[h2]].values[-1])

    # Horizontal blend by obstacle height
    if h1 == h2:
        y = y1
    else:
        alpha = (h - h1) / (h2 - h1)
        y = (1 - alpha) * y1 + alpha * y2

    return float(y)

obs50 = lookup_tbl4_interp(df4, wind_adj, h=50, ref_col=0, _debug=True, _st=st)
st.markdown("### Step 4: 50 ft Obstacle Correction")
st.success(f"Final landing distance over 50 ft obstacle: **{obs50:.0f} ft**")

# ─── Additional Output: Distance in Meters ─────────────────────────────────
obs50_m = obs50 * 0.3048
st.markdown("### Landing Distance in Meters 50 ft")
st.success(f"{obs50_m:.1f} m")

# ─── Step 6: Apply a Factor ────────────────────────────────────────────────
factor = factor_options[factor_label]
# apply factor to the raw over-50 ft distance
factored_ft = obs50 * factor
factored_m  = factored_ft * 0.3048

st.markdown("### Factored Landing Distance")
col1, col2 = st.columns(2)
col1.success(f"{factored_ft:.0f} ft")
col2.success(f"{factored_m:.1f} m")

# ─── Step X: Ground Roll Corrections (Wet & Slope) ─────────────────────────
# Build rollout-only slope factor:
# - Only apply if downslope magnitude > 1%
if slope_deg < -1.0:
    S = 1.0 + abs(slope_deg) * 0.10   # +10% per 1% downslope
else:
    S = 1.0                           # ignore upslope or small magnitudes

# Combined rollout factor (surface × slope)
rollout_factor = W * S

# Add rollout-only increment to the factored total
delta_rollout_ft = wind_adj * (rollout_factor - 1.0)

required_ft = factored_ft + delta_rollout_ft
required_m  = required_ft * 0.3048

st.markdown("### Ground Roll Corrections (Wet & Slope)")
r1, r2, r3, r4 = st.columns(4)
r1.write(f"**Surface:** {surface_choice}")
r2.write(f"**Surface factor (W):** ×{W:.2f}")
r3.write(f"**Slope:** {slope_deg:+.1f}%")
r4.write(f"**Slope factor (S):** ×{S:.2f}")

c1, c2 = st.columns(2)
c1.success(f"GroundRoll Δ: **{delta_rollout_ft:.0f} ft**")
c2.success(f"Final Landing Distance Required: **{required_ft:.0f} ft** / **{required_m:.1f} m**")

# ─── Step Y: Landing Distance Available & Go/No-Go ─────────────────────────
# Convert to feet
avail_ft = avail_m / 0.3048

# Display the available distance
st.markdown("### Available Runway Length")
c1, c2 = st.columns(2)
c1.write(f"**{avail_m:.0f} m**")
c2.write(f"**{avail_ft:.0f} ft**")

# Determine if tailwind exists (positive wind value)
has_tailwind = wind < 0

# Check if the 1.20 factor is selected
using_1_2_factor = factor_label == "Approved Factor (1.20)"

# Go/No-Go Decision Logic
st.markdown("### Go/No-Go Decision")
if using_1_2_factor and has_tailwind:
    st.error("❌ Landing not permitted: No tailwind component permitted with 1.2 Factoring")
elif avail_ft >= required_ft:
    st.success("✅ Enough runway available for landing")
else:
    st.error("❌ Insufficient runway available for landing")

st.markdown("### Data extracted from B200-601-80 HFG Perfomance Landing Distance Without Propeller Reversing - Flap 100%")
st.markdown("Created by H Watson and R Thomas")
