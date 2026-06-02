"""
Outlier detection on per-frame signal means.

For every frame in every file, compute mean(128 sensors).
Flag frames whose value is beyond a threshold (IQR or Z-score).
Report which files + frame indices + windows contain outliers.

Outputs
-------
- Console summary: ranked file list + per-file outlier counts
- outlier_frames_<preprocessed>.csv : every outlier frame with
    file, frame_idx, window_indices, frame_mean, z_score, iqr_outlier
- frame_mean_timeseries_<preprocessed>.png : per-file time-series
    with outlier frames marked in red

Run
---
python plot_signal_means.py [data_root] [non_log|log] [window_size] [overlap] [z_thresh]
Defaults: data_root=., non_log, window_size=60, overlap=0.5, z_thresh=3.0
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from dataloader import _discover_files, _load_csv

# ── Config ────────────────────────────────────────────────────────────────────
data_root    = sys.argv[1] if len(sys.argv) > 1 else "."
preprocessed = sys.argv[2] if len(sys.argv) > 2 else "non_log" #or non_log
window_size  = int(sys.argv[3])   if len(sys.argv) > 3 else 60
overlap      = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
z_thresh     = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0

step = max(1, int(window_size * (1 - overlap)))
data_root = Path(data_root)

print(f"Preprocessing : {preprocessed}")
print(f"Window size   : {window_size}  step: {step}  overlap: {overlap}")
print(f"Z threshold   : ±{z_thresh}")


def frame_to_windows(frame_idx: int, file_len: int) -> list[int]:
    """Return all window indices (0-based) that contain this frame."""
    starts = range(0, file_len - window_size + 1, step)
    return [
        wi for wi, s in enumerate(starts)
        if s <= frame_idx < s + window_size
    ]


# ── 1. Load all files, collect per-frame means ───────────────────────────────
files = _discover_files(str(data_root), preprocessed)
print(f"\nFound {len(files)} files.\n")

records = []   # list of dicts, one per frame

for f in files:
    sig, _ = _load_csv(f)                  # (T, 16, 8)
    T = len(sig)
    frame_means = sig.mean(axis=(1, 2))    # (T,)
    for t, m in enumerate(frame_means):
        records.append({
            "file":       f.name,
            "frame_idx":  t,
            "file_len":   T,
            "frame_mean": float(m),
        })

df = pd.DataFrame(records)

# ── 2. Compute global stats and flag outliers ─────────────────────────────────
global_mean = df["frame_mean"].mean()
global_std  = df["frame_mean"].std()
q1, q3      = df["frame_mean"].quantile(0.25), df["frame_mean"].quantile(0.75)
iqr         = q3 - q1

df["z_score"]     = (df["frame_mean"] - global_mean) / global_std
df["iqr_outlier"] = (df["frame_mean"] < q1 - 1.5 * iqr) | (df["frame_mean"] > q3 + 1.5 * iqr)
df["z_outlier"]   = df["z_score"].abs() > z_thresh
df["is_outlier"]  = df["iqr_outlier"]   # primary flag (change to iqr_outlier if preferred)

print(f"Global frame-mean stats:")
print(f"  mean={global_mean:.4f}  std={global_std:.4f}")
print(f"  Q1={q1:.4f}  Q3={q3:.4f}  IQR={iqr:.4f}")
print(f"  Z-thresh ±{z_thresh} → [{global_mean - z_thresh*global_std:.4f}, "
      f"{global_mean + z_thresh*global_std:.4f}]")
print(f"  IQR fence  → [{q1 - 1.5*iqr:.4f}, {q3 + 1.5*iqr:.4f}]")
print(f"\nTotal frames: {len(df):,}   Outliers (Z): {df['z_outlier'].sum():,}   "
      f"Outliers (IQR): {df['iqr_outlier'].sum():,}\n")

# ── 3. Attach window indices for outlier frames ───────────────────────────────
outliers = df[df["is_outlier"]].copy()

outliers["window_indices"] = outliers.apply(
    lambda r: frame_to_windows(int(r["frame_idx"]), int(r["file_len"])),
    axis=1,
)

# ── 4. Console summary: files ranked by outlier count ────────────────────────
file_summary = (
    df.groupby("file")
    .agg(
        total_frames=("frame_mean", "count"),
        outlier_frames=("is_outlier", "sum"),
        file_mean=("frame_mean", "mean"),
        file_std=("frame_mean", "std"),
        min_mean=("frame_mean", "min"),
        max_mean=("frame_mean", "max"),
    )
    .reset_index()
)
file_summary["outlier_pct"] = (
    100 * file_summary["outlier_frames"] / file_summary["total_frames"]
)
file_summary = file_summary.sort_values("outlier_frames", ascending=False)

print("─" * 90)
print(f"{'File':<50} {'Frames':>7} {'Outliers':>9} {'%':>6}  {'mean':>8}  "
      f"{'min':>8}  {'max':>8}")
print("─" * 90)
for _, row in file_summary.iterrows():
    marker = " ◄" if row["outlier_frames"] > 0 else ""
    print(f"{row['file']:<50} {int(row['total_frames']):>7} "
          f"{int(row['outlier_frames']):>9} {row['outlier_pct']:>5.1f}%  "
          f"{row['file_mean']:>8.4f}  {row['min_mean']:>8.4f}  "
          f"{row['max_mean']:>8.4f}{marker}")
print("─" * 90)

# ── 6. Save outlier CSV ───────────────────────────────────────────────────────
out_csv = data_root / f"outlier_frames_{preprocessed}.csv"
out_df = outliers[["file", "frame_idx", "frame_mean", "z_score",
                   "iqr_outlier", "window_indices"]].copy()
out_df["window_indices"] = out_df["window_indices"].apply(str)
out_df.to_csv(out_csv, index=False)
print(f"\nSaved outlier CSV → {out_csv.name}  ({len(out_df)} rows)")

# ── 7. Build per-window means ─────────────────────────────────────────────────
# For each file, slide windows over the frame-mean series and compute
# the mean of frame-means within that window.
window_records = []
for f in files:
    fname = f.name
    sub = df[df["file"] == fname]["frame_mean"].values   # (T,)
    T = len(sub)
    starts = range(0, T - window_size + 1, step)
    for wi, s in enumerate(starts):
        w_mean = sub[s : s + window_size].mean()
        window_records.append({
            "file":      fname,
            "window_idx": wi,
            "window_mean": w_mean,
        })

wdf = pd.DataFrame(window_records)

# flag outlier windows using the same global frame-mean thresholds
wdf["is_outlier"] = wdf["window_mean"].abs() > 0   # placeholder, overwrite below
wdf["z_score"]    = (wdf["window_mean"] - global_mean) / global_std
wdf["is_outlier"] = wdf["z_score"].abs() > z_thresh

n_outlier_windows = int(wdf["is_outlier"].sum())
print(f"\nTotal windows : {len(wdf):,}   Outlier windows (Z): {n_outlier_windows:,}")

if n_outlier_windows > 0:
    print(f"\nOutlier windows (global z > ±{z_thresh}):\n")
    outlier_wins = wdf[wdf["is_outlier"]].sort_values(
        ["file", "window_idx"]
    )
    for fname, grp in outlier_wins.groupby("file"):
        short = (fname.replace("_non_log_preprocessed.csv", "")
                      .replace("_log_preprocessed.csv", ""))
        print(f"  {short}  ({len(grp)} window{'s' if len(grp) > 1 else ''})")
        for _, r in grp.iterrows():
            direction = "HIGH" if r["z_score"] > 0 else "LOW"
            print(f"    window {int(r['window_idx']):>4}  mean={r['window_mean']:>8.4f}"
                  f"  z={r['z_score']:>+.2f}  [{direction}]")

# ── 8. Histogram of all window means, coloured by file ────────────────────────
import matplotlib.cm as cm

all_files_sorted = file_summary["file"].tolist()
n_files = len(all_files_sorted)
colours = cm.tab20(np.linspace(0, 1, 20))
colour_map = {fname: colours[i % 20] for i, fname in enumerate(all_files_sorted)}

bins = np.linspace(wdf["window_mean"].min(), wdf["window_mean"].max(), 80)

fig, ax = plt.subplots(figsize=(14, 6))

for fname in all_files_sorted:
    sub = wdf[wdf["file"] == fname]["window_mean"].values
    if len(sub):
        short = (fname.replace("_non_log_preprocessed.csv", "")
                      .replace("_log_preprocessed.csv", ""))
        ax.hist(sub, bins=bins, alpha=0.4, color=colour_map[fname],
                label=short, density=False)

# threshold lines
ax.axvline(global_mean + z_thresh * global_std, color="tomato",
           linestyle="--", linewidth=1.2, label=f"+{z_thresh}σ threshold")
ax.axvline(global_mean - z_thresh * global_std, color="tomato",
           linestyle="--", linewidth=1.2, label=f"−{z_thresh}σ threshold")
ax.axvline(global_mean, color="black", linestyle=":", linewidth=1, label="global mean")

ax.set_title(
    f"Window mean distribution — all files  |  {preprocessed}  |  "
    f"window={window_size}  step={step}  |  {len(wdf):,} windows total",
    fontsize=10,
)
ax.set_xlabel("Mean sensor value (avg of 128 sensors, avg over window frames)")
ax.set_ylabel("Window count")

if n_files <= 15:
    ax.legend(fontsize=6, ncol=3, loc="upper right")
else:
    ax.legend(fontsize=5, ncol=4, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0)

fig.tight_layout()
out_png = data_root / f"window_mean_histogram_{preprocessed}.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved plot     → {out_png.name}")

plt.close("all")
print("Done.")
