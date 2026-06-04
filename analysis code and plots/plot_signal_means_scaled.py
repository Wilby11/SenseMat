"""
Outlier detection on per-frame signal means — SCALED version.

Identical to plot_signal_means.py but applies the same per-sensor
StandardScaler that get_dataloaders() uses before computing frame means.

Scaler is fit on the TRAINING split only (same as the model pipeline),
then applied to all files before any stats or outlier detection.

Outputs
-------
- Console summary: ranked file list + per-file outlier counts
- outlier_frames_<preprocessed>_scaled.csv
- window_mean_histogram_<preprocessed>_scaled.png

Run
---
python plot_signal_means_scaled.py [data_root] [non_log|log] [window_size] [overlap] [z_thresh] [train_ratio] [val_ratio] [seed]
Defaults: data_root=., log, 60, 0.5, 3.0, 0.70, 0.15, 42
"""

import sys
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from dataloader import (
    _discover_files,
    _load_csv,
    _split_files,
    _fit_signal_scaler,
    _apply_signal_scaler,
)

# ── Config ────────────────────────────────────────────────────────────────────
data_root    = sys.argv[1] if len(sys.argv) > 1 else "."
preprocessed = sys.argv[2] if len(sys.argv) > 2 else "non_log"
window_size  = int(sys.argv[3])   if len(sys.argv) > 3 else 60
overlap      = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
z_thresh     = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0
train_ratio  = float(sys.argv[6]) if len(sys.argv) > 6 else 0.70
val_ratio    = float(sys.argv[7]) if len(sys.argv) > 7 else 0.15
seed         = int(sys.argv[8])   if len(sys.argv) > 8 else 42

step = max(1, int(window_size * (1 - overlap)))
data_root = Path(data_root)

print(f"Preprocessing : {preprocessed}")
print(f"Window size   : {window_size}  step: {step}  overlap: {overlap}")
print(f"Z threshold   : ±{z_thresh}")
print(f"Train ratio   : {train_ratio}  Val ratio: {val_ratio}  Seed: {seed}")


def frame_to_windows(frame_idx: int, file_len: int) -> list[int]:
    """Return all window indices (0-based) that contain this frame."""
    starts = range(0, file_len - window_size + 1, step)
    return [wi for wi, s in enumerate(starts) if s <= frame_idx < s + window_size]


# ── 1. Discover files and reproduce the train/val/test split ─────────────────
files = _discover_files(str(data_root), preprocessed)
print(f"\nFound {len(files)} files.")

train_files, val_files, test_files = _split_files(files, train_ratio, val_ratio, seed)
print(f"Split  →  train: {len(train_files)}  val: {len(val_files)}  test: {len(test_files)}\n")

# Build a lookup so we can label each file's split in the output
split_label = {}
for f in train_files: split_label[f.name] = "TRAIN"
for f in val_files:   split_label[f.name] = "VAL"
for f in test_files:  split_label[f.name] = "TEST"

# ── 2. Load raw signals ───────────────────────────────────────────────────────
print("Loading raw signals...")
all_raw = {}   # fname -> (sig_raw (T,16,8), T)
for f in files:
    sig, _ = _load_csv(f)
    all_raw[f.name] = (sig, len(sig))

# ── 3. Fit StandardScaler on training split only ──────────────────────────────
train_sigs = [all_raw[f.name][0] for f in train_files]
scaler = _fit_signal_scaler(train_sigs)
print(f"Scaler fit on {len(train_sigs)} training file(s).")
print(f"  per-sensor mean  : min={scaler.mean_.min():.4f}  max={scaler.mean_.max():.4f}")
print(f"  per-sensor scale : min={scaler.scale_.min():.4f}  max={scaler.scale_.max():.4f}\n")

# ── 4. Apply scaler and compute per-frame means ───────────────────────────────
records = []

for f in files:
    sig_raw, T = all_raw[f.name]

    # Apply exactly the same scaling as get_dataloaders()
    sig_scaled = _apply_signal_scaler(sig_raw[np.newaxis], scaler)[0]  # (T, 16, 8)

    frame_means = sig_scaled.mean(axis=(1, 2))   # (T,)

    for t, m in enumerate(frame_means):
        records.append({
            "file":       f.name,
            "split":      split_label[f.name],
            "frame_idx":  t,
            "file_len":   T,
            "frame_mean": float(m),
        })

df = pd.DataFrame(records)

# ── 5. Compute global stats and flag outliers ─────────────────────────────────
global_mean = df["frame_mean"].mean()
global_std  = df["frame_mean"].std()
q1, q3      = df["frame_mean"].quantile(0.25), df["frame_mean"].quantile(0.75)
iqr         = q3 - q1

df["z_score"]     = (df["frame_mean"] - global_mean) / global_std
df["iqr_outlier"] = (df["frame_mean"] < q1 - 1.5 * iqr) | (df["frame_mean"] > q3 + 1.5 * iqr)
df["z_outlier"]   = df["z_score"].abs() > z_thresh
df["is_outlier"]  = df["z_outlier"]

print("Global frame-mean stats (SCALED signal):")
print(f"  mean={global_mean:.4f}  std={global_std:.4f}")
print(f"  Q1={q1:.4f}  Q3={q3:.4f}  IQR={iqr:.4f}")
print(f"  Z-thresh ±{z_thresh} → [{global_mean - z_thresh*global_std:.4f}, "
      f"{global_mean + z_thresh*global_std:.4f}]")
print(f"  IQR fence  → [{q1 - 1.5*iqr:.4f}, {q3 + 1.5*iqr:.4f}]")
print(f"\nTotal frames: {len(df):,}   Outliers (Z): {df['z_outlier'].sum():,}   "
      f"Outliers (IQR): {df['iqr_outlier'].sum():,}\n")

# ── 6. Attach window indices for outlier frames ───────────────────────────────
outliers = df[df["is_outlier"]].copy()
outliers["window_indices"] = outliers.apply(
    lambda r: frame_to_windows(int(r["frame_idx"]), int(r["file_len"])),
    axis=1,
)

# ── 7. Console summary: files ranked by outlier count ────────────────────────
file_summary = (
    df.groupby("file")
    .agg(
        split=("split", "first"),
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

print("─" * 100)
print(f"{'File':<50} {'Split':>5} {'Frames':>7} {'Outliers':>9} {'%':>6}  {'mean':>8}  "
      f"{'min':>8}  {'max':>8}")
print("─" * 100)
for _, row in file_summary.iterrows():
    marker = " ◄" if row["outlier_frames"] > 0 else ""
    print(f"{row['file']:<50} {row['split']:>5} {int(row['total_frames']):>7} "
          f"{int(row['outlier_frames']):>9} {row['outlier_pct']:>5.1f}%  "
          f"{row['file_mean']:>8.4f}  {row['min_mean']:>8.4f}  "
          f"{row['max_mean']:>8.4f}{marker}")
print("─" * 100)

# ── 8. Save outlier CSV ───────────────────────────────────────────────────────
out_csv = data_root / f"outlier_frames_{preprocessed}_scaled.csv"
out_df = outliers[["file", "split", "frame_idx", "frame_mean", "z_score",
                    "iqr_outlier", "window_indices"]].copy()
out_df["window_indices"] = out_df["window_indices"].apply(str)
out_df.to_csv(out_csv, index=False)
print(f"\nSaved outlier CSV → {out_csv.name}  ({len(out_df)} rows)")

# ── 9. Build per-window means ─────────────────────────────────────────────────
window_records = []
for f in files:
    fname = f.name
    sub = df[df["file"] == fname]["frame_mean"].values   # (T,) — already scaled
    T = len(sub)
    starts = range(0, T - window_size + 1, step)
    for wi, s in enumerate(starts):
        window_records.append({
            "file":        fname,
            "split":       split_label[fname],
            "window_idx":  wi,
            "window_mean": sub[s : s + window_size].mean(),
        })

wdf = pd.DataFrame(window_records)
wdf["z_score"]   = (wdf["window_mean"] - global_mean) / global_std
wdf["is_outlier"] = wdf["z_score"].abs() > z_thresh

n_outlier_windows = int(wdf["is_outlier"].sum())
print(f"\nTotal windows : {len(wdf):,}   Outlier windows (Z): {n_outlier_windows:,}")

if n_outlier_windows > 0:
    print(f"\nOutlier windows (global z > ±{z_thresh}):\n")
    outlier_wins = wdf[wdf["is_outlier"]].sort_values(["file", "window_idx"])
    for fname, grp in outlier_wins.groupby("file"):
        short = (fname.replace("_non_log_preprocessed.csv", "")
                      .replace("_log_preprocessed.csv", ""))
        print(f"  {short}  [{split_label[fname]}]  ({len(grp)} window{'s' if len(grp) > 1 else ''})")
        for _, r in grp.iterrows():
            direction = "HIGH" if r["z_score"] > 0 else "LOW"
            print(f"    window {int(r['window_idx']):>4}  mean={r['window_mean']:>8.4f}"
                  f"  z={r['z_score']:>+.2f}  [{direction}]")

# ── 10. Histogram of all window means, coloured by file ───────────────────────
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
        split = split_label[fname]
        # Distinguish splits by line style: train=solid, val=dashed, test=dotted
        linestyle = {"TRAIN": "-", "VAL": "--", "TEST": ":"}.get(split, "-")
        ax.hist(sub, bins=bins, alpha=0.4, color=colour_map[fname],
                label=f"{short} [{split}]", density=False,
                histtype="stepfilled", linestyle=linestyle)

ax.axvline(global_mean + z_thresh * global_std, color="tomato",
           linestyle="--", linewidth=1.2, label=f"+{z_thresh}σ threshold")
ax.axvline(global_mean - z_thresh * global_std, color="tomato",
           linestyle="--", linewidth=1.2, label=f"−{z_thresh}σ threshold")
ax.axvline(global_mean, color="black", linestyle=":", linewidth=1, label="global mean")

ax.set_title(
    f"Window mean distribution — SCALED signal  |  {preprocessed}  |  "
    f"window={window_size}  step={step}  |  {len(wdf):,} windows total  |  "
    f"scaler fit on {len(train_files)} train file(s)",
    fontsize=10,
)
ax.set_xlabel("Mean sensor value (StandardScaler-normalised, avg of 128 sensors over window frames)")
ax.set_ylabel("Window count")

if n_files <= 15:
    ax.legend(fontsize=6, ncol=3, loc="upper right")
else:
    ax.legend(fontsize=5, ncol=4, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0)

fig.tight_layout()
out_png = data_root / f"window_mean_histogram_{preprocessed}_scaled.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved plot     → {out_png.name}")

plt.close("all")
print("Done.")