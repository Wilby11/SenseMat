"""
SenseMat DataLoader
===================
Loads preprocessed sensor CSV files, reconstructs the 2D SenseMat array,
windows the data, optionally loads subject metadata, and returns
PyTorch DataLoaders ready for CNN / RNN / LSTM / Transformer training.

Expected repository layout (with example files)
---------------------------
data_root/
├── Non-log preprocessed data/
│   └── subject9_run3_non_log_preprocessed.csv
├── Log preprocessed data/
│   └── subject17_run1_log_preprocessed.csv
└── recordings/
    ├── pn01/
    │   └── metadata_pn01.json
    ├── pn09/
    │   └── metadata_pn09.json
    └── ...

Metadata JSON example (not real data, just format) 
- example path: for subject9 this file is in recordings/pn09/metadata_pn09.json
---------------------
{
    "sex": "female",
    "age": 27,
    "height": 1.70,
    "weight": 64,
    "head_circumference": 55.9
}

Input array shape after reconstruction
---------------------------------------
The SenseMat is a physical mat oriented vertically: 16 rows tall x 8 columns wide.
CSV column naming is S_{mat_col}_{mat_row}, i.e. the FIRST index is the column (0-7)
and the SECOND index is the row (0-15).

Reconstructed layout — array[t, row, col] = S_{col}_{row}:

    S_0_0  S_1_0  S_2_0  …  S_7_0    <- row 0  (top)
    S_0_1  S_1_1  S_2_1  …  S_7_1
    ...
    S_0_15 S_1_15 S_2_15 …  S_7_15   <- row 15 (bottom)

Corners:
    S_0_0  -> top-left      S_7_0  -> top-right
    S_0_15 -> bottom-left   S_7_15 -> bottom-right

Per-timestep raw shape : (16, 8)
Windowed raw shape     : (window, 16, 8)

Model-specific shapes (set `flat_spatial=True/False`)
- N = batch_size
- window = window_size (frames per window)
------------------------------------------------------
Model               flat_spatial    returned X shape          notes
──────────────────  ─────────────   ────────────────────────  ─────────────────────────────────
LSTM / RNN          True            (N, window, 128)          128 = 8×16 flattened per timestep
                                                              LSTM processes along dim 1
CNN-1D              True            (N, 128, window)          channels-first; conv over time
CNN-2D              False           (N, 1, window, 16, 8)     or reshape to (N, window, 16, 8)
                                                              treat spatial dims as H×W per frame
CNN + LSTM          False           (N, window, 16, 8)        CNN extracts spatial, LSTM over time
Transformer         True            (N, window, 128)          same as LSTM; add pos encoding in model

Usage
-----
from sensemat_dataloader import get_dataloaders

train_dl, val_dl, test_dl = get_dataloaders(
    data_root       = "/path/to/data",
    preprocessed    = "non_log",      # "non_log" or "log"
    window_size     = 60,             # frames per window
    overlap         = 0.5,            # fraction overlap between windows (0.0–<1.0)
    flat_spatial    = True,           # True → (N, window, 128), False → (N, window, 16, 8)
    use_metadata    = True,           # include subject metadata in batch
    normalize_signal = True,         # True → per-sensor z-score normalisation; False → raw values
    batch_size      = 32,
    train_ratio     = 0.70,
    val_ratio       = 0.15,           # test_ratio is inferred as 1 - train_ratio - val_ratio
    seed            = 42,
    run_labels_csv  = "/path/to/run_labels_notes.csv",  # auto-detected: same folder as this file
    quality         = "standard",     # "all" | "standard" | "conservative" | "clean" | "reliable"
    degug           = False,          # print detailed diagnostics at each stage (just for checking the file, don't set to True while training)
)

# One batch:
#   use_metadata=True  → (X, meta, y)
#   use_metadata=False → (X, y)
#
# X    : torch.float32  (batch, window, 128)  or  (batch, window, 16, 8)
# meta : torch.float32  (batch, 5)            — only when use_metadata=True
# y    : torch.float32  (batch, window, 6)    — X/Y/Z/Pitch/Yaw/Roll per timestep

# where batch= batch@size, window=window_size (frames per window)
"""

import json
import os
import re
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

N_ROWS      = 16  # mat rows    (second S index: 0-15)
N_COLS      = 8   # mat columns (first  S index: 0-7)
LABEL_COLS  = ["X", "Y", "Z", "Pitch", "Yaw", "Roll"]
SIGNAL_COLS = [f"S_{s}_{f}" for s in range(N_COLS) for f in range(N_ROWS)]

# Metadata field order (must stay fixed for consistent vector encoding)
META_FIELDS = ["sex", "age", "height", "weight", "head_circumference"]

# ── Notes-based quality filtering ────────────────────────────────────────────
# Notes values that are always excluded regardless of mode
_NOTES_ALWAYS_EXCLUDE = {"no_data", "not_sync"}

# Mapping from quality mode → additional notes to exclude (on top of always-excluded)
_NOTES_EXCLUDE_EXTRA: dict[str, set[str]] = {
    "all":         set(),
    "standard":    {"different_pillow", "shoulders"},
    "conservative":{"different_pillow", "shoulders", "pixel_error", "out_of_frame"},
    "clean":       {"different_pillow", "shoulders", "pixel_error", "out_of_frame",
                    "sensor_slipped", "bounce"},
    "reliable":    None,   # special case: keep ONLY "reliable"
}

QualityMode = Literal["all", "standard", "conservative", "clean", "reliable"]


# ─────────────────────────────────────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────────────────────────────────────

def _discover_files(data_root: str, preprocessed: Literal["non_log", "log"]) -> List[Path]:
    """Return sorted list of CSV paths from the chosen preprocessing folder."""
    folder_map = {
        "non_log": "Non-log preprocessed data",
        "log":     "Log preprocessed data",
    }
    folder = Path(data_root) / folder_map[preprocessed]
    if not folder.exists():
        raise FileNotFoundError(f"Data folder not found: {folder}")
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder}")
    return files


def _parse_filename(path: Path) -> Tuple[int, int]:
    """Extract (subject_id, run_id) from a filename like subject9_run3_..."""
    m = re.search(r"subject(\d+)_run(\d+)", path.stem, re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse subject/run from filename: {path.name}")
    return int(m.group(1)), int(m.group(2))


# ─────────────────────────────────────────────────────────────────────────────
# Run-label / notes CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_run_labels(csv_path: str) -> pd.DataFrame:
    """
    Load the run-labels CSV and return a clean DataFrame with columns:
        subject (int), run (int), label (str), notes (str)
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    for col in ("label", "notes"):
        df[col] = df[col].str.strip()
    return df


def _lookup_label(run_df: pd.DataFrame, subject_id: int, run_id: int) -> Optional[str]:
    """Return the movement label for a given subject+run, or None if not found."""
    row = run_df[(run_df["subject"] == subject_id) & (run_df["run"] == run_id)]
    if row.empty:
        return None
    return str(row.iloc[0]["label"])


def _lookup_notes(run_df: pd.DataFrame, subject_id: int, run_id: int) -> Optional[str]:
    """Return the notes value for a given subject+run, or None if not found."""
    row = run_df[(run_df["subject"] == subject_id) & (run_df["run"] == run_id)]
    if row.empty:
        return None
    return str(row.iloc[0]["notes"])


def _filter_files_by_notes(
    files:       List[Path],
    run_df:      pd.DataFrame,
    quality:     QualityMode,
    debug:       bool = False,
) -> List[Path]:
    """
    Filter *files* according to the chosen quality mode.

    Modes
    -----
    all          – drop no_data, not_sync
    standard     – drop no_data, not_sync, different_pillow, shoulders
    conservative – drop no_data, not_sync, different_pillow, shoulders,
                   pixel_error, out_of_frame
    clean        – drop no_data, not_sync, different_pillow, shoulders,
                   pixel_error, out_of_frame, sensor_slipped, bounce
    reliable     – keep only files whose notes == "reliable"
    """
    if quality not in _NOTES_EXCLUDE_EXTRA:
        raise ValueError(
            f"Unknown quality mode '{quality}'. "
            f"Choose from: {list(_NOTES_EXCLUDE_EXTRA.keys())}"
        )

    kept, dropped = [], []
    for f in files:
        subject_id, run_id = _parse_filename(f)
        notes = _lookup_notes(run_df, subject_id, run_id)

        if notes is None:
            # No entry in CSV → warn and keep (conservative fall-through)
            if debug:
                print(f"  [FILTER] {f.name}: not found in CSV — keeping")
            kept.append(f)
            continue

        if quality == "reliable":
            keep = (notes == "reliable")
        else:
            exclude_set = _NOTES_ALWAYS_EXCLUDE | _NOTES_EXCLUDE_EXTRA[quality]
            keep = notes not in exclude_set

        if keep:
            kept.append(f)
        else:
            dropped.append((f, notes))

    if debug:
        print(f"\n[DEBUG] Quality filter mode='{quality}': "
              f"kept {len(kept)}, dropped {len(dropped)}")
        for f, n in dropped:
            print(f"  [DROPPED] {f.name}  (notes='{n}')")

    return kept


# ─────────────────────────────────────────────────────────────────────────────
# Metadata loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_metadata(data_root: str, subject_id: int) -> dict:
    """Load the JSON metadata for a given subject."""
    pn = f"pn{subject_id:02d}"
    meta_path = Path(data_root) / "recordings" / pn / f"metadata_{pn}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    with open(meta_path) as f:
        return json.load(f)


def _encode_metadata(meta: dict) -> np.ndarray:
    """Encode metadata dict → float32 vector of length 5."""
    sex_enc = 1.0 if str(meta.get("sex", "")).lower() == "female" else 0.0
    return np.array([
        sex_enc,
        float(meta["age"]),
        float(meta["height"]),
        float(meta["weight"]),
        float(meta["head_circumference"]),
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Train / val / test split  (file-level, before any windowing)
# ─────────────────────────────────────────────────────────────────────────────

def _split_files(
    files:       List[Path],
    run_df:      pd.DataFrame,
    train_ratio: float,
    val_ratio:   float,
    seed:        int,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Stratified file-level split on movement label.

    Strategy
    --------
    1. Look up each file's movement label (e.g. "slow/slow", "fast/fast",
       "cough") from the run-labels CSV using its subject + run numbers.
    2. Group files by that label so every movement type is represented
       proportionally in train / val / test.
    3. Within each label-group, shuffle deterministically with *seed* and
       split by the requested ratios.
    4. Files with no matching CSV entry fall back to an "unknown" group so
       they are still included without silently biasing the split.

    This replaces the previous run-number grouping and ensures that
    movement-type diversity is balanced across all three splits.
    """
    assert train_ratio + val_ratio < 1.0, "train + val ratios must be < 1.0"
    rng = np.random.default_rng(seed)

    # Group files by movement label
    label_groups: dict[str, List[Path]] = {}
    for f in files:
        subject_id, run_id = _parse_filename(f)
        label = _lookup_label(run_df, subject_id, run_id) or "unknown"
        label_groups.setdefault(label, []).append(f)

    train_files, val_files, test_files = [], [], []

    for label in sorted(label_groups):
        group = label_groups[label]
        idx      = rng.permutation(len(group))
        shuffled = [group[i] for i in idx]

        n_train = max(1, round(len(shuffled) * train_ratio))
        n_val   = max(1, round(len(shuffled) * val_ratio))
        n_val   = min(n_val, len(shuffled) - n_train)

        train_files.extend(shuffled[:n_train])
        val_files.extend(shuffled[n_train : n_train + n_val])
        test_files.extend(shuffled[n_train + n_val :])

    return train_files, val_files, test_files


# ─────────────────────────────────────────────────────────────────────────────
# CSV loading and windowing
# ─────────────────────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load one CSV file.

    Returns
    -------
    signal : float32 ndarray (T, 16, 8)   — SenseMat spatial layout
    labels : float32 ndarray (T, 6)        — X/Y/Z/Pitch/Yaw/Roll
    """
    df = pd.read_csv(path)

    # ── Check for missing signal columns ─────────────────────────────────────
    missing_cols = [c for c in SIGNAL_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{path.name}: missing signal columns: {missing_cols[:5]}{'…' if len(missing_cols) > 5 else ''}")

    # ── Signal → (T, 16, 8) ──────────────────────────────────────────────────
    # reshape(-1, N_COLS, N_ROWS) groups by mat-col first, then .transpose(0,2,1) gives
    # array[t, row, col] = S_{col}_{row}: portrait layout, 16 rows x 8 cols.
    sig_df = df[SIGNAL_COLS]
    nan_rows = sig_df.isnull().any(axis=1)
    if nan_rows.any():
        bad_indices = nan_rows[nan_rows].index.tolist()
        print(f"  [WARNING] {path.name}: NaN signal at row(s) {bad_indices} — dropping {len(bad_indices)} row(s)")
        df = df.drop(index=bad_indices).reset_index(drop=True)
        sig_df = df[SIGNAL_COLS]

    signal = sig_df.values.astype(np.float32)                         # (T, 128)
    signal = signal.reshape(-1, N_COLS, N_ROWS).transpose(0, 2, 1)   # (T, 16, 8)

    # ── Labels ───────────────────────────────────────────────────────────────
    labels = df[LABEL_COLS].values.astype(np.float32)       # (T, 6)

    return signal, labels


def _make_windows(
    signal:      np.ndarray,   # (T, 16, 8)
    labels:      np.ndarray,   # (T, 6)
    window_size: int,
    overlap:     float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slice (signal, labels) into overlapping windows.

    Parameters
    ----------
    overlap : fraction of window that overlaps with the next (default 0.5 → 50 %)

    Returns
    -------
    X_windows : (N, window_size, 16, 8)
    y_windows : (N, window_size, 6)   — full label sequence per window
    """
    step = max(1, int(window_size * (1 - overlap)))
    T    = len(signal)

    starts = range(0, T - window_size + 1, step)
    X_windows = np.stack([signal[i : i + window_size] for i in starts])  # (N, W, 16, 8)
    y_windows = np.stack([labels[i : i + window_size] for i in starts])  # (N, W, 6)

    return X_windows, y_windows


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation  (fit on train only)
# ─────────────────────────────────────────────────────────────────────────────

def _fit_signal_scaler(X_list: List[np.ndarray]) -> StandardScaler:
    """Fit a StandardScaler on flattened signal from training files."""
    flat = np.concatenate([x.reshape(-1, N_ROWS * N_COLS) for x in X_list], axis=0)
    scaler = StandardScaler()
    scaler.fit(flat)
    return scaler


def _apply_signal_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Apply scaler to (N, W, 16, 8) array."""
    shape  = X.shape                                    # (N, W, 16, 8)
    flat   = X.reshape(-1, N_ROWS * N_COLS)     # (N*W, 128)
    scaled = scaler.transform(flat).astype(np.float32)
    return scaled.reshape(shape)


def _fit_meta_scaler(meta_list: List[np.ndarray]) -> StandardScaler:
    stacked = np.stack(meta_list, axis=0)               # (M, 5)
    scaler  = StandardScaler()
    scaler.fit(stacked)
    return scaler


# ─────────────────────────────────────────────────────────────────────────────
# IQR window filtering  (computed on unscaled signal, fit on train only)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_window_means(
    raw_pairs:   List[Tuple[np.ndarray, np.ndarray]],
    window_size: int,
    overlap:     float,
) -> np.ndarray:
    """
    For a list of (signal(T,16,8), labels) pairs, slide windows and compute
    the mean of all 128 sensors over every frame in the window.

    Returns a 1-D array of per-window means (unscaled signal).
    """
    step = max(1, int(window_size * (1 - overlap)))
    all_means = []
    for sig, _ in raw_pairs:
        T = len(sig)
        for s in range(0, T - window_size + 1, step):
            w = sig[s : s + window_size]        # (W, 16, 8)
            all_means.append(float(w.mean()))
    return np.array(all_means, dtype=np.float32)


def _iqr_fence(window_means: np.ndarray, multiplier: float = 1.5) -> Tuple[float, float]:
    """Return (lower_fence, upper_fence) using multiplier×IQR rule."""
    q1, q3 = float(np.percentile(window_means, 25)), float(np.percentile(window_means, 75))
    iqr    = q3 - q1
    return q1 - multiplier * iqr, q3 + multiplier * iqr


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class LinkedSenseMatAugmentation:
    """
    Applies spatial augmentations to the SenseMat pressure grid 
    AND dynamically recalculates the 6DoF labels to maintain physical truth.
    """
    def __init__(self, p_flip=0.5, scale_range=(0.8, 1.2), noise_std=0.05):
        self.p_flip = p_flip
        self.scale_range = scale_range
        self.noise_std = noise_std

    def __call__(self, x, y):
        # x shape: (W, 16, 8)
        # y shape: (W, 6) -> [X, Y, Z, Pitch, Yaw, Roll]

        # --- 1. The Linked Spatial Augmentation (Horizontal Flip) ---
        if torch.rand(1).item() < self.p_flip:
            # Flip the physical mat horizontally (dimension 2 is the 8 columns)
            x = torch.flip(x, dims=[2])
            
            # Dynamically invert the corresponding physical labels
            # Index 0: X Position (Left/Right)
            # Index 4: Yaw (Look Left/Right)
            # Index 5: Roll (Tilt Left/Right)
            y[:, 0] = -y[:, 0]
            y[:, 4] = -y[:, 4]
            y[:, 5] = -y[:, 5]

        # --- 2. The Safe Physical Augmentations (Always applied randomly) ---
        # Scale pressure
        scale_factor = torch.empty(1).uniform_(*self.scale_range).item()
        x = x * scale_factor

        # Inject noise
        noise = torch.randn_like(x) * self.noise_std
        x = x + noise

        return x, y

class SenseMatDataset(Dataset):
    """
    Parameters
    ----------
    X_windows    : (N, window, 16, 8)
    y_windows    : (N, window, 6)
    meta_vectors : (N, 5)  or None
    flat_spatial : if True  → X returned as (window, 128)   [LSTM/RNN/Transformer]
                   if False → X returned as (window, 16, 8) [CNN-2D / CNN+LSTM]
    """

    def __init__(
        self,
        X_windows:    np.ndarray,
        y_windows:    np.ndarray,
        meta_vectors: Optional[np.ndarray],
        flat_spatial: bool,
        transform:    Optional[callable] = None,
    ):
        self.flat_spatial = flat_spatial
        self.transform = transform
        self.X    = torch.from_numpy(X_windows)    # (N, W, 16, 8)
        self.y    = torch.from_numpy(y_windows)    # (N, W, 6)
        self.meta = (
            torch.from_numpy(meta_vectors)         # (N, 5)
            if meta_vectors is not None else None
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        # Use .clone() to prevent contiguous memory warnings during augmentation
        x = self.X[idx].clone()                        # (W, 16, 8)
        y = self.y[idx].clone()                        # (W, 16, 8)
        
        # Apply the augmentation before any reshaping
        if self.transform is not None:
            x, y = self.transform(x, y)

        if self.flat_spatial:
            # LSTM / RNN / Transformer: (W, 128)
            x = x.reshape(x.shape[0], -1)

        # CNN-1D note: caller should permute to (128, W) inside the model or
        # add a .permute(0,2,1) in the model's forward() — not done here so
        # that the time axis stays consistent across all model types.

        #y = self.y[idx]                            # (W, 6)

        if self.meta is not None:
            return x, self.meta[idx], y
        return x, y


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    data_root:      str,
    preprocessed:   Literal["non_log", "log"] = "non_log",
    window_size:    int   = 60,
    overlap:        float = 0.5,
    flat_spatial:   bool  = True,
    use_metadata:   bool  = True,
    batch_size:     int   = 32,
    train_ratio:    float = 0.70,
    val_ratio:      float = 0.15,
    seed:           int   = 42,
    normalize_signal: bool  = True,
    num_workers:    int   = 0,
    quality:        QualityMode   = "standard",
    iqr_multiplier: Optional[float] = 3.0,
    debug:          bool  = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders.

    Parameters
    ----------
    data_root     : root directory containing the data folders and recordings/
    preprocessed  : "non_log" or "log" — selects the source folder
    window_size   : number of frames per window
    overlap       : fraction overlap between consecutive windows (0.0–<1.0)
    flat_spatial  : True  → X shape (batch, window, 128)      LSTM/RNN/Transformer
                    False → X shape (batch, window, 16, 8)    CNN-2D / CNN+LSTM
    use_metadata  : whether to load and return subject metadata
    batch_size    : DataLoader batch size
    train_ratio   : fraction of files for training
    val_ratio     : fraction of files for validation
                    (test_ratio = 1 - train_ratio - val_ratio)
    normalize_signal : if True, fit a per-sensor StandardScaler on train and apply to all splits
                       if False, raw signal values are passed through unchanged
    seed          : random seed for reproducible splits
    num_workers   : DataLoader worker processes
    iqr_multiplier : fence multiplier for window-mean outlier removal.
                    Windows whose unscaled mean (avg of 128 sensors over all
                    frames in the window) falls outside
                    Q1 − k×IQR … Q3 + k×IQR are dropped, where k is this
                    value. The fence is fit on training windows only.
                    A per-split summary of removed windows (file name, window
                    index, mean value, and HIGH/LOW direction) is always printed.
                      1.5  — standard Tukey fence, very harsh
                      2.0  — looser, only removes more extreme outliers
                      3.0  — very loose, we use
                      None — disabled, no windows are removed
    quality       : data quality filter applied before splitting. Options:
                    "all"          – drop no_data, not_sync
                    "standard"     – drop no_data, not_sync, different_pillow, shoulders
                    "conservative" – also drop pixel_error, out_of_frame
                    "clean"        – also drop sensor_slipped, bounce
                    "reliable"     – keep only files labelled "reliable" in notes
    debug         : print detailed diagnostics at each pipeline stage

    Returns
    -------
    train_loader, val_loader, test_loader

    Each batch yields:
        use_metadata=True  → (X, meta, y)
        use_metadata=False → (X, y)

        X    : float32  (batch, window, 128)      if flat_spatial=True
                        (batch, window, 16, 8)    if flat_spatial=False
        meta : float32  (batch, 5)                only when use_metadata=True
        y    : float32  (batch, window, 6)
    """

    # ── 1. Discover files, filter by quality, then split ────────────────────
    all_files = _discover_files(data_root, preprocessed)

    # Load the run-labels CSV from the same directory as this script
    _csv_path = Path(__file__).parent / "run_labels_notes.csv"
    run_df: Optional[pd.DataFrame] = None
    if _csv_path.exists():
        run_df = _load_run_labels(str(_csv_path))
        if debug:
            print(f"\n[DEBUG] Loaded run-labels CSV: {_csv_path}  "
                  f"({len(run_df)} entries)")
    else:
        print(f"[WARNING] run_labels_notes.csv not found at {_csv_path} — "
              f"skipping quality filter and using run-number stratification")
    # ── 1a. Filter by notes (quality mode) ───────────────────────────────────
    if run_df is not None:
        all_files = _filter_files_by_notes(all_files, run_df, quality, debug=debug)
        print(f"Quality filter (mode='{quality}')  →  {len(all_files)} files kept")
    elif debug:
        print("[DEBUG] run_labels_notes.csv not found — skipping quality filter")

    # ── 1b. Split (stratify by movement label when CSV is available) ─────────
    if run_df is not None:
        train_files, val_files, test_files = _split_files(
            all_files, run_df, train_ratio, val_ratio, seed
        )
    else:
        # Fallback: stratify by run number (original behaviour)
        def _split_files_by_run(files, train_ratio, val_ratio, seed):
            rng = np.random.default_rng(seed)
            run_groups: dict[int, List[Path]] = {}
            for f in files:
                _, run_id = _parse_filename(f)
                run_groups.setdefault(run_id, []).append(f)
            tr, va, te = [], [], []
            for run_id in sorted(run_groups):
                group    = run_groups[run_id]
                idx      = rng.permutation(len(group))
                shuffled = [group[i] for i in idx]
                n_train  = max(1, round(len(shuffled) * train_ratio))
                n_val    = min(max(1, round(len(shuffled) * val_ratio)),
                               len(shuffled) - n_train)
                tr.extend(shuffled[:n_train])
                va.extend(shuffled[n_train : n_train + n_val])
                te.extend(shuffled[n_train + n_val :])
            return tr, va, te
        train_files, val_files, test_files = _split_files_by_run(
            all_files, train_ratio, val_ratio, seed
        )

    print(f"Files  →  train: {len(train_files)}  val: {len(val_files)}  test: {len(test_files)})")

    if debug:
        print("\n[DEBUG] File assignments:")
        split_map = {f: "TRAIN" for f in train_files}
        split_map.update({f: "VAL"   for f in val_files})
        split_map.update({f: "TEST"  for f in test_files})
        for f in all_files:
            sid, rid = _parse_filename(f)
            label = _lookup_label(run_df, sid, rid) if run_df is not None else "—"
            notes = _lookup_notes(run_df, sid, rid) if run_df is not None else "—"
            print(f"  [{split_map[f]}]  {f.name}  "
                  f"(subject={sid}, run={rid}, label='{label}', notes='{notes}')")
        assert not (set(train_files) & set(val_files)),  "[ERROR] Train/val file overlap!"
        assert not (set(train_files) & set(test_files)), "[ERROR] Train/test file overlap!"
        assert not (set(val_files)   & set(test_files)), "[ERROR] Val/test file overlap!"
        print("  [OK] No file overlap between splits.")

    # ── 2. Load raw signals from training files (for scaler fitting) ─────────
    def _load_signals(files, split_name=""):
        result = []
        for f in files:
            sig, lab = _load_csv(f)
            if debug:
                nan_sig = int(np.isnan(sig).sum())
                nan_lab = int(np.isnan(lab).sum())
                warn = "  <<< WARNING: NaN detected!" if (nan_sig or nan_lab) else ""
                print(f"  [{split_name}] {f.name}: signal={sig.shape}  labels={lab.shape}  "
                      f"sig_range=[{sig.min():.3f}, {sig.max():.3f}]  "
                      f"NaN(sig={nan_sig}, lab={nan_lab}){warn}")
            result.append((sig, lab))
        return result

    if debug:
        print("\n[DEBUG] Loading CSVs:")
    train_raw = _load_signals(train_files, "TRAIN")
    val_raw   = _load_signals(val_files,   "VAL")
    test_raw  = _load_signals(test_files,  "TEST")

    # ── 3. Fit signal scaler on train only (optional) ────────────────────────
    sig_scaler = _fit_signal_scaler([sig for sig, _ in train_raw]) if normalize_signal else None

    if debug:
        if normalize_signal:
            print(f"\n[DEBUG] Signal scaler (fit on train only):")
            print(f"  per-sensor mean : min={sig_scaler.mean_.min():.4f}  max={sig_scaler.mean_.max():.4f}")
            print(f"  per-sensor scale: min={sig_scaler.scale_.min():.4f}  max={sig_scaler.scale_.max():.4f}")
            _s0_raw    = train_raw[0][0]
            _s0_scaled = _apply_signal_scaler(_s0_raw[np.newaxis], sig_scaler)[0]
            print(f"  Scaled train[0] : mean={_s0_scaled.mean():.4f}  std={_s0_scaled.std():.4f}  "
                  f"(expected ≈ 0.0, ≈ 1.0)")
        else:
            print(f"\n[DEBUG] Signal scaling disabled (normalize_signal=False).")

    # ── 4. Load metadata (optional) ──────────────────────────────────────────
    meta_scaler = None
    if use_metadata:
        def _load_meta_for_files(files):
            metas = []
            for f in files:
                subject_id, _ = _parse_filename(f)
                raw_meta  = _load_metadata(data_root, subject_id)
                metas.append(_encode_metadata(raw_meta))
            return metas   # list of (5,) arrays, one per file

        train_meta_raw = _load_meta_for_files(train_files)
        val_meta_raw   = _load_meta_for_files(val_files)
        test_meta_raw  = _load_meta_for_files(test_files)

        meta_scaler = _fit_meta_scaler(train_meta_raw)

        if debug:
            print(f"\n[DEBUG] Metadata (train files):")
            print(f"  Fields order: {META_FIELDS}")
            for f, enc in zip(train_files, train_meta_raw):
                print(f"  {f.name}: {enc}")
            print(f"  Scaler mean : {meta_scaler.mean_}")
            print(f"  Scaler scale: {meta_scaler.scale_}")

        def _norm_meta(meta_list):
            stacked = np.stack(meta_list, axis=0)
            return meta_scaler.transform(stacked).astype(np.float32)  # (M, 5)

        train_meta_norm = _norm_meta(train_meta_raw)   # (n_train_files, 5)
        val_meta_norm   = _norm_meta(val_meta_raw)
        test_meta_norm  = _norm_meta(test_meta_raw)

    # ── 5. Window the data ───────────────────────────────────────────────────
    #    Scale signal → window → tag each window with its file's meta vector.
    #    Windowing happens AFTER file-level split → zero leakage.

    def _build_split_arrays(raw_pairs, meta_norm_per_file, split_name="",
                            iqr_fence_bounds=None, file_list=None):
        """
        raw_pairs          : list of (signal(T,16,8), labels(T,6))
        meta_norm_per_file : (n_files, 5) array or None
        iqr_fence_bounds   : (lower, upper) tuple or None — windows outside
                             this range (by unscaled mean) are dropped
        file_list          : List[Path] matching raw_pairs, used for reporting

        Returns (X_all, y_all, meta_all or None)
        """
        _step = max(1, int(window_size * (1 - overlap)))
        if debug:
            print(f"\n[DEBUG] Windowing [{split_name}]  window={window_size}  step={_step}:")
        X_all, y_all, meta_all = [], [], []
        removed_summary = []   # list of (filename, n_removed, [(wi, mean), ...])

        for i, (sig, lab) in enumerate(raw_pairs):
            fname = file_list[i].name if file_list is not None else f"file_{i}"

            # Scale signal for model input (skipped if normalize_signal=False)
            sig_s = _apply_signal_scaler(sig[np.newaxis], sig_scaler)[0] if sig_scaler is not None else sig

            # Window both scaled (model input) and unscaled (IQR check)
            X_win, y_win = _make_windows(sig_s, lab,   window_size, overlap)
            # Also window the raw signal purely for IQR mean calculation
            X_raw, _     = _make_windows(sig,   lab,   window_size, overlap)
            # X_win: (N_w, W, 16, 8)   y_win: (N_w, W, 6)   X_raw: (N_w, W, 16, 8)

            # ── IQR mask ─────────────────────────────────────────────────────
            if iqr_fence_bounds is not None:
                lo, hi = iqr_fence_bounds
                win_means  = X_raw.mean(axis=(1, 2, 3))   # (N_w,) unscaled per-window mean
                keep_mask  = (win_means >= lo) & (win_means <= hi)
                n_removed  = int((~keep_mask).sum())

                if n_removed > 0:
                    bad_indices = [(int(wi), float(win_means[wi]))
                                   for wi in np.where(~keep_mask)[0]]
                    removed_summary.append((fname, n_removed, bad_indices))

                X_win = X_win[keep_mask]
                y_win = y_win[keep_mask]
                if debug and n_removed:
                    print(f"  [{split_name}] {fname}: removed {n_removed} IQR-outlier windows "
                          f"({len(X_win)} kept)")

            if debug:
                print(f"  file {i}: T={len(sig)}  →  {len(X_win)} windows  "
                      f"X_win={X_win.shape}  y_win={y_win.shape}")

            X_all.append(X_win)
            y_all.append(y_win)

            if meta_norm_per_file is not None:
                n_windows  = len(X_win)
                meta_tiled = np.tile(meta_norm_per_file[i], (n_windows, 1))
                meta_all.append(meta_tiled)

        # ── Print removed-window summary (mirrors plot_signal_means style) ───
        if iqr_fence_bounds is not None:
            total_before = sum(len(X) for X in X_all) + sum(n for _, n, _ in removed_summary)
            total_removed = sum(n for _, n, _ in removed_summary)
            print(f"\n  IQR window filter [{split_name}]: "
                  f"removed {total_removed} / {total_before} windows  "
                  f"(fence: [{iqr_fence_bounds[0]:.4f}, {iqr_fence_bounds[1]:.4f}])")
            for fname, n_rem, bad in removed_summary:
                short = (fname.replace("_non_log_preprocessed.csv", "")
                              .replace("_log_preprocessed.csv", ""))
                print(f"    {short}  ({n_rem} window{'s' if n_rem > 1 else ''})")
                for wi, wm in bad:
                    direction = "HIGH" if wm > iqr_fence_bounds[1] else "LOW"
                    print(f"      window {wi:>4}  mean={wm:>8.4f}  [{direction}]")

        X_out    = np.concatenate(X_all,    axis=0).astype(np.float32)
        y_out    = np.concatenate(y_all,    axis=0).astype(np.float32)
        meta_out = (
            np.concatenate(meta_all, axis=0).astype(np.float32)
            if meta_all else None
        )
        return X_out, y_out, meta_out

    train_meta_nf = train_meta_norm if use_metadata else None
    val_meta_nf   = val_meta_norm   if use_metadata else None
    test_meta_nf  = test_meta_norm  if use_metadata else None

    # ── 5b. Fit IQR fence on unscaled train windows (no leakage) ─────────────
    if iqr_multiplier is not None:
        train_window_means = _compute_window_means(train_raw, window_size, overlap)
        iqr_lo, iqr_hi     = _iqr_fence(train_window_means, iqr_multiplier)
        print(f"IQR fence (k={iqr_multiplier}, train unscaled) → [{iqr_lo:.4f}, {iqr_hi:.4f}]  "
              f"({len(train_window_means):,} train windows used to fit)")
        fence = (iqr_lo, iqr_hi)
    else:
        fence = None
        if debug:
            print("\n[DEBUG] IQR window filtering disabled (iqr_multiplier=None).")

    X_train, y_train, meta_train = _build_split_arrays(
        train_raw, train_meta_nf, "TRAIN",
        iqr_fence_bounds=fence, file_list=train_files)
    X_val,   y_val,   meta_val   = _build_split_arrays(
        val_raw,   val_meta_nf,   "VAL",
        iqr_fence_bounds=fence, file_list=val_files)
    X_test,  y_test,  meta_test  = _build_split_arrays(
        test_raw,  test_meta_nf,  "TEST",
        iqr_fence_bounds=fence, file_list=test_files)

    print(f"Windows → train: {len(X_train)}  val: {len(X_val)}  test: {len(X_test)}")
    print(f"X shape (train): {X_train.shape}   →  flat_spatial={flat_spatial}")
    print(f"y shape (train): {y_train.shape}")
    if use_metadata:
        print(f"meta shape (train): {meta_train.shape}")

    # ── 6. Build Datasets and DataLoaders ────────────────────────────────────
    def _make_loader(X, y, meta, shuffle, transform=None):
        ds = SenseMatDataset(X, y, meta, flat_spatial=flat_spatial, transform=transform)
        return DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = shuffle,
            num_workers = num_workers,
            pin_memory  = torch.cuda.is_available(),
        )

    # Initialize our new augmentation class
    train_augmenter = LinkedSenseMatAugmentation(p_flip=0.3, scale_range=(0.8, 1.2), noise_std=0.05)

    # Isolate memory
    X_val_isolated = X_val.copy()
    y_val_isolated = y_val.copy()
    X_test_isolated = X_test.copy()
    y_test_isolated = y_test.copy()

    train_loader = _make_loader(X_train, y_train, meta_train, shuffle=True, transform=train_augmenter)
    val_loader   = _make_loader(X_val_isolated,   y_val_isolated,   meta_val,   shuffle=False, transform=None)
    test_loader  = _make_loader(X_test_isolated,  y_test_isolated,  meta_test,  shuffle=False, transform=None)

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test  (run: python sensemat_dataloader.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    data_root = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent)

    print("=" * 60)
    print("Smoke-test: flat_spatial=True, use_metadata=True")
    print("=" * 60)
    train_dl, val_dl, test_dl = get_dataloaders(
        data_root    = data_root,
        preprocessed = "non_log",
        window_size  = 60,
        flat_spatial = True,
        use_metadata = True,
        batch_size   = 8,
        seed         = 42,
        debug        = True,
    )
    batch = next(iter(train_dl))
    X, meta, y = batch
    print(f"\n  X    : {X.shape}   dtype={X.dtype}")
    print(f"  meta : {meta.shape}   dtype={meta.dtype}")
    print(f"  y    : {y.shape}   dtype={y.dtype}")

    assert not torch.isnan(X).any(),    "[ERROR] NaN in X!"
    assert not torch.isnan(meta).any(), "[ERROR] NaN in meta!"
    assert not torch.isnan(y).any(),    "[ERROR] NaN in y!"
    assert not torch.isinf(X).any(),    "[ERROR] Inf in X!"
    assert X.shape[1] == 60,   f"[ERROR] Expected window dim=60, got {X.shape[1]}"
    assert X.shape[2] == 128,  f"[ERROR] Expected flat spatial=128, got {X.shape[2]}"
    assert y.shape[2] == 6,    f"[ERROR] Expected 6 label dims, got {y.shape[2]}"
    assert meta.shape[1] == 5, f"[ERROR] Expected 5 meta dims, got {meta.shape[1]}"
    print(f"\n  X    stats: mean={X.mean():.4f}  std={X.std():.4f}  "
          f"min={X.min():.4f}  max={X.max():.4f}")
    print(f"  meta stats: mean={meta.mean():.4f}  std={meta.std():.4f}  "
          f"min={meta.min():.4f}  max={meta.max():.4f}")
    print(f"  y    stats: mean={y.mean():.4f}  std={y.std():.4f}  "
          f"min={y.min():.4f}  max={y.max():.4f}")
    print("  [OK] All assertions passed for flat_spatial=True.")

    print()
    print("=" * 60)
    print("Smoke-test: flat_spatial=False, use_metadata=False")
    print("=" * 60)
    train_dl2, _, _ = get_dataloaders(
        data_root    = data_root,
        preprocessed = "log",
        window_size  = 60,
        flat_spatial = False,
        use_metadata = False,
        batch_size   = 8,
        seed         = 42,
        debug        = False,
    )
    batch2 = next(iter(train_dl2))
    X2, y2 = batch2
    print(f"\n  X    : {X2.shape}   dtype={X2.dtype}")
    print(f"  y    : {y2.shape}   dtype={y2.dtype}")

    assert not torch.isnan(X2).any(), "[ERROR] NaN in X2!"
    assert not torch.isnan(y2).any(), "[ERROR] NaN in y2!"
    assert X2.shape[1] == 60, f"[ERROR] Expected window dim=6, got {X2.shape[1]}"
    assert X2.shape[2] == 16, f"[ERROR] Expected 16 rows, got {X2.shape[2]}"
    assert X2.shape[3] == 8,  f"[ERROR] Expected 8 cols, got {X2.shape[3]}"
    assert y2.shape[2] == 6,  f"[ERROR] Expected 6 label dims, got {y2.shape[2]}"
    print(f"\n  X    stats: mean={X2.mean():.4f}  std={X2.std():.4f}  "
          f"min={X2.min():.4f}  max={X2.max():.4f}")
    print("  [OK] All assertions passed for flat_spatial=False.")
