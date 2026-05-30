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
    flat_spatial    = True,           # True → (N, window, 128), False → (N, window, 16, 8)
    use_metadata    = True,           # include subject metadata in batch
    batch_size      = 32,
    train_ratio     = 0.70,
    val_ratio       = 0.15,           # test_ratio is inferred as 1 - train_ratio - val_ratio
    seed            = 42,
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
    files: List[Path],
    train_ratio: float,
    val_ratio:   float,
    seed:        int,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Stratified file-level split.

    Strategy
    --------
    1. Group files by run number (run 1 = slow, run 3 = fast, etc.).
    2. Within each run-group, shuffle by subject using the fixed seed and
       assign files to train / val / test proportionally.
    3. This guarantees that every run speed (movement style) is represented
       in all three splits, and that windows from the same subject+run file
       stay entirely in one split (no data leakage).

    Different subjects' runs may land in different splits, which is fine
    (e.g. subject9_run1 → train, subject9_run2 → val).
    """
    assert train_ratio + val_ratio < 1.0, "train + val ratios must be < 1.0"
    rng = np.random.default_rng(seed)

    # Group by run number
    run_groups: dict[int, List[Path]] = {}
    for f in files:
        _, run_id = _parse_filename(f)
        run_groups.setdefault(run_id, []).append(f)

    train_files, val_files, test_files = [], [], []

    for run_id in sorted(run_groups):
        group = run_groups[run_id]
        # Shuffle deterministically within each run-group
        idx = rng.permutation(len(group))
        shuffled = [group[i] for i in idx]

        n_train = max(1, round(len(shuffled) * train_ratio))
        n_val   = max(1, round(len(shuffled) * val_ratio))
        # Protect against overlap when group is very small
        n_val   = min(n_val, len(shuffled) - n_train)
        n_test  = len(shuffled) - n_train - n_val

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

    # ── Signal → (T, 16, 8) ──────────────────────────────────────────────────
    # reshape(-1, N_COLS, N_ROWS) groups by mat-col first, then .transpose(0,2,1) gives
    # array[t, row, col] = S_{col}_{row}: portrait layout, 16 rows x 8 cols.
    signal = df[SIGNAL_COLS].values.astype(np.float32)      # (T, 128)
    signal = signal.reshape(-1, N_COLS, N_ROWS).transpose(0, 2, 1)  # (T, 16, 8)

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
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

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
    ):
        self.flat_spatial = flat_spatial
        self.X    = torch.from_numpy(X_windows)    # (N, W, 16, 8)
        self.y    = torch.from_numpy(y_windows)    # (N, W, 6)
        self.meta = (
            torch.from_numpy(meta_vectors)         # (N, 5)
            if meta_vectors is not None else None
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]                            # (W, 16, 8)

        if self.flat_spatial:
            # LSTM / RNN / Transformer: (W, 128)
            x = x.reshape(x.shape[0], -1)

        # CNN-1D note: caller should permute to (128, W) inside the model or
        # add a .permute(0,2,1) in the model's forward() — not done here so
        # that the time axis stays consistent across all model types.

        y = self.y[idx]                            # (W, 6)

        if self.meta is not None:
            return x, self.meta[idx], y
        return x, y


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    data_root:      str,
    preprocessed:   Literal["non_log", "log"] = "non_log",
    window_size:    int   = 64,
    overlap:        float = 0.5,
    flat_spatial:   bool  = True,
    use_metadata:   bool  = True,
    batch_size:     int   = 32,
    train_ratio:    float = 0.70,
    val_ratio:      float = 0.15,
    seed:           int   = 42,
    num_workers:    int   = 0,
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
    seed          : random seed for reproducible splits
    num_workers   : DataLoader worker processes

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

    # ── 1. Discover and split files ──────────────────────────────────────────
    all_files = _discover_files(data_root, preprocessed)
    train_files, val_files, test_files = _split_files(
        all_files, train_ratio, val_ratio, seed
    )
    print(f"Files  →  train: {len(train_files)}  val: {len(val_files)}  test: {len(test_files)}")

    # ── 2. Load raw signals from training files (for scaler fitting) ─────────
    def _load_signals(files):
        return [_load_csv(f) for f in files]   # list of (signal(T,8,16), labels(T,6))

    train_raw = _load_signals(train_files)
    val_raw   = _load_signals(val_files)
    test_raw  = _load_signals(test_files)

    # ── 3. Fit signal scaler on train only ───────────────────────────────────
    sig_scaler = _fit_signal_scaler([sig for sig, _ in train_raw])

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

        def _norm_meta(meta_list):
            stacked = np.stack(meta_list, axis=0)
            return meta_scaler.transform(stacked).astype(np.float32)  # (M, 5)

        train_meta_norm = _norm_meta(train_meta_raw)   # (n_train_files, 5)
        val_meta_norm   = _norm_meta(val_meta_raw)
        test_meta_norm  = _norm_meta(test_meta_raw)

    # ── 5. Window the data ───────────────────────────────────────────────────
    #    Scale signal → window → tag each window with its file's meta vector.
    #    Windowing happens AFTER file-level split → zero leakage.

    def _build_split_arrays(raw_pairs, meta_norm_per_file):
        """
        raw_pairs         : list of (signal(T,8,16), labels(T,6))
        meta_norm_per_file: (n_files, 5) array or None

        Returns (X_all, y_all, meta_all or None)
        """
        X_all, y_all, meta_all = [], [], []

        for i, (sig, lab) in enumerate(raw_pairs):
            # Scale signal
            sig_s = _apply_signal_scaler(sig[np.newaxis], sig_scaler)[0]   # (T,8,16)

            # Window
            X_win, y_win = _make_windows(sig_s, lab, window_size, overlap)
            # X_win: (N_w, W, 16, 8)   y_win: (N_w, W, 6)
            X_all.append(X_win)
            y_all.append(y_win)

            if meta_norm_per_file is not None:
                # Broadcast file's meta vector to all windows from that file
                n_windows = len(X_win)
                meta_tiled = np.tile(meta_norm_per_file[i], (n_windows, 1))  # (N_w, 5)
                meta_all.append(meta_tiled)

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

    X_train, y_train, meta_train = _build_split_arrays(train_raw, train_meta_nf)
    X_val,   y_val,   meta_val   = _build_split_arrays(val_raw,   val_meta_nf)
    X_test,  y_test,  meta_test  = _build_split_arrays(test_raw,  test_meta_nf)

    print(f"Windows → train: {len(X_train)}  val: {len(X_val)}  test: {len(X_test)}")
    print(f"X shape (train): {X_train.shape}   →  flat_spatial={flat_spatial}")
    print(f"y shape (train): {y_train.shape}")
    if use_metadata:
        print(f"meta shape (train): {meta_train.shape}")

    # ── 6. Build Datasets and DataLoaders ────────────────────────────────────
    def _make_loader(X, y, meta, shuffle):
        ds = SenseMatDataset(X, y, meta, flat_spatial=flat_spatial)
        return DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = shuffle,
            num_workers = num_workers,
            pin_memory  = torch.cuda.is_available(),
        )

    train_loader = _make_loader(X_train, y_train, meta_train, shuffle=True)
    val_loader   = _make_loader(X_val,   y_val,   meta_val,   shuffle=False)
    test_loader  = _make_loader(X_test,  y_test,  meta_test,  shuffle=False)

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test  (run: python sensemat_dataloader.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    data_root = sys.argv[1] if len(sys.argv) > 1 else "./data"

    print("=" * 60)
    print("Smoke-test: flat_spatial=True, use_metadata=True")
    print("=" * 60)
    train_dl, val_dl, test_dl = get_dataloaders(
        data_root    = data_root,
        preprocessed = "non_log",
        window_size  = 64,
        flat_spatial = True,
        use_metadata = True,
        batch_size   = 8,
        seed         = 42,
    )
    batch = next(iter(train_dl))
    X, meta, y = batch
    print(f"  X    : {X.shape}   dtype={X.dtype}")
    print(f"  meta : {meta.shape}   dtype={meta.dtype}")
    print(f"  y    : {y.shape}   dtype={y.dtype}")

    print()
    print("=" * 60)
    print("Smoke-test: flat_spatial=False, use_metadata=False")
    print("=" * 60)
    train_dl2, _, _ = get_dataloaders(
        data_root    = data_root,
        preprocessed = "non_log",
        window_size  = 64,
        flat_spatial = False,
        use_metadata = False,
        batch_size   = 8,
        seed         = 42,
    )
    batch2 = next(iter(train_dl2))
    X2, y2 = batch2
    print(f"  X    : {X2.shape}   dtype={X2.dtype}")
    print(f"  y    : {y2.shape}   dtype={y2.dtype}")
