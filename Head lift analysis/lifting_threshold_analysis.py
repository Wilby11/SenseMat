"""
01_recalculate_smean_cleaned_data.py

Purpose
-------
Recalculate S_mean from the 128 SenseMAT sensor columns in Cleaned data.
The existing S_mean column is ignored because it may have been computed before
interpolation and not updated afterwards.

Outputs
-------
Creates analysis_outputs_recalculated_smean/ with:
- metadata_coverage_report.csv
- recording_smean_summary.csv
- recordings_ranked_by_min_smean_ratio.csv
- candidate_unloading_events.csv
- all_samples_recalculated_smean.csv
- global_histogram_smean_recalculated.png
- global_histogram_smean_ratio.png
- boxplot_min_smean_ratio_by_status.png
- selected_recording_plots/*.png

How to run
----------
Put this script in the SenseMat-main folder, next to:
- Cleaned data/
- metadata_head_lifts.csv

Then run:
python 01_recalculate_smean_cleaned_data.py
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, List, Dict
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

# This script is expected to be inside:
# SenseMat-main / Head lift analysis / lifting_threshold_analysis.py

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

CLEANED_DATA_DIR = PROJECT_DIR / "Cleaned data"
METADATA_FILE = BASE_DIR / "metadata_head_lifts.csv"
OUTPUT_DIR = BASE_DIR / "analysis_outputs_recalculated_smean"
PLOTS_DIR = OUTPUT_DIR / "selected_recording_plots"

OUTPUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


# ============================================================
# PARAMETERS
# ============================================================

SMOOTHING_WINDOW = 5

# These are exploratory candidate regions, not final thresholds.
MODERATE_DROP_RATIO = 0.70
STRONG_DROP_RATIO = 0.50
VERY_LOW_RATIO = 0.35

# Minimum duration for an unloading candidate.
# If sampling is around 40 Hz, 8 samples is around 0.2 s.
MIN_DURATION_SAMPLES = 8

# Active sensor helper threshold.
ACTIVE_SENSOR_THRESHOLD = 500


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_subject_run(file_name: str) -> Tuple[int | None, int | None]:
    """
    Extract subject and run from filenames such as:
    subject7_run5_sensemat.csv
    subject10_run1_sensemat.csv
    """
    match = re.search(r"subject(\d+)_run(\d+)", file_name)

    if match is None:
        return None, None

    subject = int(match.group(1))
    run = int(match.group(2))

    return subject, run


def get_metadata_for_file(file_name: str, metadata: pd.DataFrame) -> Tuple[str, str]:
    """
    Match metadata using subject and run instead of exact filename.
    This is more robust than matching the complete filename.
    """
    subject, run = extract_subject_run(file_name)

    if subject is None or run is None:
        return "no_known_lift", "Could not extract subject/run; defaulted to no_known_lift"

    metadata = metadata.copy()
    metadata["subject"] = pd.to_numeric(metadata["subject"], errors="coerce")
    metadata["run"] = pd.to_numeric(metadata["run"], errors="coerce")

    match = metadata[
        (metadata["subject"] == subject) &
        (metadata["run"] == run)
    ]

    if len(match) == 0:
        return "no_known_lift", "Not listed in metadata; defaulted to no_known_lift"

    return str(match["lift_status"].iloc[0]), str(match["comment"].iloc[0])


def find_sensor_columns(df: pd.DataFrame) -> List[str]:
    """
    Find only the 8x16 SenseMAT sensor columns:
    S_0_0 ... S_7_15

    This deliberately excludes the old S_mean column.
    """
    sensor_cols = [
        col for col in df.columns
        if re.fullmatch(r"S_\d+_\d+", col)
    ]

    if len(sensor_cols) != 128:
        raise ValueError(f"Expected 128 sensor columns, found {len(sensor_cols)}")

    # Sort sensors by row and column number.
    sensor_cols = sorted(
        sensor_cols,
        key=lambda c: tuple(map(int, re.findall(r"\d+", c)))
    )

    return sensor_cols


def add_time_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a relative time column.
    Uses RECV_TIME if available, otherwise uses sample index.
    """
    if "RECV_TIME" in df.columns:
        df["RECV_TIME"] = pd.to_numeric(df["RECV_TIME"], errors="coerce")
        first_valid_time = df["RECV_TIME"].dropna().iloc[0]
        df["time_rel_s"] = df["RECV_TIME"] - first_valid_time
    else:
        df["time_rel_s"] = np.arange(len(df))

    return df


def add_recalculated_pressure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalculate pressure features from the 128 current sensor values.

    Important:
    The old S_mean column from Cleaned data is ignored.
    It may have been calculated before interpolation and may be outdated.
    """
    sensor_cols = find_sensor_columns(df)

    sensors = df[sensor_cols].apply(pd.to_numeric, errors="coerce")

    df["S_mean_recalculated"] = sensors.mean(axis=1)
    df["S_sum"] = sensors.sum(axis=1)
    df["S_max"] = sensors.max(axis=1)
    df["S_min"] = sensors.min(axis=1)
    df["active_sensors_500"] = (sensors > ACTIVE_SENSOR_THRESHOLD).sum(axis=1)

    return df


def estimate_baseline_smean(smean: pd.Series) -> float:
    """
    Estimate the normal pressure level of a recording.

    We use the 75th percentile because it is more robust than the maximum.
    The maximum can include short spikes.
    """
    return float(np.nanpercentile(smean, 75))


def add_smean_ratio_and_candidate_state(df: pd.DataFrame, baseline: float) -> pd.DataFrame:
    """
    Smooth S_mean and calculate ratio relative to baseline.

    S_mean_ratio = smoothed S_mean / baseline S_mean
    """
    df["S_mean_smooth"] = (
        df["S_mean_recalculated"]
        .rolling(window=SMOOTHING_WINDOW, center=True, min_periods=1)
        .median()
    )

    df["S_mean_ratio"] = df["S_mean_smooth"] / baseline

    conditions = [
        df["S_mean_ratio"] < VERY_LOW_RATIO,
        df["S_mean_ratio"] < STRONG_DROP_RATIO,
        df["S_mean_ratio"] < MODERATE_DROP_RATIO,
    ]

    labels = [
        "very_low_contact_candidate",
        "strong_unloading_candidate",
        "moderate_unloading_candidate",
    ]

    df["candidate_state"] = np.select(
        conditions,
        labels,
        default="normal_or_supported"
    )

    return df


def extract_candidate_events(df: pd.DataFrame) -> List[Dict]:
    """
    Group consecutive non-normal samples into unloading candidate events.
    """
    candidate_mask = df["candidate_state"] != "normal_or_supported"

    events = []
    in_event = False
    start_idx = None

    for idx, is_candidate in enumerate(candidate_mask):
        if is_candidate and not in_event:
            in_event = True
            start_idx = idx

        elif not is_candidate and in_event:
            end_idx = idx - 1
            events.append((start_idx, end_idx))
            in_event = False

    if in_event:
        events.append((start_idx, len(df) - 1))

    event_rows = []

    for start_idx, end_idx in events:
        duration_samples = end_idx - start_idx + 1

        if duration_samples < MIN_DURATION_SAMPLES:
            continue

        event_data = df.iloc[start_idx:end_idx + 1]

        min_ratio = event_data["S_mean_ratio"].min()

        if min_ratio < VERY_LOW_RATIO:
            event_type = "very_low_contact_candidate"
        elif min_ratio < STRONG_DROP_RATIO:
            event_type = "strong_unloading_candidate"
        else:
            event_type = "moderate_unloading_candidate"

        event_rows.append({
            "start_index": start_idx,
            "end_index": end_idx,
            "start_time_s": event_data["time_rel_s"].iloc[0],
            "end_time_s": event_data["time_rel_s"].iloc[-1],
            "duration_samples": duration_samples,
            "duration_s": event_data["time_rel_s"].iloc[-1] - event_data["time_rel_s"].iloc[0],
            "event_type": event_type,
            "min_S_mean_recalculated": event_data["S_mean_recalculated"].min(),
            "median_S_mean_recalculated": event_data["S_mean_recalculated"].median(),
            "min_S_mean_ratio": min_ratio,
            "min_S_sum": event_data["S_sum"].min(),
            "min_S_max": event_data["S_max"].min(),
            "min_active_sensors_500": event_data["active_sensors_500"].min(),
        })

    return event_rows


def save_recording_plot(df: pd.DataFrame, file_name: str, baseline: float, events: List[Dict]) -> None:
    """
    Save S_mean over time plot for one recording.
    """
    plt.figure(figsize=(12, 5))

    plt.plot(
        df["time_rel_s"],
        df["S_mean_recalculated"],
        linewidth=1,
        label="S_mean recalculated"
    )

    plt.plot(
        df["time_rel_s"],
        df["S_mean_smooth"],
        linewidth=1,
        label="Smoothed S_mean"
    )

    plt.axhline(baseline, linestyle="--", label="Baseline p75")
    plt.axhline(baseline * MODERATE_DROP_RATIO, linestyle="--", label="70% baseline")
    plt.axhline(baseline * STRONG_DROP_RATIO, linestyle="--", label="50% baseline")
    plt.axhline(baseline * VERY_LOW_RATIO, linestyle="--", label="35% baseline")

    for event in events:
        plt.axvspan(
            event["start_time_s"],
            event["end_time_s"],
            alpha=0.2
        )

    plt.xlabel("Time [s]")
    plt.ylabel("S_mean recalculated")
    plt.title(f"Recalculated S_mean unloading candidates — {file_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    safe_name = file_name.replace(".csv", "")
    plt.savefig(PLOTS_DIR / f"{safe_name}_smean_candidates.png", dpi=300)
    plt.close()


def save_global_plots(recording_summary: pd.DataFrame, all_samples: pd.DataFrame) -> None:
    """
    Save global histograms and boxplot.
    """
    # Histogram of recalculated S_mean values
    plt.figure(figsize=(10, 5))
    plt.hist(all_samples["S_mean_recalculated"].dropna(), bins=100)
    plt.xlabel("S_mean recalculated")
    plt.ylabel("Number of samples")
    plt.title("Global histogram of recalculated S_mean")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "global_histogram_smean_recalculated.png", dpi=300)
    plt.close()

    # Histogram of S_mean ratio values
    plt.figure(figsize=(10, 5))
    plt.hist(all_samples["S_mean_ratio"].dropna(), bins=100)
    plt.xlabel("S_mean ratio")
    plt.ylabel("Number of samples")
    plt.title("Global histogram of recalculated S_mean ratio")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "global_histogram_smean_ratio.png", dpi=300)
    plt.close()

    # Boxplot of minimum S_mean ratio by metadata status
    statuses = [
        "no_known_lift",
        "known_lift",
        "possible_lift",
        "special_case",
        "strange_reading",
        "shoulders_visible",
    ]

    data = []
    labels = []

    for status in statuses:
        values = recording_summary.loc[
            recording_summary["lift_status"] == status,
            "min_S_mean_ratio"
        ].dropna()

        if len(values) > 0:
            data.append(values)
            labels.append(status)

    if len(data) > 0:
        plt.figure(figsize=(12, 5))
        plt.boxplot(data, labels=labels)
        plt.ylabel("Minimum S_mean ratio per recording")
        plt.title("Minimum recalculated S_mean ratio by metadata status")
        plt.xticks(rotation=30, ha="right")
        plt.grid(True, axis="y")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "boxplot_min_smean_ratio_by_status.png", dpi=300)
        plt.close()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("Starting recalculated S_mean analysis...")
    print(f"Project directory: {PROJECT_DIR}")
    print(f"Cleaned data directory: {CLEANED_DATA_DIR}")
    print(f"Metadata file: {METADATA_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")

    if not CLEANED_DATA_DIR.exists():
        raise FileNotFoundError(f"Cleaned data folder not found: {CLEANED_DATA_DIR}")

    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")

    metadata = pd.read_csv(METADATA_FILE)

    required_metadata_cols = {"file", "subject", "run", "lift_status", "comment"}
    missing_cols = required_metadata_cols - set(metadata.columns)

    if missing_cols:
        raise ValueError(f"Metadata file is missing columns: {missing_cols}")

    print("\nMetadata status counts:")
    print(metadata["lift_status"].value_counts())

    sensemat_files = sorted(CLEANED_DATA_DIR.glob("*_sensemat.csv"))

    if len(sensemat_files) == 0:
        raise FileNotFoundError(f"No *_sensemat.csv files found in: {CLEANED_DATA_DIR}")

    print(f"\nFound {len(sensemat_files)} SenseMAT files.")

    all_sample_rows = []
    all_event_rows = []
    recording_summary_rows = []
    metadata_coverage_rows = []

    for file_path in sensemat_files:
        file_name = file_path.name

        try:
            df = pd.read_csv(file_path)

            lift_status, comment = get_metadata_for_file(file_name, metadata)

            subject, run = extract_subject_run(file_name)

            print(f"{file_name} -> {lift_status}")

            old_smean_nan_count = None
            old_vs_new_mean_abs_diff = None
            old_vs_new_max_abs_diff = None

            old_smean = None
            if "S_mean" in df.columns:
                old_smean = pd.to_numeric(df["S_mean"], errors="coerce")
                old_smean_nan_count = int(old_smean.isna().sum())

            df = add_time_column(df)
            df = add_recalculated_pressure_features(df)

            if old_smean is not None:
                diff = (df["S_mean_recalculated"] - old_smean).abs()
                old_vs_new_mean_abs_diff = float(diff.mean())
                old_vs_new_max_abs_diff = float(diff.max())

            df = df.dropna(subset=["S_mean_recalculated"]).copy()

            baseline = estimate_baseline_smean(df["S_mean_recalculated"])

            if baseline <= 0 or np.isnan(baseline):
                print(f"Skipping {file_name}: invalid baseline.")
                continue

            df = add_smean_ratio_and_candidate_state(df, baseline)

            events = extract_candidate_events(df)

            # Save samples for global analysis
            sample_export = df[[
                "time_rel_s",
                "S_mean_recalculated",
                "S_mean_smooth",
                "S_mean_ratio",
                "S_sum",
                "S_max",
                "S_min",
                "active_sensors_500",
                "candidate_state",
            ]].copy()

            sample_export["file"] = file_name
            sample_export["subject"] = subject
            sample_export["run"] = run
            sample_export["lift_status"] = lift_status
            sample_export["baseline_S_mean_p75"] = baseline

            all_sample_rows.append(sample_export)

            # Save events
            for event in events:
                event["file"] = file_name
                event["subject"] = subject
                event["run"] = run
                event["lift_status"] = lift_status
                event["comment"] = comment
                event["baseline_S_mean_p75"] = baseline
                all_event_rows.append(event)

            # Recording summary
            recording_summary_rows.append({
                "file": file_name,
                "subject": subject,
                "run": run,
                "lift_status": lift_status,
                "comment": comment,
                "num_samples": len(df),
                "baseline_S_mean_p75": baseline,
                "min_S_mean_recalculated": df["S_mean_recalculated"].min(),
                "median_S_mean_recalculated": df["S_mean_recalculated"].median(),
                "mean_S_mean_recalculated": df["S_mean_recalculated"].mean(),
                "max_S_mean_recalculated": df["S_mean_recalculated"].max(),
                "min_S_mean_ratio": df["S_mean_ratio"].min(),
                "median_S_mean_ratio": df["S_mean_ratio"].median(),
                "num_candidate_events": len(events),
                "num_very_low_events": sum(e["event_type"] == "very_low_contact_candidate" for e in events),
                "num_strong_events": sum(e["event_type"] == "strong_unloading_candidate" for e in events),
                "num_moderate_events": sum(e["event_type"] == "moderate_unloading_candidate" for e in events),
                "old_S_mean_nan_count": old_smean_nan_count,
                "old_vs_new_mean_abs_diff": old_vs_new_mean_abs_diff,
                "old_vs_new_max_abs_diff": old_vs_new_max_abs_diff,
            })

            metadata_coverage_rows.append({
                "file": file_name,
                "subject": subject,
                "run": run,
                "lift_status": lift_status,
                "comment": comment,
                "metadata_found": comment != "Not listed in metadata; defaulted to no_known_lift",
            })

            # Save plot for selected recordings:
            # - all known/possible/special/strange/shoulder cases
            # - no_known_lift only if it has strong drops
            should_plot = True

            if should_plot:
                save_recording_plot(df, file_name, baseline, events)

        except Exception as e:
            print(f"ERROR processing {file_name}: {e}")

    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    if len(all_sample_rows) == 0:
        raise RuntimeError("No sample data was processed.")

    all_samples = pd.concat(all_sample_rows, ignore_index=True)
    all_samples.to_csv(OUTPUT_DIR / "all_samples_recalculated_smean.csv", index=False)

    recording_summary = pd.DataFrame(recording_summary_rows)
    recording_summary.to_csv(OUTPUT_DIR / "recording_smean_summary.csv", index=False)

    ranked = recording_summary.sort_values("min_S_mean_ratio", ascending=True)
    ranked.to_csv(OUTPUT_DIR / "recordings_ranked_by_min_smean_ratio.csv", index=False)

    if len(all_event_rows) > 0:
        candidate_events = pd.DataFrame(all_event_rows)
        candidate_events = candidate_events.sort_values(
            ["min_S_mean_ratio", "duration_s"],
            ascending=[True, False]
        )
    else:
        candidate_events = pd.DataFrame()

    candidate_events.to_csv(OUTPUT_DIR / "candidate_unloading_events.csv", index=False)

    metadata_coverage = pd.DataFrame(metadata_coverage_rows)
    metadata_coverage.to_csv(OUTPUT_DIR / "metadata_coverage_report.csv", index=False)

    save_global_plots(recording_summary, all_samples)

    print("\nDone.")
    print(f"Outputs saved in: {OUTPUT_DIR}")

    print("\nRecording summary by metadata status:")
    print(recording_summary["lift_status"].value_counts())

    print("\nMain files to inspect:")
    print(" - metadata_coverage_report.csv")
    print(" - recording_smean_summary.csv")
    print(" - recordings_ranked_by_min_smean_ratio.csv")
    print(" - candidate_unloading_events.csv")
    print(" - global_histogram_smean_ratio.png")
    print(" - boxplot_min_smean_ratio_by_status.png")


if __name__ == "__main__":
    main()