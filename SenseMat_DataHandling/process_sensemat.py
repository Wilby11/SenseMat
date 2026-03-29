from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

def load_raw_lines(path):
    """
    Read the file as raw text and return it as a list of lines.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read().splitlines()


def parse_sensemat_file(path):
    """
    Read the SenseMat CSV file and turn it into a DataFrame.

    This function:
    - reads all lines from the file
    - detects whether there is a config line at the top
    - reads the header
    - removes blank data lines
    - counts commas in each data line
    - trims or pads each row so it matches the header width
    - converts values to numbers where possible

    It returns the config line, header list, cleaned DataFrame,
    the comma count for each non-empty data line, and the field count
    for each non-empty data line.
    """
    lines = load_raw_lines(path)

    # Some files start with a config/comment line beginning with '#'
    config_line = lines[0] if lines[0].startswith("#") else None
    header_idx = 1 if config_line else 0

    # Read the header row and use it to define the expected width of each row
    header = lines[header_idx].split(",")
    header_width = len(header)

    # Everything after the header is data
    raw_data_lines = lines[header_idx + 1:]
    nonempty_lines = [line for line in raw_data_lines if line.strip() != ""]
    comma_counts = [line.count(",") for line in nonempty_lines]

    records = []
    field_counts = []

    for line in nonempty_lines:
        parts = line.split(",")

        # Keep track of how many values each row has
        field_counts.append(len(parts))

        # IMPORTANT:
        # we do NOT pad short rows 
        # we only trim rows that are too long
        #(because pandas cannot handle extra columns beyond the header)
        if len(parts) > header_width:
            parts = parts[:header_width]

        # If a row is shorter than the header, pandas will automatically
        # fill the missing values with NaN
        records.append(parts)

    df = pd.DataFrame(records, columns=header)

    # Try converting every column to numeric values
    # Any value that cannot be converted becomes NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return config_line, header, df, comma_counts, field_counts


def get_s_columns(df):
    """
    Return the names of all sensor columns.

    These are the columns that start with 'S_' except for stats like 'S_mean' and others,
    so only columns like S_0_0, S_0_1, ..., S_7_15.

    """
    s_cols = []

    for col in df.columns:
        parts = col.split("_")

        if len(parts) == 3 and parts[0] == "S":
            if parts[1].isdigit() and parts[2].isdigit():
                s_cols.append(col)

    return s_cols


def calculate_s_mean(df, mode="floor"):
    """
    Calculate the mean across all sensor value columns for each row.

    The mode controls how the mean is returned:
    - 'exact': keep the original decimal mean
    - 'round': round to the nearest integer
    - 'floor': round down
    - 'ceil': round up
    """
    s_cols = get_s_columns(df)
    mean_vals = df[s_cols].mean(axis=1)

    if mode == "exact":
        return mean_vals
    if mode == "round":
        return mean_vals.round()
    if mode == "floor":
        return np.floor(mean_vals)
    if mode == "ceil":
        return np.ceil(mean_vals)


def add_validation_columns(df, comma_counts, field_counts, expected_commas=150, mode="floor"):
    """
    Add columns that help validate each row.

    This function calculates the sensor mean again from the raw S columns,
    compares it to the existing S_mean column, checks whether the row has
    the expected number of commas, and checks whether the row length matches
    the header length.
    """
    out = df.copy()

    # Store how many fields each raw row had before any trimming
    out["field_count"] = field_counts

    # Check whether the raw row length matches the header length
    header_width = len(df.columns)
    out["length_match"] = out["field_count"] == header_width

    # Store both the raw mean and the mean after applying the chosen mode
    out["S_mean_calc_raw"] = df[get_s_columns(df)].mean(axis=1)
    out["S_mean_calc"] = calculate_s_mean(df, mode)

    # Check whether the calculated mean matches the given S_mean column
    out["S_mean_match"] = out["S_mean_calc"] == out["S_mean"]

    # Check whether the raw line had the expected number of commas
    out["comma_count"] = comma_counts
    out["comma_match"] = out["comma_count"] == expected_commas

    return out



def repair_bad_lines(path, output_path, expected_commas=150, mode="floor", method="linear"):
    """
    Find bad rows and replace them using interpolation.

    A row is considered bad if:
    1. the calculated sensor mean does not match S_mean, or
    2. the line does not have the expected number of commas, or
    3. the number of fields in the raw row does not match the header length

    For now, bad numeric values are replaced by interpolation between the
    nearest valid rows. The interpolation method can be changed with the
    'method' argument.
    """
    config, header, df, comma_counts, field_counts = parse_sensemat_file(path)
    df = add_validation_columns(df, comma_counts, field_counts, expected_commas, mode)

    # Mark rows that fail at least one of the checks
    df["bad"] = (~df["S_mean_match"]) | (~df["comma_match"]) | (~df["length_match"])

    # We only interpolate numeric columns
    s_cols = get_s_columns(df)

    clean = df.copy()
    clean[s_cols] = clean[s_cols].astype(float)
    
    # Remove values in bad rows so interpolation can fill them in
    clean.loc[df["bad"], s_cols] = np.nan

    # Fill missing values using interpolation based on surrounding rows
    clean[s_cols] = clean[s_cols].interpolate(method=method, limit_direction="both")

    # After interpolation, update the columns that depend on the sensor values
    # so they reflect the repaired data rather than the original broken row
    clean["S_mean"] = calculate_s_mean(clean, mode)
    clean["S_mean_calc_raw"] = clean[s_cols].mean(axis=1)
    clean["S_mean_calc"] = calculate_s_mean(clean, mode)
    clean["S_mean_match"] = clean["S_mean_calc"] == clean["S_mean"]

    # These structural checks are based on the original raw lines,
    # so they do not change after interpolation
    clean["comma_count"] = df["comma_count"]
    clean["comma_match"] = df["comma_match"]
    clean["field_count"] = df["field_count"]
    clean["length_match"] = df["length_match"]

    # Write the repaired file back out using the original header order
    with open(output_path, "w") as f:
        if config:
            f.write(config + "\n")
        f.write(",".join(header) + "\n")
        clean[header].to_csv(f, index=False, header=False)
    
    return clean

def evaluate_40hz(df, col="RECV_TIME", expected_hz=40, plot=True):
    """
    Check how closely the recording matches an expected sampling rate.

    This uses the timestamp column to compute the time difference between
    consecutive rows, then converts that into an estimated frequency.

    If plot=True, it also shows:
    - the time gap between samples over time
    - the distribution of those time gaps
    """
    # Convert timestamps to numeric and compute the difference between rows
    timestamps = pd.to_numeric(df[col], errors="coerce").dropna()
    # If B_TIME is used (which is in milliseconds), convert to seconds
    if col.upper() == "B_TIME":
        timestamps = timestamps / 1000000
    dt = timestamps.diff().dropna()

    mean_dt = dt.mean()
    median_dt = dt.median()

    result = {
        "mean_hz": 1 / mean_dt,
        "median_hz": 1 / median_dt,
        "std_dt": dt.std(),
    }

    if plot:
        # Plot 1: how the sample spacing changes over time
        plt.figure()
        plt.plot(dt.values)
        plt.axhline(1 / expected_hz, linestyle="--")
        plt.title("Timing over time")
        plt.xlabel("Sample index")   
        plt.ylabel("Δt (seconds)")   

        # Plot 2: histogram of sample spacing values
        plt.figure()
        plt.hist(dt.values, bins=50)
        plt.axvline(1 / expected_hz, linestyle="--")
        plt.title("Timing distribution")
        plt.xlabel("Δt (seconds)")            
        plt.ylabel("Count") 

        plt.show()

    return result

#only used in when running the file to validate valid parsing
def count_commas_per_line(path):
    """
    Count how many commas appear in each data line of the file.

    This is useful for checking whether each row has the expected structure.
    Blank lines are included in the output and marked separately.
    """
    lines = load_raw_lines(path)

    # Skip the config line and header if a config line exists,
    # otherwise skip only the header
    start = 2 if lines[0].startswith("#") else 1

    results = []
    for i, line in enumerate(lines[start:], start=start):
        results.append({
            "line_number": i,
            "comma_count": line.count(","),
            "is_blank": line.strip() == "",
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    """
    Simple entry point so you can run this file directly.

    Just change the file paths below and run:
        python your_script.py
    """

    # ---- SETTINGS ----
    input_path = "recordings/20260323T153346-head-sensemat-serial-log.csv" 
    input_path_obj = Path(input_path)

    output_dir = input_path_obj.parent / "processed_sensemat_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / (input_path_obj.stem + "_processed" + input_path_obj.suffix)    

    expected_commas = 150
    mean_mode = "floor"                   # "floor" works for your data
    interpolation_method = "linear"

    # ---- STEP 1: inspect comma counts ----
    print("--- Checking comma counts ---")
    comma_df = count_commas_per_line(input_path)
    print(comma_df["comma_count"].value_counts().sort_index())

    # ---- STEP 2: load and validate data ----
    print("--- Parsing and validating data ---")
    config, header, df, comma_counts, field_counts = parse_sensemat_file(input_path)
    df = add_validation_columns(df, comma_counts, field_counts, expected_commas, mean_mode)

    print("Total rows:", len(df))
    print("Mean matches:", df["S_mean_match"].sum())
    print("Mean mismatches:", (~df["S_mean_match"]).sum())

    # ---- STEP 3: repair bad rows ----
    print("--- Repairing bad rows ---")
    repaired_df = repair_bad_lines(
        input_path,
        str(output_path),
        expected_commas,
        mean_mode,
        interpolation_method,
    )
    print("Repaired file saved to:", output_path)

    # ---- STEP 4: evaluate timing (40 Hz check) ----
    print("\n--- Evaluating frequency (RECV_TIME) ---")
    print(evaluate_40hz(repaired_df, col="RECV_TIME", plot=True))

    print("\n--- Evaluating frequency (B_TIME) ---")
    print(evaluate_40hz(repaired_df, col="B_TIME", plot=True))