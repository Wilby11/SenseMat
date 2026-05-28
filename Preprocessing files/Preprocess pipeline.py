# This preprocessing pipeline does the following:
# 1. For a specified participant's folder: Load the TrackIR and SenseMat data from the CSV files.
# 2. Apply the `Preprocess Sensemat data.py` script to the SenseMat data to clean and preprocess it.
# 3. Apply the `Synchronize_data.py` script to synchronize the TrackIR and SenseMat data based on their timestamps.
#    Additionally, this script will identify the timestamp where the TrackIR data was reset to zero 
#    and trim both datasets to start from that point. It will also trim at the same endpoint.
# 4. Save the synchronized and trimmed datasets to new CSV files in 'Cleaned data' folder for the participant.
# 5. Apply the 'Visualization preprocessing.py' script to prepare the data for visualization. This script collects
#    both TrackIR and Sensemat data, as well as the predicted 6 DOF, into one CSV file for easier visualization in the 'visualization.html' file.

import os
import re
import glob
import pandas as pd
import sys
from pathlib import Path

# Add current directory to path to import modules with spaces in names
sys.path.insert(0, str(Path(__file__).parent))

# Import preprocessing functions
import importlib.util

# Import repair_bad_lines from "Preprocess Sensemat data.py"
spec = importlib.util.spec_from_file_location(
    "preprocess_sensemat_file",
    Path(__file__).parent / "Preprocess Sensemat data.py"
)
preprocess_sensemat_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocess_sensemat_module)
preprocess_sensemat_file = preprocess_sensemat_module.preprocess_sensemat_file

# Import sync_trackir_data from "Synchronize_data.py"
from Synchronize_data import sync_trackir_data


def preprocess_participant_data(participant_folder):
    """
    Preprocess all runs for a participant.
    
    For each matching pair of TrackIR and SenseMat files with the same run number:
    1. Preprocess the SenseMat data
    2. Synchronize TrackIR and SenseMat data
    3. Save synchronized data

    Args:
        participant_folder (str): Path to the participant's folder containing raw data
    """
    participant_path = Path(participant_folder)
    
    # Find all TrackIR files
    trackir_files = sorted(glob.glob(str(participant_path / f"*trackir_data*")))
    
    # Extract run numbers from TrackIR files and find matching SenseMat files
    for trackir_file in trackir_files:
        trackir_path = Path(trackir_file)
        
        # Extract run number from filename (e.g., "subject1_run2" from "..._trackir_data_subject1_run2.csv")
        match = re.search(r'subject(\d+)_run(\d+)', trackir_path.name)
        if not match:
            print(f"Warning: Could not parse run number from {trackir_path.name}")
            continue
        
        subject_num = match.group(1)
        run_num = match.group(2)
        
        # Find corresponding SenseMat file
        # SenseMat format: "...-head-sensemat-subjectX-runY.csv"
        sensemat_pattern = str(participant_path / f"*subject{subject_num}_run*{run_num}-head-sensemat-serial-log.csv")
        # sensemat_pattern = str(participant_path / f"*-head-sensemat-subject{subject_num}-run*{run_num}.csv") # The first couple of runs use this format
        sensemat_files = glob.glob(sensemat_pattern)
        
        if not sensemat_files:
            print(f"Warning: No SenseMat file found for subject{subject_num}_run{run_num}")
            continue
        
        sensemat_file = sensemat_files[0]
        sensemat_path = Path(sensemat_file)
        
        print(f"\nProcessing subject{subject_num}_run{run_num}...")
        print(f"  TrackIR: {trackir_path.name}")
        print(f"  SenseMat: {sensemat_path.name}")
        
        # Step 1: Preprocess SenseMat data
        print(f"  Step 1: Preprocessing SenseMat data...")
        
        try:
            preprocessed_sensemat_df = preprocess_sensemat_file(
                input_path=str(sensemat_path),
                expected_fields=150,
                mean_mode="floor",
                interpolation_method="linear"
            )
            print(f"    ✓ SenseMat preprocessed (in-memory)")
        except Exception as e:
            print(f"    ✗ Error preprocessing SenseMat data: {e}")
            continue
        
        # Step 2 & 3: Synchronize TrackIR and SenseMat data
        print(f"  Step 2-3: Synchronizing TrackIR and SenseMat data...")
        try:
            output_dir = os.path.join("Cleaned data")
            sync_trackir_data(str(trackir_path), preprocessed_sensemat_df, subject_num, run_num, output_dir, reset_roll_threshold=0.01)
            print(f"    ✓ Data synchronized")
        except Exception as e:
            print(f"    ✗ Error synchronizing data: {e}")
            continue

# Example usage:
if __name__ == "__main__":
    for i in range(11,29):
        participant_folder = f"recordings/pn{i}"  # Change to your participant folder
        preprocess_participant_data(participant_folder)    