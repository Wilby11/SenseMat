import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def sync_trackir_data(trackir_filepath, sensemat_filepath):    
    # Load the CSV file containing the raw TrackIR data
    trackir_df = pd.read_csv(trackir_filepath, sep=';')

    # Extract the absolute Unix timestamps and the 6DoF data
    trackir_times = trackir_df['Unix_Timestamp'].values
    trackir_data = trackir_df[['X', 'Y', 'Z', 'Pitch', 'Yaw', 'Roll']].values

    # Load the CSV file containing the raw SenseMat data
    # Skip the config row and the header row (skiprows=2)
    # Provide 200 integer column names (names=range(200)) to catch all the extra commas safely
    # 200 is an arbitrary large number to ensure we capture all columns without error as we will only use the first one
    sensemat_df = pd.read_csv(sensemat_filepath, skiprows=2, header=None, names=range(200), low_memory=False)

    # Extract the target timestamps we want to align to
    sensemat_times = sensemat_df[0].dropna().values

    # Check if the TrackIR timestamps overlap with the SenseMat timestamps
    assert (trackir_times[0] <= sensemat_times[-1]) & (trackir_times[-1] >= sensemat_times[0]), "TrackIR data does not overlap with SenseMat data. Please check the timestamps and ensure they are from the same recording session."

    # Build the Interpolation Machine
    # This creates a mathematical function that can predict the position at ANY given microsecond
    # kind='linear' draws straight lines between points.
    # fill_value='extrapolate' allows it to guess safely if a timestamp is slightly off the edges.
    interpolator = interp1d(trackir_times, trackir_data, axis=0, kind='linear', fill_value='extrapolate')

    # Feed it the timestamps, and it spits out the exact positions for those times.
    synched_data = interpolator(sensemat_times)

    # Save the results to a new, clean CSV
    # Rebuild a new dataframe with the perfectly synced data
    synced_df = pd.DataFrame(synched_data, columns=['X', 'Y', 'Z', 'Pitch', 'Yaw', 'Roll'])

    # Insert the perfect timestamps as the very first column
    synced_df.insert(0, 'Unix_Timestamp', sensemat_times)

    # Find the timestamp where we reset the TrackIR data to zero (approximately zero)
    TrackIR_reset_row = synced_df[(abs(synced_df['X']) < 0.0001) & (abs(synced_df['Y']) < 0.0001) & (abs(synced_df['Z']) < 0.001) & (abs(synced_df['Pitch']) < 0.0001) & (abs(synced_df['Yaw']) < 0.0001) & (abs(synced_df['Roll']) < 0.0001)].iloc[0]
    TrackIR_rest_Unix = TrackIR_reset_row["Unix_Timestamp"]

    # Find the end of the recording, i.e. the minimum last timestamp between TrackIR and SenseMat
    end_timestamp = min(trackir_times[-1], sensemat_times[-1])

    # Remove all data points before the reset for TrackIR
    trackir_synched_trimmed = synced_df[(synced_df["Unix_Timestamp"] >= TrackIR_rest_Unix) & (synced_df["Unix_Timestamp"] <= end_timestamp)]

    # Remove all data points before the reset for SenseMat
    sensemat_trimmed = sensemat_df[(sensemat_df[0] >= TrackIR_rest_Unix) & (sensemat_df[0] <= end_timestamp)]

    # Save both to 'Synched data' folder in the global repository with SenseMat-based filenames
    script_dir = os.path.dirname(os.path.abspath(__file__))
    synched_dir = os.path.join(script_dir, '..', 'Synched data')
    os.makedirs(synched_dir, exist_ok=True)

    # Extract the timestamp prefix from SenseMat filename (everything before '-head-sensemat')
    sensemat_base = os.path.basename(sensemat_filepath)
    prefix = sensemat_base.split('-head-sensemat')[0]

    # TrackIR output
    trackir_output = os.path.join(synched_dir, f"{prefix}_trackir_synched.csv")
    trackir_synched_trimmed.to_csv(trackir_output, sep=';', index=False)

    # SenseMat output
    sensemat_output = os.path.join(synched_dir, f"{prefix}_sensemat_synched.csv")
    sensemat_trimmed.to_csv(sensemat_output, index=False)

    print(f"Success! Synced TrackIR data saved to {trackir_output}.")
    print(f"Success! Trimmed SenseMat data saved to {sensemat_output}.")

if __name__ == "__main__":
    trackir_file_path = "recordings\\Recorded data 22-04\\20260422_121324_trackir_data (Wilbur long).csv"
    sensemat_file_path = "recordings\\Recorded data 22-04\\processed_sensemat_data\\20260422T120945-head-sensemat-serial-log_(WilburLong)_processed.csv"
    sync_trackir_data(trackir_file_path, sensemat_file_path)