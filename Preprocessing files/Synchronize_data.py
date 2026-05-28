import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def sync_trackir_data(
    trackir_filepath,
    sensemat_filepath,
    subject=None,
    run=None,
    output_dir=None,
    reset_roll_threshold=0.0002):
    """
    Synchronize TrackIR data to SenseMat timestamps.

    Steps:
    1. Load TrackIR and SenseMat files.
    2. Detect TrackIR recenter event (Roll ~= 0).
    3. Trim both datasets:
        - start = TrackIR recenter timestamp
        - end   = minimum final timestamp
    4. Interpolate TrackIR data onto SenseMat timestamps.
    5. Preserve the first TrackIR row after recentering
       (avoid interpolation across the discontinuity).
    """

    # ------------------------------------------------------------------
    # Load SenseMat
    # ------------------------------------------------------------------

    if isinstance(sensemat_filepath, str):
        sensemat_df = pd.read_csv(
            sensemat_filepath,
            sep=",",
            comment="#",
            header=0,
            usecols=[0] + list(range(3, 131)),
            low_memory=False)
    else:
        sensemat_df = sensemat_filepath.copy()
        sensemat_df = sensemat_df.iloc[:, [1] + list(range(4, 133))]

    # ------------------------------------------------------------------
    # Load TrackIR
    # ------------------------------------------------------------------
    trackir_df = pd.read_csv(trackir_filepath, sep=";")
    trackir_columns = ["X", "Y", "Z", "Pitch", "Yaw", "Roll"]

    # ------------------------------------------------------------------
    # Extract timestamps
    # ------------------------------------------------------------------
    sensemat_times = sensemat_df["RECV_TIME"].to_numpy()
    trackir_times = trackir_df["Unix_Timestamp"].to_numpy()

    # ------------------------------------------------------------------
    # Detect TrackIR recenter point
    # ------------------------------------------------------------------
    reset_idx = np.where(np.abs(trackir_df["Roll"].to_numpy()) < reset_roll_threshold)[0] # first row where Roll is approx zero

    if len(reset_idx) == 0:
        raise ValueError("Could not detect TrackIR recenter point.")

    # reset_idx = reset_idx[0]
    reset_time = trackir_df.iloc[reset_idx]["Unix_Timestamp"]

    # ------------------------------------------------------------------
    # Determine common recording interval
    # ------------------------------------------------------------------
    end_time = min(trackir_times[-1], sensemat_times[-1]) # minimum between their final recording timestamps

    # ------------------------------------------------------------------
    # Trim TrackIR
    # IMPORTANT:
    # Start EXACTLY at recenter point to avoid interpolation
    # across the discontinuity before recentering.
    # ------------------------------------------------------------------
    trackir_trimmed = trackir_df.iloc[reset_idx:].copy()
    trackir_trimmed = trackir_trimmed[trackir_trimmed["Unix_Timestamp"] <= end_time]

    # ------------------------------------------------------------------
    # Trim SenseMat
    # ------------------------------------------------------------------
    sensemat_trimmed = sensemat_df[(sensemat_df["RECV_TIME"] >= reset_time) & (sensemat_df["RECV_TIME"] <= end_time)].copy()
    sensemat_trimmed = sensemat_trimmed.astype(float) 
    if len(sensemat_trimmed) == 0:
        raise ValueError("No overlapping SenseMat data after recentering.")

    # ------------------------------------------------------------------
    # Build interpolator
    # ------------------------------------------------------------------
    interp_times = trackir_trimmed["Unix_Timestamp"].to_numpy()
    interp_values = trackir_trimmed[trackir_columns].to_numpy()
    interpolator = interp1d(interp_times, interp_values, axis=0, kind="linear", bounds_error=False, fill_value="extrapolate")

    # ------------------------------------------------------------------
    # Interpolate TrackIR onto SenseMat timestamps
    # ------------------------------------------------------------------
    target_times = sensemat_trimmed["RECV_TIME"].to_numpy()
    synced_values = interpolator(target_times)

    # ------------------------------------------------------------------
    # Preserve first TrackIR recentered row
    # Avoid interpolation over discontinuity
    # ------------------------------------------------------------------
    synced_values[0] = interp_values[0]

    # ------------------------------------------------------------------
    # Build synchronized TrackIR dataframe
    # ------------------------------------------------------------------
    trackir_synced_df = pd.DataFrame(synced_values, columns=trackir_columns)
    trackir_synced_df.insert(0, "Unix_Timestamp", target_times)

    # ------------------------------------------------------------------
    # Optional:
    # replace first SenseMat row by average before reset
    # ------------------------------------------------------------------
    pre_reset = sensemat_df[sensemat_df["RECV_TIME"] < reset_time] # The SenseMat data before recenter point

    if len(pre_reset) >= 20:
        mean_row = pre_reset.iloc[-20:, 1:].mean()
        sensemat_trimmed.iloc[0, 1:] = mean_row.values

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        trackir_output = os.path.join(output_dir, f"subject{subject}_run{run}_trackir.csv")
        sensemat_output = os.path.join(output_dir, f"subject{subject}_run{run}_sensemat.csv")
        trackir_synced_df.to_csv(trackir_output, sep=";", index=False)
        sensemat_trimmed.to_csv(sensemat_output, index=False)
        print(f"Saved TrackIR:  {trackir_output}")
        print(f"Saved SenseMat: {sensemat_output}")
    return trackir_synced_df, sensemat_trimmed


##############################################
##### OLD IMPLEMENTATION: #####
##############################################

# def sync_trackir_data(trackir_filepath, sensemat_filepath, subject, run, output_dir=None):    
#     # Load the CSV file containing the raw TrackIR data
#     trackir_df = pd.read_csv(trackir_filepath, sep=';')

#     # Extract the absolute Unix timestamps and the 6DoF data
#     trackir_times = trackir_df['Unix_Timestamp'].values
#     trackir_data = trackir_df[['X', 'Y', 'Z', 'Pitch', 'Yaw', 'Roll']].values

#     # Load the SenseMat data: accept either a file path or a DataFrame
#     if isinstance(sensemat_filepath, str):
#         # Load from CSV file
#         sensemat_df = pd.read_csv(sensemat_filepath, sep=",", comment="#", header=0, usecols=[0]+list(range(3,131)), low_memory=False)
#     else:
#         # Assume it's already a DataFrame (in-memory)
#         sensemat_df = sensemat_filepath

#     # Extract the target timestamps we want to align to
#     sensemat_times = sensemat_df["RECV_TIME"].dropna().values

#     # Check if the TrackIR timestamps overlap with the SenseMat timestamps
#     assert (trackir_times[0] <= sensemat_times[-1]) & (trackir_times[-1] >= sensemat_times[0]), "TrackIR data does not overlap with SenseMat data. Please check the timestamps and ensure they are from the same recording session."

#     # Build the Interpolation Machine
#     # This creates a mathematical function that can predict the position at ANY given microsecond
#     # kind='linear' draws straight lines between points.
#     # fill_value='extrapolate' allows it to guess safely if a timestamp is slightly off the edges.
#     interpolator = interp1d(trackir_times, trackir_data, axis=0, kind='linear', fill_value='extrapolate')

#     # Feed it the timestamps, and it spits out the exact positions for those times.
#     synched_data = interpolator(sensemat_times)

#     # Save the results to a new, clean CSV
#     # Rebuild a new dataframe with the perfectly synced data
#     synced_df = pd.DataFrame(synched_data, columns=['X', 'Y', 'Z', 'Pitch', 'Yaw', 'Roll'])

#     # Insert the perfect timestamps as the very first column
#     synced_df.insert(0, 'Unix_Timestamp', sensemat_times)

#     # Find the timestamp where we reset the TrackIR data to zero (approximately zero)
#     TrackIR_reset_row = trackir_df[(abs(trackir_df['Roll']) < 0.0005)].iloc[0]
#     TrackIR_rest_Unix = TrackIR_reset_row["Unix_Timestamp"]

#     # Find the end of the recording, i.e. the minimum last timestamp between TrackIR and SenseMat
#     end_timestamp = min(trackir_times[-1], sensemat_times[-1])

#     # Remove all data points before the reset for TrackIR
#     trackir_synched_trimmed = synced_df[(synced_df["Unix_Timestamp"] >= TrackIR_rest_Unix) & (synced_df["Unix_Timestamp"] <= end_timestamp)]

#     # Remove all data points before the reset for SenseMat
#     sensemat_trimmed = sensemat_df[(sensemat_df["RECV_TIME"] >= TrackIR_rest_Unix) & (sensemat_df["RECV_TIME"] <= end_timestamp)]

#     os.makedirs(output_dir, exist_ok=True)
#     trackir_output = os.path.join(output_dir, f"subject{subject}_run{run}_trackir.csv")
#     sensemat_output = os.path.join(output_dir, f"subject{subject}_run{run}_sensemat.csv") 
    
#     # Save the synced data to the provided output directory
#     trackir_synched_trimmed.to_csv(trackir_output, sep=';', index=False)
#     sensemat_trimmed.to_csv(sensemat_output, index=False)
    
#     print(f"Success! Synced TrackIR data saved to {trackir_output}.")
#     print(f"Success! Trimmed SenseMat data saved to {sensemat_output}.")