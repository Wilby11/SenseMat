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
    target_times = sensemat_df[0].dropna().values

    # Build the Interpolation Machine
    # This creates a mathematical function that can predict the position at ANY given microsecond
    # kind='linear' draws straight lines between points.
    # fill_value='extrapolate' allows it to guess safely if a timestamp is slightly off the edges.
    interpolator = interp1d(trackir_times, trackir_data, axis=0, kind='linear', fill_value='extrapolate')
    
    # Feed it the timestamps, and it spits out the exact positions for those times.
    synced_data = interpolator(target_times)
    
    # Save the results to a new, clean CSV
    # Rebuild a new dataframe with the perfectly synced data
    synced_df = pd.DataFrame(synced_data, columns=['X', 'Y', 'Z', 'Pitch', 'Yaw', 'Roll'])
    
    # Insert the perfect timestamps as the very first column
    synced_df.insert(0, 'Unix_Timestamp', target_times)
    
    # Save it to a new CSV file with a clear name
    output_filename = trackir_filepath.replace('_trackir_data', '_synced_to_sensemat')
    synced_df.to_csv(output_filename, sep=';', index=False)
    
    print(f"Success! Synced data saved to {output_filename}.")

if __name__ == "__main__":
    sync_trackir_data()