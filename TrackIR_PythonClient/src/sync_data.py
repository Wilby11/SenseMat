import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def sync_trackir_data():    
    # Load the CSV file containing the raw TrackIR data
    input_filename = os.path.join("recordings", "20260323_153457_trackir_data.csv")
    raw_df = pd.read_csv(input_filename, sep=';')
    
    # Extract the absolute Unix timestamps
    original_times = raw_df['Unix_Timestamp'].values
    
    # Extract ONLY the continuous 6DoF data for interpolation
    original_data = raw_df[['X', 'Y', 'Z', 'Pitch', 'Yaw', 'Roll']].values
    
    print(f"Loaded {len(original_times)} frames of raw data.")

    # Build the Interpolation Machine
    # This creates a mathematical function that can predict the position at ANY given microsecond
    # kind='linear' draws straight lines between points.
    # fill_value='extrapolate' allows it to guess safely if a timestamp is slightly off the edges.
    interpolator = interp1d(original_times, original_data, axis=0, kind='linear', fill_value='extrapolate')
    
    target_frequency = 40 
    interval = 1.0 / target_frequency
    
    start_time = original_times[0]
    end_time = original_times[-1]
    
    perfect_40hz_times = np.arange(start_time, end_time, interval)
    
    # Feed it the timestamps, and it spits out the exact positions for those times.
    synced_data = interpolator(perfect_40hz_times)
    
    # Save the results to a new, clean CSV
    # Rebuild a new dataframe with the perfectly synced data
    synced_df = pd.DataFrame(synced_data, columns=['X', 'Y', 'Z', 'Pitch', 'Yaw', 'Roll'])
    
    # Insert the perfect timestamps as the very first column
    synced_df.insert(0, 'Unix_Timestamp', perfect_40hz_times)
    
    # Save it to a new CSV file with a clear name
    output_filename = input_filename.replace('_trackir_data', '_synced_40hz')
    synced_df.to_csv(output_filename, sep=';', index=False)
    
    print(f"Success! Synced data saved to {output_filename}.")

if __name__ == "__main__":
    sync_trackir_data()