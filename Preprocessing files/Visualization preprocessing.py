# Loading libraries
import pandas as pd
import numpy as np

# This notebook loads synchronized Sensemat and TrackIR data, predicts the 6 DOF, 
# and the data and prediction in one CSV which will may be passed to `visualization.py`.

# Currently the predicted 6 DOF is simply a copy of the TrackIR data, but in the future 
# this will be replaced by the actual predictions from the AI model.

def Visualization_preprocessing(sensemat_synched_filepath, trackir_synched_filepath):
    # Reading SenseMat and TrackIR data:
    sensemat_file_path = sensemat_synched_filepath
    trackir_file_path = trackir_synched_filepath
    sensemat_df = pd.read_csv(sensemat_file_path, sep=",")
    trackir_df = pd.read_csv(trackir_file_path, sep=";")
    # Check if indeed the timestamps of sensemat and trackir are the same
    assert sensemat_df["RECV_TIME"].equals(trackir_df["Unix_Timestamp"]), "Timestamps do not match!"    
    
    # Load the predicted DOF data (currently just a copy of the TrackIR data, but will be replaced by actual predictions)
    predicted_dof_df = trackir_df.copy()  
    # Rename the columns to indicate these are predictions
    predicted_dof_df.columns = ["Unix_Timestamp", "Predicted_X", "Predicted_Y", "Predicted_Z", "Predicted_Yaw", "Predicted_Pitch", "Predicted_Roll"]
    
    # Combine all data and predictions into one DataFrame
    combined_df = pd.concat([sensemat_df, trackir_df.drop(columns=["Unix_Timestamp"]), predicted_dof_df.drop(columns=["Unix_Timestamp"])], axis=1)
    # Rename the timestamp column to a common name
    combined_df.rename(columns={"RECV_TIME": "Timestamp"}, inplace=True)
    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv("Synched data\\20260422T120945_combined_data.csv", index=False)


if __name__ == "__main__":
    sensemat_synched_filepath = "Synched data\\20260422T120945_sensemat_synched.csv"
    trackir_synched_filepath = "Synched data\\20260422T120945_trackir_synched.csv"
    Visualization_preprocessing(sensemat_synched_filepath, trackir_synched_filepath)