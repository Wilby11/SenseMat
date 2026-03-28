# TrackIR data collection and synchronization pipeline
This repository contains the python tools required to capture high-speed, 6 Degrees of Freedom (6DoF) head-tracking data from the TrackIR hardware and synchronize it to any specific frequency timeline for sensor fusion with the Sensemat system.

### Prerequisites
Before running these scripts, ensure you have:

* The official TrackIR software installed and running in the background.

* The required Python libraries installed. You can install them via your terminal:

```
pip install -r requirements.txt
```
## Phase 1: Recording raw data
This step opens the interface to record your physical movements. The script captures absolute Unix timestamps with microsecond precision.

1. Start the hardware: Ensure your TrackIR camera is plugged in, the track-clip is visible to the camera, and the official TrackIR application is actively running on your computer.

2. Run the GUI application: Open your terminal, navigate to this project folder, and execute the recording script.
```
python TrackIR_ClientPython.py
```
3. Interact with the GUI
   * Once the GUI window appears on your screen, click Register to connect to the TrackIR background service.
   * Position your head and ensure the system is tracking your movement (you should see the coordinates updating on the screen). Once centered, click the ___ReCenter___ button
   * To record: Simply keep the window open and move your head as required for the experiment.
   * To stop and save: Click ___stop___ and ___Unregister___ in the GUI window, as well as the 'X' on the top right corner to close the application and save the data automatically. The script will automatically generate a CSV file in the recordings folder, dynamically named with the exact date and time (e.g., 20260328_153000_trackir_data.csv).


## Phase 2: Synchronizing to the desired frequency
Because TrackIR captures data at variable, high-speed intervals (~120Hz), we must mathematically interpolate this data to match the frequency required by the Sensemat system.

**IMPORTANT:** Hard-code the filenames first!
This script requires to manually specify which file you want to process.
1. Open the ```recordings``` folder and copy the exact name of the raw CSV you just generated.
2. Open ```sync_data.py``` in your text editor.
3. Locate the input variable, at the top of the main function, and hard-code the specific filename.
```
input_filename = os.path.join("recordings", "YYYYMMDD_HHMMSS_trackir_data.csv")
```
4. Save the ```sync_data.py``` file.
5. Once the filename is updated, return to the terminal and execute:
```
python sync_data.py
```
## The output
The script will strip away the hardware telemetry and output a clean, 7-column CSV file into your ```recordings``` folder:
```Unix_Timestamp | X | Y | Z | Pitch | Yaw | Roll```