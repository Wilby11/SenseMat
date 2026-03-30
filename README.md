# SenseMat Instructions

## Run server for use with one board and no nexus

When the SenseMat is connected to your laptop via USB (for Wilbur's laptop, com4 corresponds to the far usb-port, closest to my screen), run the following command to load the server in your webbrowser:
```
python src_sensemat/server.py -p COM4 -pn head -co configuration-single.json
```
Perhaps if the server doesn't automatically open in the browser, open a new tab and navigate to: http://127.0.0.1:42000

### Recording

Go to `http://127.0.0.1:42000` and click "Start recording" to start recording and forwarding sensor data over a websocket. To stop recording click "End recording".

Recordings are saved to the recordings folder as `recordings/[Ymd-HMS]-sensemat-serial-log.csv`

### Server
We use SocketIO to create a server that can send data to the browser. The server is located in src_sensemat/server.py. It uses the configuration file located in configuration-single.json to determine how to read data from the SenseMat and how to send it to the browser.

The server works as follows:
1. Python program runs a server on your laptop
    - The server reads data from the device (USB, serial, etc.).
    - It also runs Socket.IO.
2. Browser opens a webpage
    - The webpage connects to the server using Socket.IO.
3. Both sides keep one connection open
    - Instead of repeatedly asking for data, the connection stays active.
4. When new device data appears
    - The Python program sends it through the connection.
5. The browser immediately receives it
    - JavaScript updates a graph or dashboard.

### Configuration

The configuration has the folowing structure:

```
[
  {
      "id": [string identifier],
      "rx": [int active opto diodes],
      "tx": [int transmitting diodes],
      "d": [int divisor],
      "ledpower": [string list of LED power values],
      "gain": [int gain of LED power],
      "integration": [int integration time ms],
      "guard": [int guard time ms],
      "samplerate": [int sample rate Hz],
      "ttl": [int uses ttl]
  },
  ...
]
```

### Example for single board usage

```
[
  {
      "id": "head",
      "rx": 8,
      "tx": 16,
      "d": 64000,
      "ledpower": "4095, 4095, 4095, 4095, 4095, 4095, 4095, 4095, 4095, 4095, 4095, 4095,4095, 4095, 4095, 4095",
      "gain": 1000,
      "integration": 256,
      "guard": 10,
      "samplerate": 40,
      "ttl": 1
  }
]
```


# TrackIR Instructions
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