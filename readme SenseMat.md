# Sensemat Data Gateway

## Run server for use with one board and no nexus

When the SenseMat is connected to your laptop via USB (for Wilbur's laptop, com4 corresponds to the far usb-port, closest to my screen), run the following command to load the server in your webbrowser:
```
python src/server.py -p COM4 -pn head -co configuration-single.json
```
Perhaps if the server doesn't automatically open in the browser, open a new tab and navigate to: http://127.0.0.1:42000

## Recording

Go to `http://127.0.0.1:42000` and click "Start recording" to start recording and forwarding sensor data over a websocket. To stop recording click "End recording".

Recordings are saved to the recordings folder as `recordings/[Ymd-HMS]-sensemat-serial-log.csv`

## Server
We use SocketIO to create a server that can send data to the browser. The server is located in src/server.py. It uses the configuration file located in configuration-single.json to determine how to read data from the SenseMat and how to send it to the browser.

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

## Configuration

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