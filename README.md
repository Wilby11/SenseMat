# Sensemat Data Gateway

## Install

For this software to work, python needs to be installed. Depending on your OS, use your preffered package manager or way to install things (apt, brew, win-get, download from python.org)

## Setup Linux / MacOS

In the project folder run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Setup Windows

In the project folder run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Run server for use with one board and no nexus

```
python src/server.py --port1 [SERIAL_PORT] --port1Name [NAME] --config [CONFIG_FILE]
```

The value for SERIAL_PORT depends on your OS. Mac uses `/dev/tty.some-description`. Linux uses `/dev/ttyACMn` where n is the nth connected device. Windows uses `COMn` where n is also a number (I think COM1 and COM2 are "reserved" so start with COM3)

### Example Linux

```
python src/server.py --port1 /dev/ttyACM0 --port1Name head --config configuration-single.json
```

### Example Windows

```
python src/server.py --port1 COM3 --port1Name head --config configuration-single.json
```
python src/server.py -p COM3 -pn head -co configuration-single.json    # nieuw command want het bovenstaande is outdated


## Opening the visualiser

When the server is running, open a webbrowser and navigate to: `http://127.0.0.1:42000`

### TODO add doc about 2 sensemat boards
todo

## Recording

Go to `http://127.0.0.1:42000` and click "Start recording" to start recording and forwarding sensor data over a websocket. To stop recording click "End recording".

Recordings are saved to the recordings folder as `recordings/[Ymd-HMS]-sensemat-serial-log.csv`

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
      "tx": 7,
      "d": 64000,
      "ledpower": "2000, 2000, 2000, 2000, 2000, 2000, 2000",
      "gain": 1000,
      "integration": 1500,
      "guard": 10,
      "samplerate": 75,
      "ttl": 1
  }
]
```