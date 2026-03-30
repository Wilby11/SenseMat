#!/usr/bin/env python
"""Data recording and visualisation service for the SenseMat study"""

#standard imports
import json
#import socket
from datetime import datetime
import multiprocessing
import time
import os
import logging
import argparse
#not used for now but probably will be useful later with then NN
from threading import Thread

#third-party packages
import uvicorn
import socketio
import click
import serial

#imports functions that create the header from another file csv_log.py
from csv_log import (
    generate_header,
    generate_configuration_comment
)

#CONFIGURATION_FILENAME = "configuration.json"

#gets folder with the script
basepath = os.path.dirname(__file__)

#sets up CLI argument parsing
parser = argparse.ArgumentParser(
    prog="Sensemat Data Stream Gateway",
    description="""This server streams sensemat data over a websocket,
    either from the dummy.csv file or from the serial port""",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

#ensures the config file exists
def check_file(path):
    """Check file argument"""
    if not os.path.exists(path):
        raise ValueError
    return path

#adds frequency argument (not really used, we use samplerate instead)
parser.add_argument(
    "-f",
    "--freq",
    type=int,
    default=50,
    help="Frequency of CSV data stream ",
)

#checks whether serial port path exists (mac: /dev/...)
def check_port(path):
    """Check file argument"""
    if not os.path.exists(path):
        raise ValueError
    return path

#adds port argument (serial device required)
parser.add_argument(
    "-p",
    "--port",
    type=check_port,
    help="Set the serial port for input 1",
)

#adds portname argument to match the config entry
parser.add_argument(
    "-pn",
    "--portName",
    type=str,
    default="head",
    help="Set the name for input 1",
)

#adds config file argument to load settings from JSON
parser.add_argument(
    "-co",
    "--config",
    type=check_file,
    default=f"{basepath}/configuration-single.json",
    help="What csv file to load",
)

#parses the arguments in the user input
args = parser.parse_args()

#define the input arguments
#(again freq was not really used, we use sample rate from config file)
#SAMPLE_RATE = 1 / args.freq
SERIAL_PORT = args.port
SERIAL_IDENTIFIER = args.portName
CONFIGURATION_FILENAME = args.config

#sets up green console logging in terminal
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter(click.style("INFO", fg="green") + ":     %(message)s")
)
logger.addHandler(console_handler)

#async Socket.IO server (any origin can connect)
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
app = socketio.ASGIApp(
    sio,
    static_files={
        "/": f"{basepath}/client/index.html",   #frontend page
        "/static": f"{basepath}/client",        #frontend assets
        "/recordings": "./recordings",          #saved CSV files
    },
)

#not used atm but later for threading
ws_task:Thread = None
#not used either?
serial_data = {}

#saves JSON config to disk
async def save_configuration(config_data):
    """Store configuration for later re-use."""
    with open(CONFIGURATION_FILENAME, mode="w", encoding="utf8") as outfile:
        json.dump(config_data, outfile, indent=4)

#reads config from disk
#if file is missing, returns minimal default config
async def load_configuration():
    """Load stored configuration when the file exists."""
    if not os.path.exists(CONFIGURATION_FILENAME):
        logger.info("Using default config")
        return [
            {"id": SERIAL_IDENTIFIER, 
             "rx": 8, 
             "tx": 16,
             "ledpower": 4095,
             "gain": 1000,
             "integration": 1000,
             "guard": 10,
             "samplerate": 40,
             }
        ]

    return await read_json_file(CONFIGURATION_FILENAME)

#JSON loader
async def read_json_file(filename):
    """Load config file."""
    with open(filename, mode="r", encoding="utf8") as infile:
        data = json.load(infile)
        return data

# reads rows, stores latest data, optionally records to CSV
async def fetch_samples_from_serial_task(serial_port, serial_identifier, configuration):
    """Read serial port and forward to websocket."""
    #opens serial port, starts with recording OFF
    logger.info("Starting serial data stream task using %s", serial_port)
    ser = serial.Serial(serial_port, baudrate=115200)
    recording_data = False
    log_file = None

    #just in case defaults in case something is wrong with config
    led_power = 4095  # Default LED power
    gain = 1000  # Default LED power gain
    integration = 1000  # Default integration time (microseconds)
    guard = 10  # Default guard time (microseconds)

    #finds matching config entry for the device whose id is the portname
    current_config = next(
        (x for x in configuration if x["id"] == serial_identifier), {}
    )

    #override defaults from config file
    led_power = current_config.get("ledpower", 4095)
    gain = current_config.get("gain", 1000)
    integration = current_config.get("integration", 1000)
    guard = current_config.get("guard", 10)
    sample_rate = current_config.get("samplerate", 40)
    tx = current_config.get("tx", 16)
    rx = current_config.get("rx", 8)

    #reports config settings to console
    logger.info(
        "Set board %s Matrix to TX:%s RX:%s LED Power to %s with gain %s / 1000",
        serial_identifier,
        tx,
        rx,
        led_power,
        gain,
    )

    #sends configuration command to the board
    command_string = f"S{sample_rate},{integration},{guard},{gain},{tx},{rx},{led_power}\n"
    logger.info("Send Command: %s", command_string)
    ser.write(command_string.encode())

    #infinite read rows loop
    while True:
        if not backgroundTaskStarted.value:
            if recording_data:
                requestRecordingData.value = False
                recording_data = False
                log_file.close()
            ser.flush()
            break
        
        #reads one line from serial device, decodes bytes to text, splits csv style to python list
        try:
            line = ser.readline()
            txtline = line.decode(errors="replace").strip()
            if not txtline:
                continue

            row = txtline.split(",")

            #if recording, write to CSV with a timestamp
            if recording_data:
                recv_time = datetime.now().timestamp()
                txtline = f"{recv_time},{txtline}"
                log_file.write(txtline)
            # sio.emit("on_sample_data", {"id":serial_identifier, "data": row})
            #stores the latest sample only, the websocket broadcast loop later emits this latest sample
            if serial_port == SERIAL_PORT:
                serial_data.value = {"id": serial_identifier, "data": row}
            await sio.sleep(0.00001)
        #logs any error
        except Exception as msg:
            logger.exception(msg)
            break

        #if frontend turnned recording on, creates filename, CSV an notifies frontend
        if requestRecordingData.value and not recording_data:
            recording_data = True
            start_time = datetime.now().strftime("%Y%m%dT%H%M%S")
            filename = f"{start_time}-{serial_identifier}-sensemat-serial-log.csv"
            # configuration = await load_configuration()
            log_file = create_logfile(
                filename, len(row), configuration, serial_identifier
            )
            await sio.emit("on_recording_started", {"data": filename})
        #stop recording and close file
        elif not requestRecordingData.value and recording_data:
            recording_data = False
            await sio.emit("on_recording_ended", {"data": filename})
            log_file.close()

    #close the serial port when the loop finishes
    ser.close()
    logger.info("Exit data stream task for %s", serial_identifier)


async def background():
    """
    This thread emits the sampled data to the front end. It is possible and totally fine
    that this skips packages as it is for preview visualisation only.
    """
    #wraps latest sample in packet structure
    while True:
        if not backgroundTaskStarted.value:
            break
    
        current = serial_data.value

        packet = {
            "sensemat": [{
                "id": current.get("id"),
                "raw": current.get("data", []),
                "prediction": None
            }]
        }

        # INSERT NEURAL NET HERE (run inference)
        #sensors = serial_data.value["data"][3:128+3]
        #logger.info("f: %s",sensors)

        #emits live data to frontend
        await sio.emit("on_sample_data", packet)
        await sio.sleep(0.00001)



@sio.event
async def ping_from_client(sid):
    """Respond to client ping."""
    await sio.emit("pong_from_server", room=sid)


@sio.event
async def end_recording(sid):
    """End recording sensemat data."""
    logger.info("End recording requested")
    requestRecordingData.value = False


@sio.event
async def start_recording(sid):
    """Start recording sensemat data."""
    logger.info("Recording requested")
    await reconnect(sid)
    requestRecordingData.value = True


@sio.event
async def connect(sid, env):
    """New client connection."""
    logger.info("Client %s connected", sid)

    configuration = await load_configuration()
    await emit_config_update(configuration)

    if not backgroundTaskStarted.value:
        backgroundTaskStarted.value = True

        logger.info("Stream Socket data from port 1 %s", SERIAL_PORT)
        sio.start_background_task(
            fetch_samples_from_serial_task,
            SERIAL_PORT,
            SERIAL_IDENTIFIER,
            configuration,
        )

        ws_task = sio.start_background_task(background)


@sio.event
async def reconnect(sid):
    """Reconnect to client"""
    logger.info("Client %s disconnected", sid)
    backgroundTaskStarted.value = False
    await sio.sleep(1)
    await connect(sid, None)


@sio.event
async def disconnect(sid):
    """Disconnected Client connection, drop client count"""
    logger.info("Client %s disconnected", sid)
    backgroundTaskStarted.value = False


@sio.event
async def update_config(_sid, config_data):
    """Updating config"""
    logger.info("Update config with %s", config_data)
    await save_configuration(config_data)
    await emit_config_update(config_data)
    await reconnect(_sid)


async def emit_config_update(config_data):
    """Send config update over sio socket."""
    await sio.emit("server_config_updated", data=config_data)


def create_logfile(file_name, fields, config, serial_identifier):
    """Create new log file with headers"""
    log_file = open(f"recordings/{file_name}", "w", encoding="utf-8")
    log_file.write(generate_configuration_comment(config))
    header = generate_header(config, serial_identifier)
    log_file.write(header)
    return log_file

if __name__ == "__main__":
    logger.info("Start Sensemat Data Gateway Websocket Server")
    m = multiprocessing.Manager()
    backgroundTaskStarted = m.Value(bool, False)
    requestRecordingData = m.Value(bool, False)
    serial_data = m.Value({}, {})
    uvicorn.run(app, host="127.0.0.1", port=42000)
