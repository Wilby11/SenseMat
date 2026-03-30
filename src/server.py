#!/usr/bin/env python
"""Data recording and visualisation service for the SenseMat study"""
import json
import socket
from datetime import datetime
import multiprocessing
import time
import os
import logging
import argparse
from threading import Thread

import uvicorn
import socketio
import click
import serial


from csv_log import (
    generate_header,
    generate_configuration_comment
)

#CONFIGURATION_FILENAME = "configuration.json"

basepath = os.path.dirname(__file__)

parser = argparse.ArgumentParser(
    prog="Sensemat Data Stream Gateway",
    description="""This server streams sensemat data over a websocket,
    either from the dummy.csv file or from the serial port""",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

def check_file(path):
    """Check file argument"""
    if not os.path.exists(path):
        raise ValueError
    return path

parser.add_argument(
    "-f",
    "--freq",
    type=int,
    default=50,
    help="Frequency of CSV data stream ",
)

def check_port(path):
    """Check file argument"""
    if not os.path.exists(path):
        raise ValueError
    return path

parser.add_argument(
    "-p",
    "--port",
    type=check_port,
    help="Set the serial port for input 1",
)

parser.add_argument(
    "-pn",
    "--portName",
    type=str,
    default="head",
    help="Set the name for input 1",
)

parser.add_argument(
    "-co",
    "--config",
    type=check_file,
    default=f"{basepath}/configuration-single.json",
    help="What csv file to load",
)

args = parser.parse_args()

SAMPLE_RATE = 1 / args.freq
SERIAL_PORT = args.port
SERIAL_IDENTIFIER = args.portName
CONFIGURATION_FILENAME = args.config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter(click.style("INFO", fg="green") + ":     %(message)s")
)
logger.addHandler(console_handler)

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
app = socketio.ASGIApp(
    sio,
    static_files={
        "/": f"{basepath}/client/index.html",
        "/static": f"{basepath}/client",
        "/recordings": "./recordings",
    },
)
ws_task:Thread = None
serial_data = {}

async def save_configuration(config_data):
    """Store configuration for later re-use."""
    with open(CONFIGURATION_FILENAME, mode="w", encoding="utf8") as outfile:
        json.dump(config_data, outfile, indent=4)


async def load_configuration():
    """Load stored configuration when the file exists."""
    if not os.path.exists(CONFIGURATION_FILENAME):
        logger.info("Using default config")
        return [
            {"id": SERIAL_IDENTIFIER, "rx": 8, "tx": 16}
        ]

    return await read_json_file(CONFIGURATION_FILENAME)


async def read_json_file(filename):
    """Load config file."""
    with open(filename, mode="r", encoding="utf8") as infile:
        data = json.load(infile)
        return data


async def fetch_samples_from_serial_task(serial_port, serial_identifier, configuration):
    """Read serial port and forward to websocket."""
    logger.info("Starting serial data stream task using %s", serial_port)
    ser = serial.Serial(serial_port, baudrate=115200)
    recording_data = False
    log_file = None

    led_power = 4095  # Default LED power
    gain = 1000  # Default LED power gain
    integration = 1000  # Default integration time (microseconds)
    guard = 10  # Default guard time (microseconds)

    current_config = next(
        (x for x in configuration if x["id"] == serial_identifier), None
    )

    if current_config is not None:
        led_power = current_config["ledpower"]
        gain = current_config["gain"]
        integration = current_config["integration"]
        guard = current_config["guard"]
        sample_rate = current_config["samplerate"]
        tx = current_config["tx"]
        rx = current_config["rx"]

    logger.info(
        "Set boad %s Matrix to TX:%s RX:%s LED Power to %s with gain %s / 1000",
        serial_identifier,
        tx,
        rx,
        led_power,
        gain,
    )
    command_string = f"S{sample_rate},{integration},{guard},{gain},{tx},{rx},{led_power}\n"
    logger.info("Send Command: %s", command_string)
    ser.write(command_string.encode())

    while True:
        if not backgroundTaskStarted.value:
            if recording_data:
                requestRecordingData.value = False
                recording_data = False
                log_file.close()
            ser.flush()
            break

        try:
            line = ser.readline()
            txtline = line.decode()
            row = txtline.split(",")
            if recording_data:
                recv_time = datetime.now().timestamp()
                txtline = f"{recv_time},{txtline}"
                log_file.write(txtline)
            # sio.emit("on_sample_data", {"id":serial_identifier, "data": row})
            if serial_port == SERIAL_PORT:
                serial_data.value = {"id": serial_identifier, "data": row}
            await sio.sleep(0.00001)
        except Exception as msg:
            logger.exception(msg)
            break

        if requestRecordingData.value and not recording_data:
            recording_data = True
            start_time = datetime.now().strftime("%Y%m%dT%H%M%S")
            filename = f"{start_time}-{serial_identifier}-sensemat-serial-log.csv"
            # configuration = await load_configuration()
            log_file = create_logfile(
                filename, len(row), configuration, serial_identifier
            )
            await sio.emit("on_recording_started", {"data": filename})
        elif not requestRecordingData.value and recording_data:
            recording_data = False
            await sio.emit("on_recording_ended", {"data": filename})
            log_file.close()

    logger.info("Exit data stream task for %s", serial_identifier)


async def background():
    """
    This thread emits the sampled data to the front end. It is possible and totally fine
    that this skips packages as it is for preview visualisation only.
    """
    while True:
        if not backgroundTaskStarted.value:
            break
        packet = {
            "sensemat": [serial_data.value]
        }

        # INSERT NEURAL NET HERE
        #sensors = serial_data.value["data"][3:128+3]
        #logger.info("f: %s",sensors)

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
    time.sleep(1)
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
