#!/usr/bin/env python
"""Launch both the SenseMat server and the TrackIR client together."""

import argparse
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_SENSEMAT_PORT = "COM4"
DEFAULT_SENSEMAT_PORT_NAME = "head"
DEFAULT_SENSEMAT_CONFIG = ROOT / "configuration-single.json"
DEFAULT_SENSEMAT_SCRIPT = ROOT / "src" / "server.py"
DEFAULT_TRACKIR_SCRIPT = ROOT / "TrackIR_PythonClient" / "src" / "TrackIR_ClientPython.py"
DEFAULT_BROWSER_URL = "http://127.0.0.1:42000"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Start the SenseMat server and TrackIR client together.")
    parser.add_argument(
        "--sensemat-port",
        default=DEFAULT_SENSEMAT_PORT,
        help="Serial port for SenseMat device.")
    parser.add_argument(
        "--sensemat-port-name",
        default=DEFAULT_SENSEMAT_PORT_NAME,
        help="Port name/id for SenseMat.")
    parser.add_argument(
        "--sensemat-config",
        default=str(DEFAULT_SENSEMAT_CONFIG),
        help="JSON configuration file for SenseMat.")
    parser.add_argument(
        "--trackir-script",
        default=str(DEFAULT_TRACKIR_SCRIPT),
        help="Path to the TrackIR Python client script.")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the SenseMat web interface.")
    return parser.parse_args()


def main():
    args = parse_args()

    trackir_script = Path(args.trackir_script).resolve()
    sensemat_config = Path(args.sensemat_config).resolve()

    if not DEFAULT_SENSEMAT_SCRIPT.exists():
        raise FileNotFoundError(f"Cannot find SenseMat server script at {DEFAULT_SENSEMAT_SCRIPT}")
    if not trackir_script.exists():
        raise FileNotFoundError(f"Cannot find TrackIR client script at {trackir_script}")
    if not sensemat_config.exists():
        raise FileNotFoundError(f"Cannot find SenseMat config file at {sensemat_config}")

    sensemat_cmd = [
        sys.executable,
        str(DEFAULT_SENSEMAT_SCRIPT),
        "-p",
        args.sensemat_port,
        "-pn",
        args.sensemat_port_name,
        "-co",
        str(sensemat_config),
    ]
    trackir_cmd = [
        sys.executable,
        str(trackir_script),
    ]

    print("Starting SenseMat server...")
    sensemat_proc = subprocess.Popen(
        sensemat_cmd,
        cwd=str(ROOT),
    )

    time.sleep(1)
    print("Starting TrackIR client...")
    trackir_proc = subprocess.Popen(
        trackir_cmd,
        cwd=str(ROOT),
    )

    if not args.no_browser:
        print(f"Opening web interface at {DEFAULT_BROWSER_URL}")
        webbrowser.open(DEFAULT_BROWSER_URL)

    print("\nBoth applications are running.")
    print("- Use the browser to start SenseMat recording.")
    print("- Use the TrackIR GUI to register/start tracking.")
    print("Press Ctrl+C to stop both processes.")

    try:
        while True:
            time.sleep(1)
            if sensemat_proc.poll() is not None:
                print("SenseMat server exited.")
                break
            if trackir_proc.poll() is not None:
                print("TrackIR client exited.")
                break
    except KeyboardInterrupt:
        print("Stopping both processes...")
    finally:
        for proc, name in ((sensemat_proc, "SenseMat"), (trackir_proc, "TrackIR")):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    print(f"{name} process killed.")

        print("Done.")


if __name__ == "__main__":
    main()
