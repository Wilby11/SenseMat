#!/usr/bin/env python

import csv
import os
from random import randrange
import neurokit2 as nk

SENSORS = 128
MIN_VALUE=0
MAX_VALUE=65535
SAMPLE_RATE=50
LENGTH=10
ROWS=10*SAMPLE_RATE
FILENAME = f"{os.path.dirname(__file__)}/dummy.csv"

fields=[]
rows = []
timestamp = 0

rsp7 = nk.rsp_simulate(duration=LENGTH, respiratory_rate=15, method="breathmetrics", sampling_rate=SAMPLE_RATE)
ecg50 = nk.ecg_simulate(duration=LENGTH, noise=0.05, method="simple", heart_rate=50, sampling_rate=SAMPLE_RATE)

fields.append("TIME")
for sensor in range(0, SENSORS):
    fields.append(f"Sensor_{sensor}")
fields.append("X")
fields.append("Y")
fields.append("Z")
fields.append("PPG")
fields.append("RSP")

for row in range(0, ROWS):
    sensors=[]
    sensors.append(timestamp)
    for sensor in range(0, SENSORS):
        sensors.append(randrange(MIN_VALUE, MAX_VALUE))
    sensors.append(randrange(0, 255)) # X
    sensors.append(randrange(0, 255)) # Y
    sensors.append(randrange(0, 255)) # Z
    sensors.append(ecg50[row]) # PPG
    sensors.append(rsp7[row]) # RSP
    rows.append(sensors)
    timestamp += 20

# writing to csv file
with open(FILENAME, 'w', encoding="utf-8") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(rows)