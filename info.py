import sys
from collections import defaultdict

locards_sensors = set()

cur_sensors = set()

with open(sys.argv[1]) as f:
    line = f.readline()
    while (line := f.readline()):
        serial,latitude,longitude,height,type,trusted = line.strip().split(',', 5)
        key = float(latitude),float(longitude),float(height),type
        locards_sensors.add(key)


messages = defaultdict(set)
with open(sys.argv[2]) as f:
    line = f.readline()
    while (line := f.readline()):
        sensorType,sensorLatitude,sensorLongitude,sensorAltitude,timeAtServer,timeAtSensor,timestamp,rawMessage,sensorSerialNumber,RSSIPacket,RSSIPreamble,SNR,confidence = line.split(",", 12)
        if sensorAltitude == "null":
            continue
        if sensorLongitude == "null":
            continue
        if sensorLatitude == "null":
            continue
        key = float(sensorLatitude),float(sensorLongitude),float(sensorAltitude),sensorType

        if key in locards_sensors and sensorType in []:
            messages[rawMessage].add(key)
        cur_sensors.add(key)

lens = defaultdict(int)
for m in messages:
    lens[len(messages[m])] += 1

print(lens)

print("loca", len(locards_sensors))
print("cur", len(cur_sensors))
print("loca - cur", len(locards_sensors - cur_sensors))
print("cur - loca", len(cur_sensors - locards_sensors))

print(next(iter(locards_sensors)))
print(next(iter(cur_sensors)))
