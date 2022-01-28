from re import L
import sys

locards_sensors = set()

cur_sensors = set()

with open(sys.argv[1]) as f:
    line = f.readline()
    while (line := f.readline()):
        serial,latitude,longitude,height,type = line.strip().split(',', 4)
        key = float(latitude),float(longitude),float(height),type
        locards_sensors.add(key)

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

        cur_sensors.add(key)

print("loca", len(locards_sensors))
print("cur", len(cur_sensors))
print("loca - cur", len(locards_sensors - cur_sensors))
print("cur - loca", len(cur_sensors - locards_sensors))
