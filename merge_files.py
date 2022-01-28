import sys

if len(sys.argv) != 7:
    print("Usage:", sys.argv[0], "LocaRDS_sensors.csv", "LocaRDS_msgs.csv", "custom_sensors.csv", "custom_msgs.csv", "out_sensors.csv", "out_msgs.csv")




locards_sensors_f = open(sys.argv[1], "r")
locards_msgs_f = open(sys.argv[2], "r")

custom_sensors_f = open(sys.argv[3], "r")
custom_msgs_f = open(sys.argv[4], "r")

sensors = dict()

sensor_mappings = dict()

current_serial = 1
def process_sensors(file):
    global sensors, sensor_mappings, current_serial
    line = file.readline()
    while (line := file.readline().strip()):
        #print(line)
        serial,latitude,longitude,height,type = line.split(",")[:5]
        key = (latitude,longitude,height,type)

        if key not in sensors:
            sensors[key] = str(current_serial)
            current_serial += 1

        sensor_mappings[serial] = sensors[key]

print("Processing Sensors")
process_sensors(custom_sensors_f)
#print(sensor_mappings)
process_sensors(locards_sensors_f)
#print(sensor_mappings)

locards_sensors_f.close()
custom_sensors_f.close()

msgs = list()

def process_msgs(file):
    global msgs
    line = file.readline()
    while (line := file.readline().strip()):
        #print(line)
        id,timeAtServer,aircraft,latitude,longitude,baroAltitude,geoAltitude,numMeasurements,measurements = line.split(",", 8)
        measurements = eval(measurements[1:-1])

        for i in range(len(measurements)):
            measurements[i][0] = int(sensor_mappings[str(measurements[i][0])])
            
        measurements = '"' + "".join(str(measurements).split()) + '"'

        msgs.append((id,timeAtServer,aircraft,latitude,longitude,baroAltitude,geoAltitude,numMeasurements,measurements))

print("Processing messages")
process_msgs(locards_msgs_f)
process_msgs(custom_msgs_f)

locards_msgs_f.close()
custom_msgs_f.close()

print("Writing output")
with open(sys.argv[5], "w") as f:
    f.write("serial,latitude,longitude,height,type\n")
    for sensor in sensors:
        serial = sensors[sensor]
        f.write(",".join((serial, *sensor)) + "\n")


with open(sys.argv[6], "w") as f:
    f.write("id,timeAtServer,aircraft,latitude,longitude,baroAltitude,geoAltitude,numMeasurements,measurements\n")
    for msg in msgs:
        f.write(",".join(msg) + "\n")
