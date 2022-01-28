# format: sensorType,sensorLatitude,sensorLongitude,sensorAltitude,timeAtServer,timeAtSensor,timestamp,rawMessage,sensorSerialNumber,RSSIPacket,RSSIPreamble,SNR,confidence

from collections import defaultdict
import pyModeS as pms
from pyModeS.decoder.adsb import position
import util
import visualize
from datetime import datetime
import json

planes = defaultdict(list)
sensors = dict()

load_planes = True
load_positions = True

generate_sensor_csv = True
generate_msg_csv = True


if not load_planes:
    PLANES_TO_PROCESS = 5

    for i in range(200):
        filename = f"raw data/martins_dataset1.csv/part-{'{:0>5d}'.format(i)}"
        print("Processing file:", filename)

        with open(filename, "r") as f:
            header = f.readline()
            line_num = 1
            while (line := f.readline()):
                line_num += 1
                sensor_type, sensor_lat, sensor_lon, sensor_alt, time_at_server, time_at_sensor, timestamp, msg, sensor_id = line.split(",")[:9]

                time_at_server = float(time_at_server)
                sensor_lat = float(sensor_lat)
                sensor_lon = float(sensor_lon)
                sensor_alt = float(sensor_alt)
                #print()
                #print("time_at_server:", time_at_server)
                #print("time_at_sensor:", time_at_sensor)
                #print("timestamp:     ", timestamp)
                df = pms.df(msg)
                if df == 17 or df == 18:
                    # civilian ADS-B
                    icao = pms.icao(msg).upper()

                    # limit processed planes
                    if icao not in planes and len(planes) >= PLANES_TO_PROCESS:
                        continue

                    if time_at_sensor is None:
                        continue

                    planes[icao].append({
                        "msg": msg,
                        "sensor_id": sensor_id,
                        "time_at_sensor": time_at_sensor,
                        "time_at_server": time_at_server
                    })
                    if sensor_id not in sensors:
                        sensors[sensor_id] = {
                            "lat": sensor_lat,
                            "lon": sensor_lon,
                            "alt": sensor_alt,
                            "type": sensor_type,
                            "line": line_num
                        }
                    #else:
                    #    if {
                    #        "lat": sensor_lat,
                    #        "lon": sensor_lon,
                    #        "alt": sensors[sensor_id]["alt"],
                    #        "type": sensor_type,
                    #        "line": sensors[sensor_id]["line"]
                    #    } != sensors[sensor_id]:
                    #        print("inconsistent sensors!", sensor_id)
                    #        print("existing entry:", sensors[sensor_id])
                    #        print("new:           ", {
                    #            "lat": sensor_lat,
                    #            "lon": sensor_lon,
                    #            "alt": sensor_alt,
                    #            "type": sensor_type,
                    #            "line": line_num
                    #        })
                    #        #exit()
                            



    json.dump(planes, open("planes.json", "w"))
    json.dump(sensors, open("sensors.json", "w"))

else:
    print("Loading planes from JSON")
    planes = json.load(open("planes.json", "r"))
    sensors = json.load(open("sensors.json", "r"))


if not load_positions:
    print("Processing planes")
    last_loc_msg = defaultdict(lambda: [None, None])
    positions = dict()
    MAX_TIMESTAMP = 9999999999
    for icao in planes:
        print("ICAO:", icao)
        received = defaultdict(lambda: { "timestamp": MAX_TIMESTAMP, "sensors": list(), "position": None })
        for measurement in planes[icao]:
            received[measurement["msg"]]["sensors"].append((measurement["time_at_sensor"], measurement["sensor_id"]))
            received[measurement["msg"]]["timestamp"] = min(received[measurement["msg"]]["timestamp"], measurement["time_at_server"])

            if received[measurement["msg"]]["position"]:
                continue

            assert pms.df(measurement["msg"]) in [17, 18]
            tc = pms.typecode(measurement["msg"])
            if 9 <= tc <= 22 and tc != 19:
                # airborne positions w/ baro altitude or GNSS height
                oe_bit = pms.decoder.adsb.oe_flag(measurement["msg"])
                last_loc_msg[icao][oe_bit] = (measurement["time_at_server"], measurement["msg"])

                if last_loc_msg[icao][1 - oe_bit]:
                    # we should be able to localize this!
                    pos = pms.decoder.adsb.position(last_loc_msg[icao][0][1], last_loc_msg[icao][1][1], last_loc_msg[icao][0][0], last_loc_msg[icao][1][0])
                    received[measurement["msg"]]["position"] = pos

        positions[icao] = sorted(received.values(), key=lambda e: e["timestamp"])
        

    json.dump(received, open("received.json", "w"))
    json.dump(positions, open("positions.json", "w"))

else:
    print("Loading positions from JSON")
    received = json.load(open("received.json", "r"))
    positions = json.load(open("positions.json", "r"))


if generate_sensor_csv:
    with open("sensors.csv", "w+") as f:
        f.write("serial,latitude,longitude,height,type\n")
        for id, sensor in sensors.items():
            f.write(",".join([str(e) for e in
                [
                    id,
                    sensor["lat"],
                    sensor["lon"],
                    sensor["alt"],
                    sensor["type"]
                ]]) + "\n")


if generate_msg_csv:
    with open("msgs.csv", "w+") as f:
        f.write("id,timeAtServer,aircraft,latitude,longitude,baroAltitude,geoAltitude,numMeasurements,measurements\n")
        id = 0
        for icao in positions:
            for details in positions[icao]:
                if any([s[0] == "null" for s in details["sensors"]]):
                    continue

                id += 1
                f.write(",".join([str(e) for e in
                    [
                        id,
                        details["timestamp"],
                        icao,
                        "NaN",
                        "NaN",
                        "NaN",
                        "NaN",
                        len(details["sensors"]),
                        '"' + str([[int(e[1]), float(e[0])] for e in details["sensors"]]) + '"'
                    ]]) + "\n")


mw = visualize.MainWindow()
mw.visualize_flight_paths(positions, sensors)
