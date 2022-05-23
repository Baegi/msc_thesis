import statistics
from tkinter import E
import psycopg2
import psycopg2.extras
import os
from tqdm.notebook import tqdm
import re
import pyModeS as pms
from collections import defaultdict
import pyproj
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

class GeoPoint:

    transformer = pyproj.Transformer.from_crs(
        pyproj.crs.CRS(proj='latlong', ellps='WGS84', datum='WGS84').to_3d(),
        pyproj.crs.CRS(proj='geocent', ellps='WGS84', datum='WGS84').to_3d()
    )

    def __init__(self, format, a, b, c) -> None:
        assert format in ["wgs84", "ecef"]
        if format == "ecef":
            self.x = float(a)
            self.y = float(b)
            self.z = float(c)
        else:
            assert -90 <= a <= 90
            assert -180 <= b <= 180
            self.x, self.y, self.z = GeoPoint.transformer.transform(a, b, c)


    def dist(self, other):
        return ((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2) ** 0.5

    def pos(self):
        return self.x, self.y, self.z

    def __str__(self) -> str:
        return('ecef: ' + str(self.pos()))


def connect_db():
    global conn, cur
    if "conn" not in globals() or "cur" not in globals() or conn.closed:
        #conn = sqlite3.connect("data.sqlite")
        conn = psycopg2.connect("dbname=thesis user=postgres password=postgres")
        cur = conn.cursor()
        
connect_db()

def close_db():
    global conn, cur
    cur.close()
    del cur
    conn.close()
    del conn


def clear_db():
    cur.execute('DROP SCHEMA public CASCADE;')
    cur.execute('CREATE SCHEMA public;')
    conn.commit()


def init_db():
    conn.commit()
    cur.execute('''CREATE TABLE IF NOT EXISTS messages (
        id serial,
        msg char(28) NOT NULL UNIQUE,
        icao char(6) NOT NULL,
        ecef_x double precision NOT NULL,
        ecef_y double precision NOT NULL,
        ecef_z double precision NOT NULL,
        relevant boolean NOT NULL,
        PRIMARY KEY (id)
    );''')
    cur.execute('''CREATE TABLE IF NOT EXISTS sensors (
        id serial,
        type varchar(9) NOT NULL,
        ecef_x double precision NOT NULL,
        ecef_y double precision NOT NULL,
        ecef_z double precision NOT NULL,
        PRIMARY KEY (id),
        UNIQUE (type, ecef_x, ecef_y, ecef_z)
    );''')
    cur.execute('''CREATE TABLE IF NOT EXISTS records (
        msg_id integer,
        sensor_id integer,
        sensor_timestamp double precision NOT NULL,
        server_timestamp double precision NOT NULL,
        PRIMARY KEY (msg_id, sensor_id),
        FOREIGN KEY (msg_id) REFERENCES messages(id) ON DELETE CASCADE,
        FOREIGN KEY (sensor_id) REFERENCES sensors(id) ON DELETE CASCADE
    );''')
    cur.execute('''CREATE TABLE IF NOT EXISTS time_deltas (
        sensor_a integer,
        sensor_b integer,
        mean double precision NOT NULL,
        variance double precision NOT NULL,
        num integer,
        PRIMARY KEY (sensor_a, sensor_b),
        FOREIGN KEY (sensor_a) REFERENCES sensors(id) ON DELETE CASCADE,
        FOREIGN KEY (sensor_b) REFERENCES sensors(id) ON DELETE CASCADE,
        CONSTRAINT sensor_id_check CHECK (sensor_a < sensor_b)
    );''')

    cur.execute('''CREATE TABLE IF NOT EXISTS msg_positions_raw (
        msg_id integer PRIMARY KEY,
        ecef_x double precision NOT NULL,
        ecef_y double precision NOT NULL,
        ecef_z double precision NOT NULL,
        FOREIGN KEY (msg_id) REFERENCES messages(id) ON DELETE CASCADE
    );''')

    cur.execute('''CREATE TABLE IF NOT EXISTS msg_positions_corrected (
        msg_id integer PRIMARY KEY,
        ecef_x double precision NOT NULL,
        ecef_y double precision NOT NULL,
        ecef_z double precision NOT NULL,
        FOREIGN KEY (msg_id) REFERENCES messages(id) ON DELETE CASCADE
    );''')

    conn.commit()



def is_relevant(msg):
    df = pms.df(msg)
    if df == 17 or df == 18:
        tc = pms.typecode(msg)
        if 9 <= tc <= 22 and tc != 19:
            return True
    
    return False


def convert_timestamp(sensor_type, time_at_sensor, timestamp):
    if sensor_type == 'dump1090':
        assert 0 <= timestamp / 12e6 < 89.47848533333335
        return time_at_sensor * 89.47848533333334 + timestamp / 12e6 # 2**30 / 12e6
    elif sensor_type == 'Radarcape':
        assert 0 <= timestamp < 1e9
        return time_at_sensor + timestamp / 1e9
    else:
        raise ValueError("Unknown sensor type: " + sensor_type)


last_loc_msg = defaultdict(lambda: [None, None])
def get_announced_pos(msg, timestamp):
    icao = pms.icao(msg).upper()
    oe_bit = pms.decoder.adsb.oe_flag(msg)
    last_loc_msg[icao][oe_bit] = (timestamp, msg)

    if last_loc_msg[icao][1 - oe_bit]:
        # we should be able to localize this!
        pos = pms.decoder.adsb.position(last_loc_msg[icao][0][1], last_loc_msg[icao][1][1], last_loc_msg[icao][0][0], last_loc_msg[icao][1][0])
        if not pos:
            return None

        alt = pms.adsb.altitude(msg)
        if not alt:
            return None
        if alt:
            alt = alt * 0.3048 # convert feet to meters
        
        #print(msg, icao, pms.df(msg), pms.typecode(msg), pos, alt)

        if not -90 <= pos[0] <= 90 or not -180 <= pos[1] <= 180:
            #print("Invalid lat/lon:", pos[0], pos[1])
            return None
        geo_pos = GeoPoint("wgs84", *pos, alt)
        return geo_pos

    return None


def read_data(dir, format='raw'):
    assert format in ['raw', 'LocaRDS', 'AIcrowd']

    print("Deleting old data")
    cur.execute('DELETE FROM sensors')
    cur.execute('DELETE FROM messages')
    cur.execute('DELETE FROM records')

    if format == 'raw':
        messages = set()
        discarded = defaultdict(int)
        total = 0
        bad_records = set()
        for file in tqdm(os.listdir(dir)):
            assert re.match(r"^part-\d{5}$", file)
            with open(os.path.join(dir, file), "r") as f:
                line = f.readline()
                if not line:
                    continue
                assert line == "sensorType,sensorLatitude,sensorLongitude,sensorAltitude,timeAtServer,timeAtSensor,timestamp,rawMessage,sensorSerialNumber,RSSIPacket,RSSIPreamble,SNR,confidence\n"
                
                #for line in tqdm(f.readlines()):
                while (line := f.readline().strip()):
                    #line = line.strip()

                    sensorType,sensorLatitude,sensorLongitude,sensorAltitude,timeAtServer,timeAtSensor,timestamp,rawMessage,sensorSerialNumber,RSSIPacket,RSSIPreamble,SNR,confidence = line.split(',')
                    total += 1

                    if len(rawMessage) != 28:
                        discarded["bad_msg_len"] += 1
                        continue

                    if not is_relevant(rawMessage):
                        discarded["irrelevant"] += 1
                        continue

                    if sensorType != "Radarcape":
                        discarded['not_radarcape'] += 1
                        continue

                    #if sensorType in ['SBS-3', 'OpenSky', 'dump1090']:
                    #    discarded['bad_sensortype'] += 1
                    #    continue
                    #elif sensorType not in ['dump1090']:
                    #    discarded[f'unknown_sensortype_{sensorType}'] += 1
                    #    continue

                    if 'null' in [sensorLatitude, sensorLongitude, sensorAltitude, timeAtSensor, timestamp]:
                        discarded["null_values"] += 1
                        continue

                    sensorLatitude = float(sensorLatitude)
                    sensorLongitude = float(sensorLongitude)
                    sensorAltitude = float(sensorAltitude)
                    timeAtSensor = float(timeAtSensor)
                    timestamp = float(timestamp)
                    timeAtServer = float(timeAtServer)

                    # convert timestamps according to raw_data.pdf
                    converted_timestamp = convert_timestamp(sensorType, timeAtSensor, timestamp)

                    # limit to europe for now
                    if not 35 < sensorLatitude < 75:
                        discarded["invalid_sensor_pos_lat"] += 1
                        continue
                    elif not -10 < sensorLongitude < 40:
                        discarded["invalid_sensor_pos_lon"] += 1
                        continue

                    if not (sensorLatitude and sensorLongitude):
                        # lat or lon is 0
                        discarded["invalid_sensor_pos_zero"] += 1
                        continue

                    if not rawMessage in messages:
                        messages.add(rawMessage)
                        relevant = is_relevant(rawMessage)
                        msg_pos = get_announced_pos(rawMessage, timeAtSensor) if relevant else None
                        x, y, z = msg_pos.pos() if msg_pos else (None, None, None)
                        icao = pms.icao(rawMessage)
                        if icao is None:
                            discarded["no_icao"] += 1
                            continue
                        icao = icao.upper()

                        if x is None:
                            discarded["no_msg_loc_info"] += 1
                            continue

                        # store msg in DB
                        cur.execute("""
                            INSERT INTO messages (msg, icao, ecef_x, ecef_y, ecef_z, relevant)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT DO NOTHING
                        """, (rawMessage, icao, x, y, z, relevant))

                    if not -90 <= sensorLatitude <= 90 or not -180 <= sensorLongitude <= 180:
                        print("Invalid lat/lon:", sensorLatitude, sensorLongitude)
                        continue
                    sensor_pos = GeoPoint("wgs84", sensorLatitude, sensorLongitude, sensorAltitude).pos()
                    # store sensor in DB
                    cur.execute("""
                        INSERT INTO sensors (type, ecef_x, ecef_y, ecef_z)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                    """, (sensorType, *sensor_pos))

                    # create received record in DB
                    cur.execute("SELECT id FROM messages WHERE msg = %s", (rawMessage,))
                    msg_id = cur.fetchone()[0]
                    cur.execute("SELECT id FROM sensors WHERE type = %s AND ecef_x = %s AND ecef_y = %s AND ecef_z = %s", (sensorType, *sensor_pos))
                    sensor_id = cur.fetchone()[0]
                    if (msg_id, sensor_id) in bad_records:
                        continue

                    cur.execute("""
                        INSERT INTO records (msg_id, sensor_id, sensor_timestamp, server_timestamp)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                    """, (msg_id, sensor_id, converted_timestamp, timeAtServer))

                    #print(cur.rowcount, msg_id, sensor_id)
                    #if cur.rowcount == 0:
                    #    # There's already a record with this msg_id and sensor_id
                    #    bad_records.add((msg_id, sensor_id))
                    #    cur.execute("""
                    #        DELETE FROM records WHERE msg_id = %s AND sensor_id = %s
                    #    """, (msg_id, sensor_id))
                    
            conn.commit()

    else:
        naming_format = {
            'LocaRDS': {
                'subdir': 'subset_{}',
                'records_file': 'set_{}.csv',
                'sensors_file': 'set_{}_sensors.csv'
            },
            'AIcrowd': {
                'subdir': 'training_{}_category_1',
                'records_file': 'training_{}_category_1.csv',
                'sensors_file': 'sensors.csv'
            }
        }[format]
        # Read LocaRDS data
        sensors = dict()
        sensor_mapping = dict()
        for folder in tqdm(os.listdir(dir)):
            print(folder)

            messages = list()
            records = list()
            it = int(folder[-1])
            if it == 1:
                # read sensors in first iteration
                print("Reading sensors")
                with open(os.path.join(dir, folder, naming_format['sensors_file'].format(1)), 'r') as f:
                    line = f.readline()
                    assert line.startswith('serial,latitude,longitude,height,type')
                    while (line := f.readline()):
                        serial, latitude, longitude, height, type = line.strip().split(',', 4)
                        #if good == 'FALSE':
                        #    continue
                        if type != 'Radarcape':
                            continue
                        if len(type) > 9:
                            # dump1090-hptoa
                            print(type)
                            type = type[:9]
                        ecef_x, ecef_y, ecef_z = GeoPoint('wgs84', 
                            float(latitude),
                            float(longitude),
                            float(height)
                        ).pos()
                        key = type, ecef_x, ecef_y, ecef_z
                        if key not in sensors:
                            sensors[key] = int(serial)
                        sensor_mapping[int(serial)] = sensors[key]
                    print("Writing sensors to DB")
                    psycopg2.extras.execute_batch(cur, '''INSERT INTO sensors (id, type, ecef_x, ecef_y, ecef_z)
                        VALUES (%s, %s, %s, %s, %s)
                    ''', [(e[1], *e[0]) for e in sensors.items()])
                    conn.commit()

            print("Reading Set", it)
            # read set_i
            with open(os.path.join(dir, folder, naming_format['records_file'].format(it)), 'r') as f:
                line = f.readline()
                assert line == (
                    'id,timeAtServer,aircraft,latitude,longitude,baroAltitude,geoAltitude,numMeasurements,measurements\n'
                    if format == 'LocaRDS' else
                    '"id","timeAtServer","aircraft","latitude","longitude","baroAltitude","geoAltitude","numMeasurements","measurements"\n'
                )
                while (line := f.readline()):
                    (id, timeAtServer, aircraft, latitude, longitude, baroAltitude, geoAltitude,
                        numMeasurements, measurements) = line.strip().split(',', 8)
                    if not len(latitude):
                        continue
                    if not (-90 <= float(latitude) <= 90 and
                        -180 <= float(longitude) <= 180):
                        print("lat", latitude, "lon", longitude, "alt", geoAltitude)
                        print(line)
                        continue
                    #print("geoAlt:", geoAltitude, "baroAlt", baroAltitude)
                    messages.append((
                        int(id),
                        id, # use id as raw_message because LocaRDS doesn't provide the raw message and because of the unique constraint of the column
                        aircraft,
                        *GeoPoint('wgs84', float(latitude), float(longitude), float(geoAltitude)).pos(),
                        True
                    ))

                    for entry in eval(measurements[1:-1]):
                        sensor_id, timeAtSensor, rssi = entry
                        if sensor_id not in sensor_mapping:
                            continue
                        records.append((
                            int(id),
                            sensor_mapping[int(sensor_id)],
                            float(timeAtSensor) / 1e9,
                            float(timeAtServer)
                        ))

                    if len(records) > 1e7:
                        break

            print("Writing", len(messages), "messages to DB")
            psycopg2.extras.execute_batch(cur, '''INSERT INTO messages (id, msg, icao, ecef_x, ecef_y, ecef_z, relevant)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            ''', messages)

            print("removing duplicate records")
            bad_records = set()
            prev_key = None
            for record in sorted(records):
                key = record[:2]
                if key == prev_key:
                    bad_records.add(key)
                prev_key = key

            print(len(bad_records), "bad record keys")
            print(list(bad_records)[:20])

            print("Writing", len([e for e in records if e[:2] not in bad_records]), "records to DB")
            psycopg2.extras.execute_batch(cur, '''INSERT INTO records (msg_id, sensor_id, sensor_timestamp, server_timestamp)
                VALUES (%s, %s, %s, %s)
            ''', [e for e in records if e[:2] not in bad_records])

            conn.commit()
            
            # debug
            break



def visualize_timedrift():
    cur.execute('''
        SELECT sensor_id, ARRAY_AGG(sensor_timestamp), ARRAY_AGG(server_timestamp)
        FROM records
        -- WHERE sensor_id = 481
        GROUP BY sensor_id
        ORDER BY RANDOM()
        LIMIT 5
    ''')
    rows = cur.fetchall()
    
    for r in rows:
        sensor_id, sensor_timestamps, server_timestamps = r
        print(len(sensor_timestamps), len(server_timestamps))
        sensor_timestamps = [e - sensor_timestamps[0] for e in sensor_timestamps]
        server_timestamps = [e - server_timestamps[0] for e in server_timestamps]
        difference = [sensor_timestamps[i] - server_timestamps[i] for i in range(len(sensor_timestamps))]
        #fig = plt.scatter(server_timestamps, sensor_timestamps, )
        fig = plt.figure()
        plt.scatter(range(len(difference)), difference)
        plt.title(str(sensor_id))
        
    
def visualize_flightpaths(icao=None):
    if icao is None:
        cur.execute('''SELECT icao, ARRAY_AGG(ecef_x), ARRAY_AGG(ecef_y), ARRAY_AGG(ecef_z)
                    FROM messages
                    GROUP BY icao
                    ORDER BY RANDOM()
                    LIMIT 1
                    ''')
    else:
        cur.execute('''SELECT icao, ARRAY_AGG(ecef_x), ARRAY_AGG(ecef_y), ARRAY_AGG(ecef_z)
                    FROM messages
                    WHERE icao = %s
                    GROUP BY icao
                    ''', (icao,))

    icao, xs, ys, zs = cur.fetchone()

    earth_radius = 6371000
    surface = list()
    for x, y, z in zip(xs, ys, zs):
        norm = (x**2 + y**2 + z**2)**0.5
        surface.append((x * earth_radius / norm, y * earth_radius / norm, z * earth_radius / norm))


    p1 = GeoPoint('ecef', xs[0], ys[0], zs[0])
    pn = GeoPoint('ecef', xs[-1], ys[-1], zs[-1])
    print("Total Dist:", round(p1.dist(pn) / 1e3, 2), "km")

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(xs, ys, zs, 'red')
    ax.plot3D(*list(zip(*surface)), 'grey')
    plt.title(icao)


def cleanup_sensors(variance_cutoff=10):
    print("Cleaning up sensors")
    # remove sensors with high (sensor_timestamp - server_timestamp) variances
    cur.execute('''
        SELECT sensor_id, VAR_SAMP(sensor_timestamp - server_timestamp) AS var FROM records
        GROUP BY sensor_id
        ORDER BY var DESC;
    ''')

    variances = cur.fetchall()
    conn.commit()

    print("Sensors with too high serverTimstamp variances:", len([(e[0],) for e in variances if e[1] > variance_cutoff]))

    psycopg2.extras.execute_batch(cur, '''
        DELETE FROM sensors WHERE ID = %s
    ''', [(e[0],) for e in variances if e[1] > variance_cutoff])

    conn.commit()

    # remove sensors of non-radarcape types
    cur.execute('''
        DELETE FROM sensors WHERE type != 'Radarcape'
    ''')

    # remove sensors with only one record
    cur.execute('''
        SELECT sensor_id FROM records
        GROUP BY sensor_id
        HAVING COUNT(msg_id) < 2
    ''')

    bad_sensors = cur.fetchall()
    print("Sensors with < 2 records:", len(bad_sensors))

    psycopg2.extras.execute_batch(cur, '''
        DELETE FROM sensors WHERE id = ?
    ''', bad_sensors)

    conn.commit()


def cleanup_messages():
    print("Cleaning up messages")
    # delete messages that only got received by one sensor
    cur.execute('''SELECT msg_id FROM records
        GROUP BY msg_id
        HAVING COUNT(*) = 1    
    ''')
    bad_msgs = cur.fetchall()
    print("Deleting", len(bad_msgs), "messages")

    psycopg2.extras.execute_batch(cur, '''DELETE FROM messages WHERE id = %s''', bad_msgs)

    cur.execute('''DELETE FROM messages
        WHERE ecef_x = '+infinity'
        OR ecef_y = '+infinity'
        OR ecef_z = '+infinity'
    ''')


def cleanup_flightpaths(outlier_dist_cutoff=10000, variance_cutoff=0.3, more_plots=False):
    # checks for nonsensical flightpaths, e.g. pahts where the direction is too inconsistent
    #print("Deleting flight paths with less than 3 positions")
    #cur.execute('''
    #    DELETE FROM messages
    #    WHERE icao IN (
    #        SELECT icao FROM messages
    #        GROUP BY icao
    #        HAVING COUNT(*) < 3
    #    )
    #    ''')
    #print("Deleted", cur.rowcount, "messages")
#
    #print("Removing outliers")
    #cur.execute('''
    #    SELECT icao, ARRAY_AGG(id), ARRAY_AGG(ecef_x), ARRAY_AGG(ecef_y), ARRAY_AGG(ecef_z)
    #    FROM messages
    #    GROUP BY icao
    #''')
#
    #bad_ids = list()
    #dists = list()
    #for row in tqdm(cur.fetchall()):
    #    icao, ids, xs, ys, zs = row
    #    coords = [GeoPoint('ecef', *e) for e in zip(xs, ys, zs)]
    #    for i in range(len(coords)):
    #        dist = 1e9
    #        if i > 0:
    #            dist = coords[i].dist(coords[i-1])
    #        if i < len(coords) - 1:
    #            dist = min(dist, coords[i].dist(coords[i+1]))
#
    #        dists.append(dist)
    #        if dist > outlier_dist_cutoff:
    #            bad_ids.append(ids[i])
#
    ## histogram on log scale. 
    ## Use non-equal bin sizes, such that they look equal on log scale.
    #logbins = np.geomspace(max(0.1,min(dists)), max(dists), 8)
    #fig = plt.figure()
    #plt.hist(dists, bins=logbins)
    #plt.xscale('log')
    #plt.title("Sequential Distances Histogram")
    #plt.show()
    #
    #if len(bad_ids):
    #    cur.execute('DELETE FROM messages WHERE id IN %s', (tuple(bad_ids),))
    #    print("Deleted", cur.rowcount, "messages")

    print("Aggregating directional variances in flightpaths")
    cur.execute('''
        SELECT icao, ARRAY_AGG(ecef_x), ARRAY_AGG(ecef_y), ARRAY_AGG(ecef_z)
        FROM messages
        GROUP BY icao
    ''')

    def bubblesort(list):
        # Swap the elements to arrange in order
        for iter_num in range(len(list)-1,0,-1):
            for idx in range(iter_num):
                if list[idx]>list[idx+1]:
                    temp = list[idx]
                    list[idx] = list[idx+1]
                    list[idx+1] = temp

    direction_variances = list()
    bad_icaos = list()
    for row in tqdm(cur.fetchall()):
        icao, xs, ys, zs = row
        if len(xs) < 3:
            bad_icaos.append(icao)
            continue
        directions = list()
        coords = [np.array(e) for e in zip(xs, ys, zs)]
        for i in range(1, len(coords)):
            direction = (coords[i] - coords[i-1])
            direction /= np.sqrt((direction**2).sum())
            if any([np.isnan(e) for e in direction]):
                continue
            directions.append(direction)

        
        direction_variance = sum(np.var(directions, axis=0))
        if np.isnan(direction_variance):
            print("NAN!", directions)
            continue
        direction_variances.append((direction_variance, icao))

        if direction_variance > variance_cutoff:
            bad_icaos.append(icao)

    fig = plt.figure()
    plt.plot(range(len(direction_variances)), [e[0] for e in direction_variances])
    direction_variances.sort()
    fig = plt.figure()
    plt.plot(range(len(direction_variances)), [e[0] for e in direction_variances])
    
    fig = plt.figure()
    plt.hist([e[0] for e in direction_variances])
    plt.title("Direction variances")
    
    if more_plots:
        n_samples = 11
        for i in range(n_samples):
            print(i / (n_samples-1), "quantile")
            index = int(i/(n_samples-1)*(len(direction_variances)-1))
            print(index, direction_variances[index])
            visualize_flightpaths(icao=direction_variances[index][1])

    print("Removing", len(bad_icaos), "Flight Paths")
    cur.execute('''
        DELETE FROM messages
        WHERE icao IN %s
    ''', (tuple(bad_icaos),))
    print(cur.rowcount, "messages deleted")

    conn.commit()


def get_table_length(table):
    
    cur.execute('''
        SELECT COUNT(1) FROM %s
    ''', (table,))

    ret = cur.fetchone()[0][0]
    cur.commit()
    return ret