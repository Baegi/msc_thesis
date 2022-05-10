import psycopg2
import os
from tqdm.notebook import tqdm
import re
import pyModeS as pms
from collections import defaultdict
import pyproj
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt

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


def read_data(dir):
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
        

def visualize_timedrift():
    cur.execute('''
        SELECT sensor_id, ARRAY_AGG(sensor_timestamp), ARRAY_AGG(server_timestamp)
        FROM records
        WHERE sensor_id = 481
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
        plt.axes().scatter(range(len(difference)), difference)
        plt.title(str(sensor_id))
        
    


def cleanup_sensors(variance_cutoff=10):
    # remove sensors with high (sensor_timestamp - server_timestamp) variances
    cur.execute('''
        SELECT sensor_id, VAR_SAMP(sensor_timestamp - server_timestamp) AS var FROM records
        GROUP BY sensor_id
        ORDER BY var DESC;
    ''')

    variances = cur.fetchall()
    conn.commit()


    cur.executemany('''
        DELETE FROM sensors WHERE ID = %s
    ''', [(e[0],) for e in variances if e[1] > variance_cutoff])

    conn.commit()

    # remove sensors with only one record
    cur.execute('''
        SELECT sensor_id FROM records
        GROUP BY sensor_id
        HAVING COUNT(msg_id) < 2
    ''')

    bad_sensors = cur.fetchall()
    print("Sensors with < 2 records:", len(bad_sensors))

    cur.executemany('''
        DELETE FROM sensors WHERE id = ?
    ''', bad_sensors)

    conn.commit()


def cleanup_messages():
    # delete messages that only got received by one sensor
    cur.execute('''SELECT msg_id FROM records
        GROUP BY msg_id
        HAVING COUNT(*) = 1    
    ''')
    bad_msgs = cur.fetchall()
    print("Deleting", len(bad_msgs), "messages")

    cur.executemany('''DELETE FROM messages WHERE id = %s''', bad_msgs)

    cur.execute('''DELETE FROM messages
        WHERE ecef_x = '+infinity'
        OR ecef_y = '+infinity'
        OR ecef_z = '+infinity'
    ''')


def get_table_length(table):
    
    cur.execute('''
        SELECT COUNT(1) FROM %s
    ''', (table,))

    ret = cur.fetchone()[0][0]
    cur.commit()
    return ret