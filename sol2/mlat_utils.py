import util
from tqdm.notebook import tqdm
from dataclasses import dataclass
import numpy as np
import statistics
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from collections import deque
import random

M = np.diag([1, 1, 1, -1])

@dataclass
class Vertexer:

    nodes: np.ndarray

    # Defaults
    v = 299792458

    def __post_init__(self):
        # Calculate valid input range
        max = 0
        min = 1E+10
        centroid = np.average(self.nodes, axis = 0)
        for n in self.nodes:
            dist = np.linalg.norm(n - centroid)
            if dist < min:
                min = dist

            for p in self.nodes:
                dist = np.linalg.norm(n - p)

                if dist > max:
                    max = dist

        max /= self.v
        min /= self.v

        #print(min, max)

    def errFunc(self, point, times):
        # Return RSS error
        error = 0

        for n, t in zip(self.nodes, times):
            error += ((np.linalg.norm(n - point) / self.v) - t)**2

        return error

    def find(self, times):
        def lorentzInner(v, w):
            # Return Lorentzian Inner-Product
            return np.sum(v * (w @ M), axis = -1)

        A = np.append(self.nodes, times * self.v, axis = 1)
        #print(A)
        At = np.transpose(A)
        #print("At")
        #print(At)
        AtA = np.matmul(At, A)
        #print("AtA")
        #print(AtA)
        invAtA = np.linalg.inv(AtA)
        #print("invAtA")
        #print(invAtA)
        A_plus = np.matmul(invAtA, At)
        #print("A_plus")
        #print(A_plus)


        b = 0.5 * lorentzInner(A, A)
        #print("b")
        #print(b)
        #oneA = np.linalg.solve(A_plus, np.ones(4))
        #invA = np.linalg.solve(A_plus, b)

        oneA_plus = np.matmul(A_plus, np.ones(len(self.nodes)))
        invA_plus = np.matmul(A_plus, b)

        #print("oneA_plus", oneA_plus.shape, np.ones(len(self.nodes)).shape)
        #print(oneA_plus)
        #print("invA_plus")
        #print(invA_plus)


        solution = []
        for Lambda in np.roots([ lorentzInner(oneA_plus, oneA_plus),
                                (lorentzInner(oneA_plus, invA_plus) - 1) * 2,
                                 lorentzInner(invA_plus, invA_plus),
                                ]):
            #X, Y, Z, T = M @ np.linalg.solve(, Lambda * np.ones(len(self.nodes)) + b)
            X, Y, Z, T = np.matmul(A_plus, (b + Lambda * np.ones(len(self.nodes))))
            if any(np.iscomplex([X, Y, Z, T])):
                continue
            solution.append(np.array([X,Y,Z]))
            #print("Candidate:", X, Y, Z, math.sqrt(X**2 + Y**2 + Z**2))

        if not len(solution):
            return
        #print()
        #print()
        return min(solution, key = lambda err: self.errFunc(err, times))




def calc_mlat(sensor_ids, sensor_locations, sensor_timestamps, time_deltas):
    assert len(sensor_ids) >= 4

    # select sensor subset for mlat
    # for now, just check which ones have a time_delta relation to the first sensor
    relevant_sensors = list()
    for sensor_id in sensor_ids:
        if not len(relevant_sensors):
            relevant_sensors.append(sensor_id)
            td_base_sensor = sensor_id
            continue
        if td_base_sensor in time_deltas and sensor_id in time_deltas[td_base_sensor]:
            relevant_sensors.append(sensor_id)

    if len(relevant_sensors) < 4:
        return


    # prepare locations (gather around 0,0,0 for more accurate calculations)
    ecef_min_coordinates = [1e9, 1e9, 1e9]
    ecef_max_coordinates = [-1e9, -1e9, -1e9]
    for sensor_id in relevant_sensors:
        l = sensor_locations[sensor_id].pos()
        for i in range(3):
            ecef_min_coordinates[i] = min(ecef_min_coordinates[i], l[i])
            ecef_max_coordinates[i] = max(ecef_max_coordinates[i], l[i])

    center_point = np.add(ecef_min_coordinates, ecef_max_coordinates) / 2
    locations = [np.subtract(sensor_locations[e].pos(), center_point) for e in relevant_sensors]

    # prepare timestamps
    timestamps = [0] # zero represents td_base
    for sensor_id in relevant_sensors[1:]:
        timestamps.append(
            sensor_timestamps[sensor_id] + time_deltas[td_base_sensor][sensor_id][0]
            - sensor_timestamps[td_base_sensor]
        )
    


    myVertexer = Vertexer(np.array(locations))
    try:
        target_location = np.add(
            myVertexer.find(np.array([[e] for e in timestamps])),
            center_point
        )
        return util.GeoPoint('ecef', *target_location)
    except:
        #print("Fail")
        pass



def calc_positions(variance_cutoff=1e-6):

    util.cur.execute('SELECT id, ecef_x, ecef_y, ecef_z FROM sensors')
    sensor_locations = {e[0]: util.GeoPoint('ecef', *e[1:]) for e in util.cur.fetchall()}

    util.cur.execute('SELECT * FROM time_deltas WHERE VARIANCE < %s', (variance_cutoff,))
    time_deltas = dict()

    for s1, s2, mean, var, num in util.cur.fetchall():
        if s1 not in time_deltas:
            time_deltas[s1] = dict()
        time_deltas[s1][s2] = (mean, var)
        if s2 not in time_deltas:
            time_deltas[s2] = dict()
        time_deltas[s2][s1] = (-mean, var)

    
    util.cur.execute('''DELETE FROM msg_positions_raw''')

    util.cur.execute('''SELECT msg_id, ARRAY_AGG(sensor_id), ARRAY_AGG(sensor_timestamp)
        FROM records
        GROUP BY msg_id
        HAVING COUNT(*) >= 4
    ''')

    it = 0
    for msg_id, sensor_ids, sensor_timestamps in tqdm(util.cur.fetchall()):
        it += 1
        if it > 10000 and False:
            break

        assert len(sensor_timestamps) == len(sensor_ids)
        
        #print(msg_id, sensor_ids, sensor_timestamps)
        pos = calc_mlat(
            sensor_ids,
            sensor_locations,
            dict(zip(sensor_ids, sensor_timestamps)),
            #{sensor_ids[i]: sensor_timestamps[i] for i in range(len(sensor_ids))},
            time_deltas
        )
        if pos is None:
            #print(":(")
            continue
        #print(pos)
        util.cur.execute('''INSERT INTO msg_positions_raw VALUES (
            %s, %s, %s, %s
        )''', (msg_id, *pos.pos()))

    util.conn.commit()


def summarize_accuracy():
    util.conn.commit()
    util.cur.execute('''SELECT COUNT(*) FROM messages''')
    print("Number of received messages:", util.cur.fetchone())

    util.cur.execute('''SELECT COUNT(*) FROM msg_positions_raw''')
    print("Number of calculated positions:", util.cur.fetchone())

    util.cur.execute('''
        SELECT |/(
            (msg.ecef_x - msg_pos.ecef_x)^2 +
            (msg.ecef_y - msg_pos.ecef_y)^2 +
            (msg.ecef_z - msg_pos.ecef_z)^2
        ) AS dist

        FROM messages msg
        JOIN msg_positions_raw msg_pos
        ON msg.id = msg_pos.msg_id
    ''')

    dists = [e[0] for e in util.cur.fetchall()]
    util.conn.commit()

    print("Best dist:", min(dists))
    print("Worst dist:", max(dists))
    print("Mean dist:", statistics.mean(dists))
    print("Median dist:", statistics.median(dists))


def visualize_flight_paths(stage):
    assert stage in ['raw', 'corrected']

    util.conn.commit()
    util.cur.execute(f'''
        SELECT
            ARRAY_AGG((pos.msg_id, pos.ecef_x, pos.ecef_y, pos.ecef_z) ORDER BY pos.msg_id ASC),
            ARRAY_AGG((pos.msg_id, messages.ecef_x, messages.ecef_y, messages.ecef_z) ORDER BY pos.msg_id ASC)
        FROM msg_positions_{stage} pos
        JOIN messages ON messages.id = pos.msg_id
        GROUP BY messages.icao
    ''',)

    for calculated, actual in random.sample(util.cur.fetchall(), 5):
        # all messages in a path
        calculated = list(sorted([list(eval(e)) for e in eval(calculated)]))
        actual = list(sorted([list(eval(e)) for e in eval(actual)]))
        print("N messages:", len(calculated))

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        ax.plot3D(*list(zip(*calculated))[1:], 'gray')
        ax.plot3D(*list(zip(*actual))[1:], 'red')


def post_process_positions():
    # possible improvement:
    # take into account temporal data, i.e. constrain changes in velocity

    util.cur.execute('DELETE FROM msg_positions_corrected')

    util.cur.execute('''
        SELECT
            ARRAY_AGG((pos.msg_id, pos.ecef_x, pos.ecef_y, pos.ecef_z) ORDER BY pos.msg_id ASC),
            ARRAY_AGG((pos.msg_id, messages.ecef_x, messages.ecef_y, messages.ecef_z) ORDER BY pos.msg_id ASC)
        FROM msg_positions_raw pos
        JOIN messages ON messages.id = pos.msg_id
        GROUP BY messages.icao
    ''')

    WINDOW_SIZE = 20
    OUTLIER_DISTANCE_CUTOFF = 200000 # 20km
    for messages in tqdm(util.cur.fetchall()):
        # all messages in a path
        messages = list(sorted([list(eval(e)) for e in eval(messages[0])]))
        points = [util.GeoPoint('ecef', *e[1:]) for e in messages]

        bad_msgs = set()
        # first, remove outliers
        for i in range(len(points)):
            # sliding window shenanigans
            for j in range(max(0, i-WINDOW_SIZE//2), min(len(points), i + (WINDOW_SIZE+1)//2)):
                if i == j:
                    continue
                if points[i].dist(points[j]) < OUTLIER_DISTANCE_CUTOFF:
                    break
            else:
                bad_msgs.add(messages[i][0])

        util.cur.executemany('''INSERT INTO msg_positions_corrected VALUES(
            %s, %s, %s, %s
        )''', ([e for e in messages if e[0] not in bad_msgs]))
        
        print("Removed", len(bad_msgs), "/", len(messages), "positions")


        
