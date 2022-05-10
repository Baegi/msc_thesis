from functools import cache
import itertools
import util
from tqdm.notebook import tqdm
from dataclasses import dataclass
import numpy as np
import statistics
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import random
import traceback
import sympy

M = np.diag([1, 1, 1, -1])
C = 299792458

@dataclass
class Vertexer:

    nodes: np.ndarray

    # Defaults
    v = np.longdouble(299792458)

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

    def find(self, times, debug=False):
        def lorentzInner(v, w):
            # Return Lorentzian Inner-Product
            return np.sum(v * (w @ M), axis = -1)

        if debug:
            print("self.nodes")
            print(self.nodes)
            print("times")
            print(times)
            print("times * v")
            print(list(zip(np.multiply(times, self.v))))

        A = np.append(self.nodes, list(zip(np.multiply(times, self.v))), axis = 1)
        if debug:
            print("A:")
            print(A)
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


def calc_mlat(sensor_ids, sensor_locations, sensor_timestamps, time_deltas, debug=False):
    assert len(sensor_ids) >= 4

    if debug:
        print("N Sensors:", len(sensor_ids))

    # select sensor subset for mlat
    relevant_sensors = list()
    for sensor_id in sensor_ids:
        if not len(relevant_sensors):
            relevant_sensors.append(sensor_id)
            td_base_sensor = sensor_id
            continue
        if td_base_sensor in time_deltas and sensor_id in time_deltas[td_base_sensor]:
            relevant_sensors.append(sensor_id)

    if debug:
        print("Initial sensor subset:", relevant_sensors)

    if len(relevant_sensors) < 4:
        if debug:
            print("Not enough initally connected sensors")
        return

    while len(relevant_sensors) > 4:
        # find worst of the relevant sensors and potentially exclude it
        variance_sums = defaultdict(float)
        variance_n = defaultdict(int)
        for i, j in itertools.combinations(relevant_sensors, 2):
            if i in time_deltas and j in time_deltas[i]:
                v = time_deltas[i][j][1]
                variance_sums[i] += v
                variance_sums[j] += v
                variance_n[i] += 1
                variance_n[j] += 1

        for i in variance_sums:
            if variance_n[i] > 0:
                variance_sums[i] /= variance_n[i]
        sorted_var_sums = sorted(variance_sums.items(), key=lambda e: e[1], reverse=True)
        if sorted_var_sums[0][1] > 2* statistics.median(variance_sums.values()):
            relevant_sensors.remove(sorted_var_sums[0][0])
        else:
            break
        
    if debug:
        print("Using", len(relevant_sensors), "/", len(sensor_ids), "sensors for MLAT")
        print(relevant_sensors)


    # prepare locations (gather around 0,0,0 for more accurate calculations)
    ecef_min_coordinates = [1e9, 1e9, 1e9]
    ecef_max_coordinates = [-1e9, -1e9, -1e9]
    for sensor_id in relevant_sensors:
        l = sensor_locations[sensor_id].pos()
        for i in range(3):
            ecef_min_coordinates[i] = min(ecef_min_coordinates[i], l[i])
            ecef_max_coordinates[i] = max(ecef_max_coordinates[i], l[i])

    center_point = np.add(ecef_min_coordinates, ecef_max_coordinates, dtype=np.longdouble) / 2
    # debug
    #center_point = np.zeros(3)
    locations = [np.subtract(sensor_locations[e].pos(), center_point, dtype=np.longdouble) for e in relevant_sensors]

    if debug:
        print("sensors center point:", center_point)
        print("centered sensor locations:", locations)

    # prepare timestamps
    timestamps = [0] # zero represents td_base
    for sensor_id in relevant_sensors[1:]:
        timestamps.append(
            sensor_timestamps[sensor_id] + time_deltas[td_base_sensor][sensor_id][0]
            - sensor_timestamps[td_base_sensor]
        )
    
    if debug:
        print("corrected timestamps:", timestamps)


    myVertexer = Vertexer(np.array(locations, dtype=np.longdouble))
    try:
        calculated_location = myVertexer.find(np.array(timestamps, dtype=np.longdouble), debug=debug)

        if debug:
            print("Uncorrected calc location:", calculated_location)

        target_location = np.add(
            calculated_location,
            center_point
        )
        if debug:
            print("Corrected location:", target_location)
            print("difference:", np.subtract(target_location, calculated_location))
        
        return util.GeoPoint('ecef', *target_location)
    except:
        if debug:
            print("Fail")
            traceback.print_exc()
        pass


def calc_mlat_sympy(sensor_ids, sensor_locations, sensor_timestamps, time_deltas, debug=False):
    assert len(sensor_ids) >= 4

    if debug:
        print("N Sensors:", len(sensor_ids))

    # select sensor subset for mlat
    relevant_sensors = list()
    for sensor_id in sensor_ids:
        if not len(relevant_sensors):
            relevant_sensors.append(sensor_id)
            td_base_sensor = sensor_id
            continue
        if td_base_sensor in time_deltas and sensor_id in time_deltas[td_base_sensor]:
            relevant_sensors.append(sensor_id)

    if debug:
        print("Initial sensor subset:", relevant_sensors)

    if len(relevant_sensors) < 4:
        if debug:
            print("Not enough initally connected sensors")
        return

    while len(relevant_sensors) > 4 and False:
        # find worst of the relevant sensors and potentially exclude it
        variance_sums = defaultdict(float)
        variance_n = defaultdict(int)
        for i, j in itertools.combinations(relevant_sensors, 2):
            if i in time_deltas and j in time_deltas[i]:
                v = time_deltas[i][j][1]
                variance_sums[i] += v
                variance_sums[j] += v
                variance_n[i] += 1
                variance_n[j] += 1

        for i in variance_sums:
            if variance_n[i] > 0:
                variance_sums[i] /= variance_n[i]
        sorted_var_sums = sorted(variance_sums.items(), key=lambda e: e[1], reverse=True)
        if sorted_var_sums[0][1] > 2* statistics.median(variance_sums.values()):
            relevant_sensors.remove(sorted_var_sums[0][0])
        else:
            break
        
    if debug:
        print("Using", len(relevant_sensors), "/", len(sensor_ids), "sensors for MLAT")
        print(relevant_sensors)

    locations = [sensor_locations[e].pos() for e in relevant_sensors]

    # prepare timestamps
    timestamps = [0] # zero represents td_base
    for sensor_id in relevant_sensors[1:]:
        timestamps.append(
            sensor_timestamps[sensor_id] + time_deltas[td_base_sensor][sensor_id][0]
            - sensor_timestamps[td_base_sensor]
        )
    
    if debug:
        print("corrected timestamps:", timestamps)

    n = len(relevant_sensors)

    # Now the real math shenanigans begin

    def lorentz_inner(v, w):
        if debug:
            print("v", sympy.shape(v), v)
            print("w", sympy.shape(w), w)
        assert sympy.shape(v) == (4, 1)
        assert sympy.shape(w) == (4, 1)
        M = sympy.diag(1, 1, 1, -1)
        return ((M * v).T * w)[0]

    a = 0.5 * sympy.Matrix([lorentz_inner(sympy.Matrix([*locations[i], timestamps[i] * C]), sympy.Matrix([*locations[i], timestamps[i] * C])) for i in range(n)])

    e = sympy.ones(n, 1)

    B = sympy.Matrix([[*locations[i], -timestamps[i] * C] for i in range(n)])
    if debug:
        print("B:")
        print(B)
        print("a:")
        print(a)
        print("e:")
        print(e)


    Lambda, u1, u2, u3, u4 = sympy.symbols('lambda u1 u2 u3 u4', real=True)
    u = sympy.Matrix([u1, u2, u3, u4])
    solutions = sympy.solve([
        Lambda - 0.5*lorentz_inner(u, u),
        B.T*B*u - B.T*(a+Lambda*e)
    ], Lambda, u1, u2, u3, u4)

    if debug:
        print("Possible solutions:", solutions)
    
    best_sol = 1e9, None
    for sol in solutions:
        Lambda, u1, u2, u3, u4 = sol
        loc = util.GeoPoint('ecef', u1, u2, u3)
        dist = loc.dist(sensor_locations[relevant_sensors[0]])
        best_sol = min(best_sol, (dist, loc))
        
    return best_sol[1]


    B_plus = (B.T*B)**-1 * B.T

    coef_2 = lorentz_inner(B_plus * e, B_plus * e)
    coef_1 = 2 * (lorentz_inner(B_plus * a, B_plus * e) - 1)
    coef_0 = lorentz_inner(B_plus * a, B_plus * a)

    if debug:
        print("Lambda sq system coefficients:")
        print(type(coef_2), coef_2)
        print(type(coef_1), coef_1)
        print(type(coef_0), coef_0)

    Lambda = sympy.symbols('Lambda', real=True)
    lambda_solutions = sympy.solveset(
        Lambda**2 * coef_2 + Lambda * + coef_1 + coef_0, Lambda, maxsteps=1000
    )

    if debug:
        print("Possible lambdas:", lambda_solutions)



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
        if it > 100 and True:
            break

        assert len(sensor_timestamps) == len(sensor_ids)
        
        #print(msg_id, sensor_ids, sensor_timestamps)
        pos = calc_mlat_sympy(
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


def summarize_accuracy(stage):
    assert stage in ['raw', 'corrected']
    
    util.conn.commit()
    util.cur.execute('''SELECT COUNT(*) FROM messages''')
    print("Number of received messages:", util.cur.fetchone())

    util.cur.execute(f'''SELECT COUNT(*) FROM msg_positions_{stage}''')
    print("Number of calculated positions:", util.cur.fetchone())

    util.cur.execute(f'''
        SELECT |/(
            (msg.ecef_x - msg_pos.ecef_x)^2 +
            (msg.ecef_y - msg_pos.ecef_y)^2 +
            (msg.ecef_z - msg_pos.ecef_z)^2
        ) AS dist

        FROM messages msg
        JOIN msg_positions_{stage} msg_pos
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

    OUTLIER_DISTANCE_CUTOFF = 20000 # 20km
    for messages in tqdm(util.cur.fetchall()):
        # all messages in a path
        messages = list(sorted([list(eval(e)) for e in eval(messages[0])]))
        points = [util.GeoPoint('ecef', *e[1:]) for e in messages]

        bad_msgs = set()
        WINDOW_SIZE = 20
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
        
        print("Removed", len(bad_msgs), "/", len(messages), "positions")

        messages = [e for e in messages if e[0] not in bad_msgs]
        points = [util.GeoPoint('ecef', *e[1:]) for e in messages]

        # now, smooth the path
        WINDOW_SIZE = 10
        smoothed_points = list()
        for i in range(len(points)):
            # sliding window shenanigans
            sum_weights = 0
            super_point = np.array([0.0, 0.0, 0.0])
            for j in range(max(0, i-WINDOW_SIZE//2), min(len(points), i + (WINDOW_SIZE+1)//2)):
                # weights can be changed
                dist = abs(i-j) + 1
                weight = 1/dist
                weight = 1
                sum_weights += weight
                super_point += np.array(points[j].pos()) * weight
            
            super_point /= sum_weights
            smoothed_points.append(super_point)

        assert len(smoothed_points) == len(points)

        util.cur.executemany('''INSERT INTO msg_positions_corrected VALUES(
            %s, %s, %s, %s
        )''', ([(messages[i][0], *smoothed_points[i]) for i in range(len(messages))]))

    util.conn.commit()