from cmath import isnan
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import itertools
import threading
import util
import psycopg2
import statistics
from tqdm.notebook import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt

N_THREADS = 32
C = .299792458 # light speed, meters per second
sensor_locations = dict()

def calc_td_slice(sensor_id, use_default_conn=False):
    if use_default_conn:
        conn = util.conn
    else:
        thread_name = threading.current_thread().getName()
        conn = worker_connections[thread_name]

    cur = conn.cursor()

    s1_loc = sensor_locations[sensor_id]

    cur.execute('''
        SELECT r2.sensor_id, ARRAY_AGG(r1.sensor_timestamp), ARRAY_AGG(r2.sensor_timestamp), 
            ARRAY_AGG(messages.ecef_x), ARRAY_AGG(messages.ecef_y), ARRAY_AGG(messages.ecef_z) 
        FROM records AS r1
        INNER JOIN records AS r2 ON r1.msg_id = r2.msg_id
        INNER JOIN messages ON messages.id = r1.msg_id
        WHERE r1.sensor_id = %s AND r2.sensor_id > r1.sensor_id
        GROUP BY r2.sensor_id
    ''', (sensor_id,))

    for row in cur.fetchall():
        s2_id, s1_timestamps, s2_timestamps, msg_xs, msg_ys, msg_zs = row
        if len(s1_timestamps) < 2:
            continue

        s2_loc = sensor_locations[s2_id]
        time_deltas = list()

        for t1, t2, msg_x, msg_y, msg_z in zip(s1_timestamps, s2_timestamps, msg_xs, msg_ys, msg_zs):
            msg_loc = util.GeoPoint('ecef', msg_x, msg_y, msg_z)
            d_msg_s1 = msg_loc.dist(s1_loc)
            d_msg_s2 = msg_loc.dist(s2_loc)
            theoretical_td = (d_msg_s1 - d_msg_s2) / C

            td = (t1 - t2) - theoretical_td
            time_deltas.append(td)

        #for i in range(3):
        #    td_mean = statistics.median(time_deltas)
        #    # remove outliers
        #    time_deltas = [e for e in time_deltas if abs(e - td_mean) <= 10**-i]
        #    if len(time_deltas) < 2:
        #        break

        # remove top and bottom 10%
        time_deltas = sorted(time_deltas)[int(len(time_deltas) / 10) : int(len(time_deltas) * 9 / 10)]



        if len(time_deltas) < 2:
            continue

        td_mean = statistics.mean(time_deltas)
        td_var = statistics.variance(time_deltas, xbar=td_mean)
        td_median = statistics.median(time_deltas)
        #print(sensor_id, s2_id, td_mean, td_var, len(time_deltas))

        cur.execute('''
            INSERT INTO time_deltas (sensor_a, sensor_b, mean, variance, num)
            VALUES (%s, %s, %s, %s, %s)
        ''', (sensor_id, s2_id, td_median, td_var, len(time_deltas)))

    conn.commit()
    return


worker_connections = dict()
def init_worker():
    thread_name = threading.current_thread().getName()
    worker_connections[thread_name] = psycopg2.connect("dbname=thesis user=postgres password=postgres")


def calc_theoretical_timedelta(aircraft_pos, s1_pos, s2_pos):
    return (aircraft_pos.dist(s1_pos) - aircraft_pos.dist(s2_pos)) / C


def calc_timedeltas():
    
    # get sensors
    util.cur.execute('SELECT id, ecef_x, ecef_y, ecef_z FROM sensors')
    for id, x, y, z in util.cur.fetchall():
        sensor_locations[id] = util.GeoPoint('ecef', x, y, z)

    util.cur.execute('DELETE FROM time_deltas')
    util.conn.commit()


    #with ThreadPoolExecutor(initializer=init_worker) as executor:
    #    pbar = tqdm(total=len(sensor_locations))
    #    for _ in executor.map(calc_td_slice, sensor_locations.keys()):
    #        pbar.update(1)

    for sensor_id in tqdm(sensor_locations.keys()):
        calc_td_slice(sensor_id, use_default_conn=True)
        #break

    # close DB connections
    for conn in worker_connections.values():
        #conn.cursor().close()
        conn.close()
    
    #threads = list()
    #for i in range(N_THREADS):
    #    ind_a = n_sensors * i / N_THREADS
    #    ind_b = min(n_sensors - 1, n_sensors * (i+1) / N_THREADS - 1)
    #    threads.append(threading.Thread(target=calc_timedeltas, args=(ind_a, ind_b)))
    #    threads[-1].start()
    #
    #print("Waiting for threads to finish...")
    #for i in range(N_THREADS):
    #    threads[i].join()


def calc_timedeltas2():
    util.conn.commit()
    util.cur.execute('DELETE FROM time_deltas')
    # calculate theoretical time deltas
    util.cur.execute('''
        SELECT id, ecef_x, ecef_y, ecef_z FROM sensors
    ''')
    sensor_locations = {e[0]: util.GeoPoint('ecef', *e[1:]) for e in util.cur.fetchall()}

    util.cur.execute('''
    SELECT r1.sensor_id, r2.sensor_id,
        ARRAY_AGG(r1.sensor_timestamp), ARRAY_AGG(r2.sensor_timestamp),
        ARRAY_AGG(msg.ecef_x), ARRAY_AGG(ecef_y), ARRAY_AGG(ecef_z)
    FROM records r1
    JOIN records r2 ON r1.msg_id = r2.msg_id
    JOIN messages msg ON msg.id = r1.msg_id
    WHERE r1.sensor_id < r2.sensor_id
    GROUP BY r1.sensor_id, r2.sensor_id
    -- LIMIT 1
    ''')
    sensor_pairs = util.cur.fetchall()

    time_deltas = list()
    for row in tqdm(sensor_pairs):
        # check if calculated td exists
        s1_id, s2_id = row[:2]
        
        s1_pos = sensor_locations[s1_id]
        s2_pos = sensor_locations[s2_id]

        residuals = list()
        for i in range(len(row[-1])):
            msg_pos = util.GeoPoint('ecef', row[4][i], row[5][i], row[6][i])
            s1_t, s2_t = row[2][i], row[3][i]

            theoretical_td = calc_theoretical_timedelta(msg_pos, s1_pos, s2_pos)
            residual_td = theoretical_td - (s1_t - s2_t)
            residuals.append(residual_td)

        med_resid = statistics.median(residuals)

        # second run, limit error to 500ns, take median of remaining residuals
        residuals = list()
        for i in range(len(row[-1])):
            msg_pos = util.GeoPoint('ecef', row[4][i], row[5][i], row[6][i])
            s1_t, s2_t = row[2][i], row[3][i]

            theoretical_td = calc_theoretical_timedelta(msg_pos, s1_pos, s2_pos)
            residual_td = theoretical_td - (s1_t - s2_t)
            
            # here's the abort condition
            if abs(residual_td - med_resid) > 500:
                continue

            residuals.append(residual_td)

        med_resid = statistics.median(residuals)

        med_error = statistics.median([
            abs(e - med_resid) for e in residuals
        ])

        time_deltas.append((s1_id, s2_id, med_resid, med_error))

    util.cur.executemany('''
        INSERT INTO time_deltas (sensor_a, sensor_b, mean, variance)
        VALUES (%s, %s, %s, %s) 
    ''', time_deltas)
    util.conn.commit()


def propagate_timedeltas():
    
    # get sensors
    util.connect_db()
    util.conn.commit()
    util.cur.execute('SELECT id FROM sensors')
    sensor_ids = util.cur.fetchall()
    util.conn.commit()

    #def floyd_adjust_min(k, i, j):
    #    if i not in time_delta_gaussians or k not in time_delta_gaussians or k not in time_delta_gaussians[i] or j not in time_delta_gaussians[k]:
    #        return i, j, None
    #    if time_delta_gaussians[i][k] is not None and time_delta_gaussians[k][j] is not None:
    #        new_variance = time_delta_gaussians[i][k][1] + time_delta_gaussians[k][j][1]
    #        if time_delta_gaussians[i][j] is None or time_delta_gaussians[i][j][1] > new_variance:
    #            new_mean = time_delta_gaussians[i][k][0] + time_delta_gaussians[k][j][0]
    #            time_delta_gaussians[i][j] = (new_mean, new_variance)
    #            time_delta_gaussians[j][i] = (-new_mean, new_variance)
    #            #print(i, j, new_mean, new_variance)


    # floyd's algorithm
    print("Doing Floyd's algorithm")



    for i in tqdm(range(len(sensor_ids))):
        util.cur.execute('''
            INSERT INTO time_deltas SELECT s1, s2, new_mean, new_variance, -1 FROM (
                SELECT td1.sensor_a AS s1, td2.sensor_b AS s2,
                    td1.mean + td2.mean AS new_mean,
                    td1.variance + td2.variance AS new_variance
                FROM time_deltas td1 JOIN time_deltas td2
                    ON td1.sensor_b = %s AND td2.sensor_a = td1.sensor_b
                ) as x
            ON CONFLICT (sensor_a, sensor_b) DO UPDATE
            SET mean = EXCLUDED.mean, variance = EXCLUDED.variance, num = EXCLUDED.num
            WHERE time_deltas.variance > EXCLUDED.variance
        ''', (sensor_ids[i],))

    util.conn.commit()


def timedelta_statistics():
    util.conn.commit()
    util.cur.execute('SELECT COUNT(*) FROM sensors')
    n_sensors = util.cur.fetchone()[0]

    util.cur.execute('''
        SELECT sensor_a, sensor_b, mean, variance FROM time_deltas
    ''')
    time_deltas = util.cur.fetchall()
    _, _, means, variances = zip(*time_deltas)

    print("n sensors:", n_sensors)
    print("connected sensor pairs:", len(time_deltas), '/', n_sensors * (n_sensors-1), "(" + str(100*len(time_deltas)//(n_sensors*(n_sensors-1))) + "%)")
    print("Mean mean:", statistics.mean(means))
    print("Mean variance:", statistics.mean(variances))
    print("Max variance:", max(variances), "sensors:", next(e for e in time_deltas if e[3] == max(variances)))

    util.cur.execute('''
        SELECT r1.sensor_timestamp, r2.sensor_timestamp, r1.server_timestamp, r2.server_timestamp
        FROM records r1
        JOIN records r2 ON r1.msg_id = r2.msg_id
        WHERE r1.sensor_id = %s AND r2.sensor_id = %s
    ''', (next(e[:2] for e in time_deltas if e[3] == max(variances))))
    s1_timestamps, s2_timestamps, s1_server_t, s2_server_t = zip(*util.cur.fetchall())
    #s1_timestamps = [e - s1_timestamps[0] for e in s1_timestamps]
    #s2_timestamps = [e - s2_timestamps[0] for e in s2_timestamps]

    print(len(s1_timestamps), len(s2_timestamps))
    max_i = max([(abs(s1_timestamps[i] - s2_timestamps[i]), i) for i in range(len(s1_timestamps))])[1]
    print("max_i", max_i)
    print(s1_timestamps[max_i], s1_server_t[max_i])
    print(s2_timestamps[max_i], s2_server_t[max_i])
    print(max([(abs(s1_timestamps[i] - s2_timestamps[i]), i) for i in range(len(s1_timestamps))]))
    fig = plt.figure()
    plt.scatter(
        s1_timestamps,
        [s1_timestamps[i] - s2_timestamps[i] for i in range(len(s1_timestamps))]
    )


def analyze_td_error(delete_bad_sensors=False, variance_cutoff=1e8):
    # calculate theoretical time deltas
    util.cur.execute('''
        SELECT id, ecef_x, ecef_y, ecef_z FROM sensors
    ''')
    sensor_locations = {e[0]: util.GeoPoint('ecef', *e[1:]) for e in util.cur.fetchall()}

    util.cur.execute('''
        SELECT sensor_a, ARRAY_AGG(sensor_b), ARRAY_AGG(mean)
        FROM time_deltas
        GROUP BY sensor_a
    ''')
    time_deltas = {
        e[0]: dict(zip(e[1], e[2])) for e in util.cur.fetchall()
    }

    util.cur.execute('''
    SELECT r1.sensor_id, r2.sensor_id,
        ARRAY_AGG(r1.sensor_timestamp), ARRAY_AGG(r2.sensor_timestamp),
        ARRAY_AGG(msg.ecef_x), ARRAY_AGG(ecef_y), ARRAY_AGG(ecef_z)
    FROM records r1
    JOIN records r2 ON r1.msg_id = r2.msg_id
    JOIN messages msg ON msg.id = r1.msg_id
    WHERE r1.sensor_id < r2.sensor_id
    GROUP BY r1.sensor_id, r2.sensor_id
    -- LIMIT 1
    ''')
    sensor_pairs = util.cur.fetchall()

    
    all_errors = list()
    sensor_td_variances = defaultdict(list)
    for row in tqdm(sensor_pairs):
        if len(row[-1]) < 2:
            continue
        # check if calculated td exists
        s1_id, s2_id = row[:2]
        
        if s1_id not in time_deltas or s2_id not in time_deltas[s1_id]:
            continue

        s1_pos = sensor_locations[s1_id]
        s2_pos = sensor_locations[s2_id]

        td = time_deltas[s1_id][s2_id]
        sensor_errors = list()
        residuals = list()
        for i in range(len(row[-1])):
            msg_pos = util.GeoPoint('ecef', row[4][i], row[5][i], row[6][i])
            s1_t, s2_t = row[2][i], row[3][i]

            theoretical_td = calc_theoretical_timedelta(msg_pos, s1_pos, s2_pos)
            residual_td = theoretical_td - (s1_t - s2_t)
            residuals.append(residual_td)
            sensor_errors.append(
                td -
                residual_td
            )
        all_errors.extend(sensor_errors)
        sensor_td_variances[s1_id].append(statistics.variance(sensor_errors))
        sensor_td_variances[s2_id].append(statistics.variance(sensor_errors))


    def remove_outliers(items, q):
        n = len(items)
        return sorted(items)[int(n*q):int(n*(1-q))]

    print("len/min/max errors:", len(all_errors), min(all_errors), max(all_errors))
    all_errors = remove_outliers(all_errors, 0.05)
    print("len/min/max errors:", len(all_errors), min(all_errors), max(all_errors))
    #plt.figure()
    #plt.hist(td_resid_diffs, bins=100)
    plt.figure()
    plt.title("All errors")
    #plt.hist(remove_outliers(all_errors, 0), bins=100)
    plt.hist([e for e in all_errors if abs(e) <= 1000], bins=100)

    plt.figure()
    plt.title("Median error per sensor")
    plt.hist([statistics.median(e) for e in sensor_td_variances.values()], bins=100)


    if delete_bad_sensors:
        bad_sensors = list()
        print("Deleting bad sensors...")
        for s_id, variances in sensor_td_variances.items():
            if statistics.median(variances) > variance_cutoff:
                bad_sensors.append(s_id)
                util.cur.execute('DELETE FROM sensors WHERE id = %s', (s_id,))

        util.conn.commit()

        print("deleted", len(bad_sensors), "sensors")
        plt.figure()
        plt.title("Median error per sensor")
        plt.hist([statistics.median(e[1]) for e in sensor_td_variances.items() if e[0] not in bad_sensors], bins=100)

            
        

