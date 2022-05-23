from cmath import isnan
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
C = 299792458 # light speed, meters per second
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
            t_msg_t1 = msg_loc.dist(s1_loc) / C
            t_msg_t2 = msg_loc.dist(s2_loc) / C

            td = (t1 - t_msg_t1) - (t2 - t_msg_t2)
            time_deltas.append(td)

        for i in range(3):
            td_mean = statistics.mean(time_deltas)
            # remove outliers
            time_deltas = [e for e in time_deltas if abs(e - td_mean) <= 10**-i]
            if len(time_deltas) < 2:
                break

        if len(time_deltas) < 2:
            continue

        td_mean = statistics.mean(time_deltas)
        td_var = statistics.variance(time_deltas, xbar=td_mean)
        #print(sensor_id, s2_id, td_mean, td_var, len(time_deltas))

        cur.execute('''
            INSERT INTO time_deltas (sensor_a, sensor_b, mean, variance, num)
            VALUES (%s, %s, %s, %s, %s)
        ''', (sensor_id, s2_id, td_mean, td_var, len(time_deltas)))

    conn.commit()
    return


worker_connections = dict()
def init_worker():
    thread_name = threading.current_thread().getName()
    worker_connections[thread_name] = psycopg2.connect("dbname=thesis user=postgres password=postgres")

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

    # check accuracy using known positions
    
