from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import threading
import util
import psycopg2

N_THREADS = 32

def calc_td_slice(ind_a, ind_b):
    pass

worker_connections = dict()
def init_worker():
    thread_name = threading.current_thread().getName()
    worker_connections[thread_name] = psycopg2.connect("dbname=thesis user=postgres password=postgres")

def calc_timedeltas():
    
    # get number of sensors
    sensor_ids = util.cur.execute('''
        SELECT id FROM sensors
    ''')

    with ThreadPoolExecutor(initializer=init_worker) as executor:
        executor.map(calc_td_slice, sensor_ids)

    
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

