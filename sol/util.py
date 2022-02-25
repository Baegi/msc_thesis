from collections import defaultdict
import itertools
import traceback

C = 299792458 # light speed, meters per second

def calc_timedeltas(msg_pos, sensor_vals):
    print(msg_pos, sensor_vals)
    time_delta = defaultdict(lambda: defaultdict(list))
    #return "HI!"
    #return dict(time_delta)
    if msg_pos is None:
        return dict(time_delta)

    if len(sensor_vals) < 2:
        return dict(time_delta)

    #for t, s in list(sensor_vals):
    #    dists = sorted([s[0].dist(e[1][0]) for e in sensor_vals if e[1] != s])
    #    if dists[int(len(dists)/10)] > 1e6: # ge 1000 km
    #        print("removing:", t, s, dists)
    #        sensor_vals.remove((t, s))

    try:
        sensor_dists_to_msg_origin = {x[1]: msg_pos.dist(x[1][0]) for x in sensor_vals}
        time_to_sensor = { s: x / C for s, x in sensor_dists_to_msg_origin.items() }
    except:
        traceback.print_exc()
        print(msg_pos.lat, msg_pos.lon, msg_pos.alt)
        print({x[1]: (x[1][0].lat, x[1][0].lon, x[1][0].alt) for x in sensor_vals})

    for (t1, s1), (t2, s2) in itertools.combinations(sensor_vals, 2):
        assert s1 != s2
        td = (t1 - time_to_sensor[s1]) - (t2 - time_to_sensor[s2])
        time_delta[s1][s2].append(td)
        time_delta[s2][s1].append(-td)
    
    return dict(time_delta)