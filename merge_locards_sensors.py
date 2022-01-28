from sre_constants import CATEGORY_UNI_LINEBREAK
import sys
import os

cur_id = 1
sensors = dict()

if __name__ == "__main__":
    for dir in os.listdir(sys.argv[1]):
        print(dir)
        assert os.path.isdir(os.path.join(sys.argv[1], dir))
        sensor_f = next(iter(filter(
            lambda e: e.endswith('_sensors.csv'), os.listdir(os.path.join(sys.argv[1], dir))
        )))
        with open(os.path.join(sys.argv[1], dir, sensor_f), 'r') as f:
            line = f.readline()
            assert line.startswith('serial,latitude,longitude,height,type')
            while (line := f.readline()):
                serial,latitude,longitude,height,type = line.strip().split(',', 4)
                # if there's another col ('good'), remove from type var
                type = type.split(',')[0]
                key = latitude, longitude, height, type
                if key not in sensors:
                    sensors[key] = cur_id
                    cur_id += 1

        print(cur_id)

    # write output
    with open(sys.argv[2], 'w') as f:
        f.write('serial,latitude,longitude,height,type\n')
        for (s, id) in sensors.items():
            f.write(','.join([str(id), *s]) + '\n')