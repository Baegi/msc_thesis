import pyproj
import warnings
import math
warnings.filterwarnings("ignore")
import numpy as np
from position_estimator import GeoPoint


def to_ecef(geopoint):
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    x, y, z = pyproj.transform(lla, ecef, geopoint.lon, geopoint.lat, geopoint.alt, radians=False)
    return (x, y, z)

def to_wgs84(x, y, z):
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    lon, lat, alt = pyproj.transform(ecef, lla, x, y, z, radians=False)
    return (lat, lon, alt)

def lorentz_inner(u, v):
    assert len(u) == len(v) == 4
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2] - u[3]*v[3]

C = 299792458


# Bancroft method
B = list()
a = list()


sensor_positions = [
    [ 1, 0, 0 ],
    [ 0, 1, 0 ],
    [ 0, 0, 1 ],
    [-1, 0, 0 ]
]

timestamps = [ 0,0,0,0 ]

assert len(sensor_positions) == len(timestamps)

for i in range(len(sensor_positions)):
    B.append([*sensor_positions[i], C*(-timestamps[i])])
    a.append(0.5*(sensor_positions[i][0]**2 + sensor_positions[i][1]**2 + sensor_positions[i][2]**2 - (C*timestamps[i]))**2)


e = [1] * len(timestamps)

print("B =", B)
print("e =", e)
print("a =", a)
print()

#print(B)
B_plus = np.matmul(np.linalg.inv(np.matmul(np.transpose(B), B)), np.transpose(B))
lambda_coefs = (lorentz_inner(np.matmul(B_plus,e), np.matmul(B_plus,e)), 2*(lorentz_inner(np.matmul(B_plus, a), np.matmul(B_plus,e)) - 1), lorentz_inner(np.matmul(B_plus,a), np.matmul(B_plus,a)))
lambda_roots = np.roots(list(reversed(lambda_coefs)))

#print(lambda_coefs)
#print(lambda_roots)
#if len(lambda_roots) != 2:
#    continue
assert len(lambda_roots) == 2

for root in lambda_roots:
    sol = np.matmul(B_plus, (a + root * np.array(e)))
    print(sol)
    print()

