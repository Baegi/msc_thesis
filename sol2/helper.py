
C = .299792458

def dist3d(a, b):
    return (
        (a[0] - b[0])**2 +
        (a[1] - b[1])**2 +
        (a[2] - b[2])**2
    )**0.5

msg_x = 3962761.4726504735
msg_y = 5000969.088226246
msg_z = -22660.272278953715
msg = (msg_x, msg_y, msg_z)

s1_x = 3937904.6779230786
s1_y = 5017239.105008457
s1_z = -56046.23288152806
s1 = (s1_x, s1_y, s1_z)

s2_x = 3968614.893213368
s2_y = 4993170.782835068
s2_z = -10214.326898359133
s2 = (s2_x, s2_y, s2_z)

d_m_s1 = dist3d(msg, s1)
d_m_s2 = dist3d(msg, s2)

dist_diff = d_m_s1 - d_m_s2

print(dist_diff, dist_diff / C)

#import util
#print(util.GeoPoint('wgs84', 51.872524, -0.506857, 181.261017).pos())

import pyproj

lat, lon, alt = 51.872524, -0.506857, 181.261017
transformer = pyproj.Transformer.from_crs(
    {"proj":'latlon', "ellps":'WGS84', "datum":'WGS84'},
    {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
    )

lon1, lat1, alt1 = transformer.transform(lon,lat,alt,radians=False)

print (lat1, lon1, alt1 )
