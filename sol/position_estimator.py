import pyModeS as pms
from pyModeS.decoder.adsb import position
from collections import defaultdict
from geopy import distance
import math

class GeoPoint:
    def __init__(self, latitude, longitude, altitude) -> None:
        assert -90 <= latitude <= 90
        assert -180 <= longitude <= 180

        self.lat = latitude
        self.lon = longitude
        self.alt = altitude

    def dist(self, other):
        flat_dist = distance.distance((self.lat, self.lon), (other.lat, other.lon)).m
        # inaccurately throw mister pythagoras into the calculation
        return math.sqrt(flat_dist**2 + (self.alt - other.alt)**2)


    def __eq__(self, __o: object) -> bool:
        return self.lat == __o.lat and self.lon == __o.lon and self.alt == __o.alt

    
    def __str__(self) -> str:
        return f"Latitude: {self.lat}, Longitude: {self.lon}, Altitude: {self.alt}"

    
    def __hash__(self) -> int:
        return hash((self.lon, self.lat, self.alt))

last_loc_msg = defaultdict(lambda: [None, None])
plane_tracks = defaultdict(list)


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

        pos = pos or [None, None]

        geo_pos = GeoPoint(*pos, alt)
        return geo_pos

    return None


def is_relevant(msg):
    df = pms.df(msg)
    if df == 17 or df == 18:
        tc = pms.typecode(msg)
        if 9 <= tc <= 22 and tc != 19:
            return True
    
    return False


