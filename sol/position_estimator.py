import pyModeS as pms
from pyModeS.decoder.adsb import position
from collections import defaultdict

class GeoPoint:
    def __init__(self, longitude, latitude, altitude) -> None:
        self.lon = longitude
        self.lat = latitude
        self.alt = altitude


last_loc_msg = defaultdict(lambda: [None, None])
plane_tracks = defaultdict(list)


def get_pos(msg, timestamp):
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

        geo_pos = GeoPoint(pos[1], pos[0], alt)
        return geo_pos

    return None


def is_relevant(msg):
    df = pms.df(msg)
    if df == 17 or df == 18:
        tc = pms.typecode(msg)
        if 9 <= tc <= 22 and tc != 19:
            return True
    
    return False