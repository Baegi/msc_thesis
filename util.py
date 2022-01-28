import pyModeS as pms

def print_info(msg):
    print()
    print("msg len:", len(msg))

    df = pms.df(msg)
    print("downlink format:", df)
    
    icao = pms.icao(msg)
    print("ICAO:", icao)

    if df == 4 or df == 20:
        # altitude surveillance reply
        print("altitude (ft):", pms.altcode(msg))
    elif df == 5 or df == 21:
        # identity surveillance reply
        print("identity code (4-oct):", pms.idcode(msg))
    elif df == 17 or df == 18:
        tc = pms.typecode(msg)
        print("typecode:", tc)

        if 1 <= tc <= 4:
            # id and category
            print("category:", pms.adsb.category(msg))
            print("callsign:", pms.adsb.callsign(msg))
        elif 5 <= tc <= 8:
            # surface position
            print("surface velocity:", pms.adsb.surface_velocity(msg))
        elif 9 <= tc <= 18:
            # airborne position
            print("altitude:", pms.adsb.altitude(msg))
        elif tc == 19:
            # airborne velocity
            print("airborne velocity:", pms.adsb.airborne_velocity(msg))
            print("altitude diff (GNSS-barometric):", pms.adsb.altitude_diff(msg))
        else:
            print("unimplemented TC")
    

def get_announced_flight_path(msgs):
    path = list()
    last_odd = None
    last_even = None
    for msg in msgs:
        pass
    