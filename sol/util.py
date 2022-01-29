

from ssl import ALERT_DESCRIPTION_UNSUPPORTED_EXTENSION


class Sensor:

    def __init__(self, latitude, longitude, altitude, type) -> None:
        self.lat = latitude
        self.lon = longitude
        self.alt = altitude
        self.type = type

    def __eq__(self, __o: object) -> bool:
        return (
            self.lat == __o.lat and
            self.lon == __o.lon and
            self.alt == __o.alt and
            self.type == __o.type
        )

    def __hash__(self) -> int:
        return hash((self.lat, self.lon, self.alt, self.type))

