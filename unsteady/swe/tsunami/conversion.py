"""
Top matter courtesy of Tobias Bieniek, 2012.
"""
import ufl
import numpy as np
from math import pi, sqrt
from thetis import print_output


class OutOfRangeError(ValueError):
    pass


__all__ = ["to_latlon", "from_latlon", "lonlat_to_utm", "utm_to_lonlat", "degrees", "radians"]


K0 = 0.9996

E = 0.00669438
E2 = E*E
E3 = E2*E
E_P2 = E/(1.0 - E)

SQRT_E = sqrt(1 - E)
_E = (1 - SQRT_E)/(1 + SQRT_E)
_E2 = _E*_E
_E3 = _E2*_E
_E4 = _E3*_E
_E5 = _E4*_E

M1 = 1 - E/4 - 3*E2/64 - 5*E3/256
M2 = 3 * E/8 + 3*E2/32 + 45*E3/1024
M3 = 15*E2/256 + 45*E3/1024
M4 = 35*E3/3072

P2 = 3.0/2*_E-27.0/32*_E3 + 269.0/512*_E5
P3 = 21.0/16*_E2-55.0/32*_E4
P4 = 151.0/96*_E3-417.0/128*_E5
P5 = 1097.0/512*_E4

R = 6378137

ZONE_LETTERS = "CDEFGHJKLMNPQRSTUVWXX"


def degrees(rad):
    rad *= 180.0/pi
    return rad


def radians(deg):
    deg *= pi/180.0
    return deg


def to_latlon(easting, northing, zone_number, zone_letter=None, northern=None, force_longitude=False, coords=None):
    """
    Convert UTM coordinates to latitude-longitude, courtesy of Tobias Bieniek, 2012 (with some
    minor edits).

    :arg easting: eastward-measured Cartesian geographic distance.
    :arg northing: northward-measured Cartesian geographic distance.
    :arg zone_number: UTM zone number (increasing eastward).
    :param zone_letter: UTM zone letter (increasing alphabetically northward).
    :param northern: specify northern or southern hemisphere.
    :param coords: coordinate field of mesh (used to check validity of coordinates).
    :return: latitude-longitude coordinate pair.
    """
    if not zone_letter and northern is None:
        raise ValueError('either zone_letter or northern needs to be set')

    elif zone_letter and northern is not None:
        raise ValueError('set either zone_letter or northern, but not both')

    if not force_longitude:
        if not 100000 <= easting < 1000000:
            raise OutOfRangeError('easting {:f} out of range (must be between 100,000 m and 999,999 m)'.format(easting))

    msg = 'northing out of range (must be between 0 m and 10,000,000 m)'
    if isinstance(northing, ufl.indexed.Indexed):
        from firedrake import sin, cos, sqrt
        if coords is None:
            print_output("WARNING: Cannot check validity of coordinates.")
        else:
            minval, maxval = coords.dat.data[:, 1].min(), coords.dat.data[:, 1].max()
            if not (0 <= minval and maxval <= 10000000):
                raise OutOfRangeError(msg)
    elif isinstance(northing, np.ndarray):
        from numpy import sin, cos, sqrt
        minval, maxval = northing.min(), northing.max()
        if not (0 <= minval and maxval <= 10000000):
            raise OutOfRangeError(msg)
    else:
        from math import sin, cos, sqrt
        if not 0 <= northing <= 10000000:
            raise OutOfRangeError(msg)
    if not 1 <= zone_number <= 60:
        raise OutOfRangeError('zone number out of range (must be between 1 and 60)')

    if zone_letter:
        zone_letter = zone_letter.upper()

        if not 'C' <= zone_letter <= 'X' or zone_letter in ['I', 'O']:
            raise OutOfRangeError('zone letter out of range (must be between C and X)')

        northern = zone_letter >= 'N'

    x = easting - 500000
    y = northing

    if not northern:
        y -= 10000000

    m = y/K0
    mu = m/R/M1

    p_rad = (mu + P2*sin(2*mu) + P3*sin(4*mu) + P4*sin(6*mu) + P5*sin(8*mu))

    p_sin = sin(p_rad)
    p_sin2 = p_sin*p_sin

    p_cos = cos(p_rad)

    p_tan = p_sin/p_cos
    p_tan2 = p_tan*p_tan
    p_tan4 = p_tan2*p_tan2

    ep_sin = 1 - E*p_sin2
    ep_sin_sqrt = sqrt(1 - E*p_sin2)

    n = R/ep_sin_sqrt
    r = (1 - E)/ep_sin

    c = _E*p_cos**2
    c2 = c*c

    d = x/n/K0
    d2 = d*d
    d3 = d2*d
    d4 = d3*d
    d5 = d4*d
    d6 = d5*d

    latitude = (p_rad - p_tan/r*(d2/2 - d4/24 * (5 + 3*p_tan2 + 10*c - 4*c2 - 9*E_P2)) + d6/720 * (61 + 90*p_tan2 + 298 * c + 45*p_tan4 - 252*E_P2 - 3*c2))

    longitude = (d - d3/6 * (1 + 2*p_tan2 + c) + d5/120 * (5 - 2*c + 28*p_tan2 - 3*c2 + 8*E_P2 + 24*p_tan4))/p_cos

    return degrees(latitude), degrees(longitude) + zone_number_to_central_longitude(zone_number)


def from_latlon(latitude, longitude, force_zone_number=None, zone_info=False, coords=None):
    """
    Convert latitude-longitude coordinates to UTM, courtesy of Tobias Bieniek, 2012.

    :arg latitude: northward anglular position, origin at the Equator.
    :arg longitude: eastward angular position, with origin at the Greenwich Meridian.
    :param force_zone_number: force coordinates to fall within a particular UTM zone.
    :param zone_info: output zone letter and number.
    :param coords: coordinate field of mesh (used to check validity of coordinates).
    :return: UTM coordinate 4-tuple.
    """
    lat_msg = 'latitude out of range (must be between 80 deg S and 84 deg N)'
    lon_msg = 'longitude out of range (must be between 180 deg W and 180 deg E)'
    if isinstance(latitude, ufl.indexed.Indexed):
        from firedrake import sin, cos, sqrt
        if coords is None:
            print_output("WARNING: Cannot check validity of coordinates.")
        else:
            minval, maxval = coords.dat.data[:, 0].min(), coords.dat.data[:, 0].max()
            if not (-80.0 <= minval and maxval <= 84.0):
                raise OutOfRangeError(lon_msg)
            minval, maxval = coords.dat.data[:, 1].min(), coords.dat.data[:, 1].max()
            if not (-180.0 <= minval and maxval <= 180.0):
                raise OutOfRangeError(lat_msg)
    elif isinstance(latitude, np.ndarray):
        from numpy import sin, cos, sqrt
        minval, maxval = longitude.min(), longitude.max()
        if not (-180.0 <= minval and maxval <= 180.0):
            raise OutOfRangeError(lon_msg)
        minval, maxval = latitude.min(), latitude.max()
        if not (-80.0 <= minval and maxval <= 84.0):
            raise OutOfRangeError(lat_msg)
    else:
        from math import sin, cos, sqrt
        if not -180.0 <= longitude <= 180.0:
            raise OutOfRangeError(lon_msg)
        if not -80.0 <= latitude <= 84.0:
            raise OutOfRangeError(lat_msg)

    lat_rad = radians(latitude)
    lat_sin = sin(lat_rad)
    lat_cos = cos(lat_rad)

    lat_tan = lat_sin/lat_cos
    lat_tan2 = lat_tan*lat_tan
    lat_tan4 = lat_tan2*lat_tan2

    if force_zone_number is None:
        zone_number = latlon_to_zone_number(latitude, longitude)
    else:
        zone_number = force_zone_number

    lon_rad = radians(longitude)
    central_lon_rad = radians(zone_number_to_central_longitude(zone_number))

    n = R/sqrt(1 - E*lat_sin**2)
    c = E_P2*lat_cos**2

    a = lat_cos*(lon_rad - central_lon_rad)
    a2 = a*a
    a3 = a2*a
    a4 = a3*a
    a5 = a4*a
    a6 = a5*a

    m = R*(M1*lat_rad - M2*sin(2*lat_rad) + M3*sin(4*lat_rad) - M4*sin(6 * lat_rad))

    easting = K0*n*(a + a3/6*(1 - lat_tan2 + c) + a5/120*(5 - 18*lat_tan2 + lat_tan4 + 72*c - 58*E_P2)) + 500000

    northing = K0*(m + n*lat_tan*(a2/2 + a4/24*(5 - lat_tan2 + 9*c + 4*c**2) + a6/720*(61 - 58*lat_tan2 + lat_tan4 + 600*c - 330*E_P2)))

    if isinstance(latitude, ufl.indexed.Indexed):
        if coords.dat.data[:, 1].min() < 0:
            northing += 10000000
    elif isinstance(latitude, np.ndarray):
        if latitude.min() < 0:
            northing += 10000000
    else:
        if latitude < 0:
            northing += 10000000

    if zone_info:
        return easting, northing, zone_number, latitude_to_zone_letter(latitude)
    else:
        return easting, northing


def latitude_to_zone_letter(latitude):
    """
    Convert latitude UTM letter, courtesy of Tobias Bieniek, 2012.

    :arg latitude: northward anglular position, origin at the Equator.
    :return: UTM zone letter (increasing alphabetically northward).
    """
    return ZONE_LETTERS[int(latitude + 80) >> 3] if -80 <= latitude <= 84 else None


def latlon_to_zone_number(latitude, longitude):
    """
    Convert a latitude-longitude coordinate pair to UTM zone, courtesy of Tobias Bieniek, 2012.

    :arg latitude: northward anglular position, origin at the Equator.
    :arg longitude: eastward angular position, with origin at the Grenwich Meridian.
    :return: UTM zone number (increasing eastward).
    """
    if 56 <= latitude < 64 and 3 <= longitude < 12:
        return 32

    if 72 <= latitude <= 84 and longitude >= 0:
        if longitude <= 9:
            return 31
        elif longitude <= 21:
            return 33
        elif longitude <= 33:
            return 35
        elif longitude <= 42:
            return 37

    return int((longitude + 180)/6) + 1


def zone_number_to_central_longitude(zone_number):
    """
    Convert a UTM zone number to the corresponding central longitude, courtesy of Tobias Bieniek, 2012.

    :arg zone_number: UTM zone number (increasing eastward).
    :return: central eastward angular position of the UTM zone, with origin at the Grenwich Meridian.
    """
    return (zone_number - 1)*6 - 180 + 3


def lonlat_to_utm(longitude, latitude, force_zone_number, **kwargs):
    """
    Transformation from longitude-latitude coordinates to UTM coordinates.

    :arg longitude: eastward angular position, with origin at the Grenwich Meridian.
    :arg latitude: northward anglular position, origin at the Equator.
    :param force_zone_number: force coordinates to fall within a particular UTM zone.
    """
    return from_latlon(latitude, longitude, force_zone_number, **kwargs)


def utm_to_lonlat(x, y, zone_number, **kwargs):
    """
    Transformation from UTM coordinates to longitude-latitude coordinates.

    :args x,y: UTM coordinates.
    :arg zone_number: UTM zone.
    """
    lat, lon = to_latlon(x, y, zone_number, **kwargs)
    return lon, lat
