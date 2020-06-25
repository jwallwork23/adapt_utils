#!/home/jgw116/software/firedrake-pragmatic/bin/python3

import os
import argparse
import numpy as np
import scipy.interpolate as si

from adapt_utils.case_studies.tohoku.options import TohokuOptions


parser = argparse.ArgumentParser()
parser.add_argument('gauge')
args = parser.parse_args()

fname = args.gauge.upper()
try:
    assert os.path.exists('.'.join([fname, 'txt']))
except AssertionError:
    raise ValueError("Gauge data not found")

pressure_gauge = fname[0] != '8'

# Pressure gauge data is converted to free surface elevation so we need to know
#   the total water depth. We interpolate it from GEBCO bathymetry if it is unknown.
if pressure_gauge:
    op = TohokuOptions(setup=False)

    if "depth" in op.gauges[fname]:
        b = op.gauges[fname]["depth"]
    else:
        gauge_lon, gauge_lat = op.gauges[fname]["lonlat"]
        lon, lat, b = op.read_bathymetry_file()
        bath_interp = si.RectBivariateSpline(lat, lon, b)
        b = float(-bath_interp(gauge_lat, gauge_lon))

def readfile(filename, reverse=False):
    """Read a file in reverse order line by line"""
    with open(filename, 'r') as read_obj:
        lines = read_obj.readlines()
        lines = [line.strip() for line in lines]
        if reverse:
            lines = reversed(lines)
        return lines

start_time = 0
counter = 0
lines = readfile('.'.join([fname, 'txt']), reverse=fname[0] == '2')
time = 0
with open('.'.join([fname, 'dat']), 'w') as outfile:
    for line in lines:
        words = line.split()

        # --- Convert time to seconds

        if 'PG' not in fname:
            if pressure_gauge:
                hms = ''.join([words[3], words[4], words[5]])
                day = int(words[2])
                if day != 11:
                    continue
                if fname[0] == '2' and int(words[6]) == 1:
                    continue
            else:
                hms = words[0][-7:-1]
            hours, minutes, seconds = int(hms[:2]), int(hms[2:4]), int(hms[4:])
            if hours < 4 or (hours == 4 and minutes < 45) or hours > 7:
                continue
            time = seconds + 60*(minutes + 60*hours)
            if counter == 0:
                start_time = time
            time -= start_time

        # --- Get free surface height

        meas = words[-1][:-1]

        # Data error code
        if '9999' in meas:
            elev = np.nan

        # Pressure gauges whose data have already been converted to water depth
        elif fname[0] == '2' or 'PG' in fname:
            elev = float(meas)
            if 'PG' in fname:
                elev = elev/100  # Convert to metres
            if 'MPG' not in fname:
                elev -= b

        # Convert other pressure gauge data under hydrostatic assumptions
        elif pressure_gauge:
            g = 9.81                    # Gravitational acceleration
            rho = 1030.0                # Density of seawater
            pressure = float(meas)*100  # Convert to Pa
            elev = pressure/rho/g       # Hydrostatic pressure = density * g * height of fluid column
            elev -= b                   # Subtract bathymetry at rest

        # GPS gauge data are given in centimetres
        else:
            assert fname[0] == '8'
            elev = float(meas)/100  # Convert to metres

        # Write to output
        outfile.write("{:5d} {:6.4f}\n".format(time, elev))
        counter += 1
        time += 1  # This is only relevant to KPG1 and KPG2
