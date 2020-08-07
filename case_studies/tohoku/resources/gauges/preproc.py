#!/usr/bin/env python3

import argparse
import numpy as np
import os
import scipy.interpolate as si

from adapt_utils.case_studies.tohoku.options import TohokuOptions


def readfile(filename, reverse=False):
    """
    Read a file line-by-line.

    :kwarg reverse: read the lines in reverse order.
    """
    with open(filename, 'r') as read_obj:
        lines = read_obj.readlines()
    lines = [line.strip() for line in lines]
    if reverse:
        lines = reversed(lines)
    return lines

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('gauge')
parser.add_argument('-debug')
args = parser.parse_args()

# Check data file exists
gauge = args.gauge.upper()
try:
    assert os.path.exists('.'.join([gauge, 'txt']))
except AssertionError:
    raise ValueError("Gauge data not found")

# Create an Options class for convenience
op = TohokuOptions(debug=bool(args.debug or False))

# Get depth
# =========
# Pressure gauge data is converted to free surface elevation so we need to know the total water
# depth. We interpolate it from ETOPO1 bathymetry if it is unknown.
op.print_debug("GAUGES: Getting depth for gauge {:s}".format(gauge))
if gauge in op.pressure_gauges:
    if "depth" in op.gauges[gauge]:
        b = op.gauges[gauge]["depth"]
    else:
        gauge_lon, gauge_lat = op.gauges[gauge]["lonlat"]
        lon, lat, b = op.read_bathymetry_file()
        b = float(-si.RectBivariateSpline(lat, lon, b)(gauge_lat, gauge_lon))
op.print_debug("GAUGES: Done!")

# I/O
fname = os.path.join(os.path.dirname(__file__), '.'.join([gauge, 'txt']))
assert os.path.exists(fname)
lines = readfile(fname, reverse=gauge[0] == '2')
fname = os.path.join(os.path.dirname(__file__), '.'.join([gauge, 'dat']))

# Loop over data
start_time, time = None, 0
with open(fname, 'w') as outfile:
    for counter, line in enumerate(lines):
        words = line.split()

        # Skip column headings
        if gauge[0] == "8" and counter < 3:
            continue

        # --- Convert time to seconds

        if 'PG' not in gauge:

            # Get time in hours, minutes and seconds
            if gauge in op.pressure_gauges:
                hms = ''.join([words[3], words[4], words[5]])
                day = int(words[2])
                if day != 11:
                    continue
                if gauge[0] == '2' and int(words[6]) == 1:
                    continue
            else:
                hms = words[0][-7:-1]
            assert len(hms) == 6
            hours, minutes, seconds = int(hms[:2]), int(hms[2:4]), int(hms[4:])

            # Some gauge data are in JST
            if op.gauges[gauge]["class"] == "near_field_pressure":
                if hours < 14 or (hours == 14 and minutes < 46) or hours > 17:
                    continue

            # Other gauge data are in GMT
            else:
                # if hours < 4 or (hours == 4 and minutes < 46) or hours > 7:
                if hours < 5 or (hours == 5 and minutes < 46) or hours > 8:
                    continue

            # Get time in seconds
            time = seconds + 60*(minutes + 60*hours)
            if start_time is None:
                start_time = time
            time -= start_time

        # --- Get free surface height

        meas = words[-1][:-1]
        op.print_debug("time {:8f} meas {:}".format(time, meas))

        # Data error code
        if '9999' in meas:
            elev = np.nan

        # Pressure gauges whose data have already been converted to water depth
        elif gauge[0] == '2' or 'PG' in gauge:
            elev = float(meas)
            if 'PG' in gauge:
                elev = elev/100  # Convert to metres
            if 'MPG' not in gauge:
                elev -= b

        # Convert other pressure gauge data under hydrostatic assumptions
        elif gauge in op.pressure_gauges:
            g = 9.81                    # Gravitational acceleration
            rho = 1030.0                # Density of seawater
            pressure = float(meas)*100  # Convert to Pa
            elev = pressure/rho/g       # Hydrostatic pressure = density * g * height of fluid column
            elev -= b                   # Subtract bathymetry at rest

        # GPS gauge data are given in centimetres
        else:
            assert gauge[0] == '8'
            elev = float(meas)/100  # Convert to metres

        # Write to output
        outfile.write("{:5d} {:6.4f}\n".format(time, elev))
        time += 1  # For KPG1, KPG2, MPG1, MPG2
