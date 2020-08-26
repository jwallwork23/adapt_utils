#!/usr/bin/env python3

import argparse
import numpy as np
import os
import scipy.interpolate as si

from adapt_utils.case_studies.tohoku.options.options import TohokuOptions
from adapt_utils.misc import readfile


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('gauges', nargs='+', help="Gauge(s) to pre-process. Enter 'all' to pre-process all.")
parser.add_argument('-debug', help="Toggle debugging mode")
args = parser.parse_args()

# Create an Options class for convenience
op = TohokuOptions(debug=bool(args.debug or False))

# Get gauge names
gauges = [gauge.upper() for gauge in args.gauges]
if 'ALL' in gauges:
    gauges = list(op.gauges.keys())

# Check data files exists
for gauge in gauges:
    if not os.path.exists(os.path.join(os.path.dirname(__file__), gauge + '.txt')):
        raise ValueError("Gauge data not found")

# Loop over gauges
for gauge in gauges:

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
                    if gauge[0] == '2' and int(words[6]) == 1:
                        continue
                else:
                    day = int(words[0][6:8])
                    hms = words[0][-7:-1]
                if day < 11:
                    continue
                elif day > 11:
                    break
                assert len(hms) == 6
                hours, minutes, seconds = int(hms[:2]), int(hms[2:4]), int(hms[4:])

                # Near-field gauge data are in JST
                if "near_field" in op.gauges[gauge]["class"]:
                    if hours < 14 or (hours == 14 and minutes < 46) or hours > 17:
                        continue

                # DART gauge data are in GMT
                else:
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

            # Data error codes
            if '9999' in meas or '=' in meas:
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
            time_prev = time
            time += 1  # For KPG1, KPG2, MPG1, MPG2
