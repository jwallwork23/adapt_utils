#!/home/jgw116/software/firedrake-pragmatic/bin/python3

import os
import argparse
import numpy as np
import scipy.interpolate as si

from adapt_utils.case_studies.tohoku.options import TohokuOptions


parser = argparse.ArgumentParser()
parser.add_argument('gauge')
args = parser.parse_args()

fname = args.gauge
try:
    assert os.path.exists('.'.join([fname, 'txt']))
except AssertionError:
    raise ValueError("Gauge data not found")

pressure_gauge = 'p0' in fname

if pressure_gauge:
    # Read bathymetry file and get gauge locations
    op = TohokuOptions(setup=False)
    gauge_lon, gauge_lat = op.gauges[fname.upper()]["lonlat"]
    lon, lat, b = op.read_bathymetry_file()
    bath_interp = si.RectBivariateSpline(lat, lon, b)
    b = float(-bath_interp(gauge_lat, gauge_lon))

start_time = 0
N = sum(1 for line in open('.'.join([fname, 'txt']), 'r'))
with open('.'.join([fname, 'txt']), 'r') as infile:
    with open('.'.join([fname, 'dat']), 'w') as outfile:
        for line in range(N):
            words = infile.readline().split()

            # Convert time to seconds
            if pressure_gauge:
                hms = ''.join([words[3], words[4], words[5]])
            else:
                hms = words[0][-7:-1]
            hours, minutes, seconds = int(hms[:2]), int(hms[2:4]), int(hms[4:])
            time = seconds + 60*(minutes + 60*hours)
            if line == 0:
                start_time = time
            time -= start_time

            # Get free surface height
            meas = words[-1][:-1]
            if pressure_gauge:  # FIXME
                g = 9.81                    # Gravitational acceleration
                rho = 1030.0                # Density of seawater
                pressure = float(meas)*100  # Convert to Pa
                elev = pressure/rho/g       # Hydrostatic pressure = density * g * height of fluid column
                elev -= b                   # Subtract bathymetry at rest
            else:
                elev = np.nan if meas == '9999.99' else float(meas)/100  # Convert to metres

            # Write to output
            outfile.write("{:5d} {:6.4f}\n".format(time, elev))
