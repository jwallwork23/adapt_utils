#!/usr/bin/env python3
"""
Preprocess a tidal forcing .txt file to get it in (time, elevation) format, where time is measured
in seconds from the start of the simulation and elevation is measured in metres.

The user should specify the filename using the -fname flag, which defaults to 'forcing'.
"""
import argparse
import os

from adapt_utils.io import index_string, readfile
from adapt_utils.misc import num_days


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-fname')
parser.add_argument('-debug')
args = parser.parse_args()

# Debugging
debug = bool(args.debug or False)
msg = "date {:4d}/{:2s}/{:2s}  time {:2s}:{:2s}:{:2s}  elev {:8.5f}"

# I/O
fname = args.fname or 'forcing'
if fname[-4:] == '.txt':
    fname = fname[:-4]
fname = os.path.join(os.path.dirname(__file__), '.'.join([fname, 'txt']))
assert os.path.exists(fname)
lines = readfile(fname, reverse=False)
fname = os.path.join(os.path.dirname(__file__), 'forcing.dat')

# Loop over data
year_, month_, day_ = None, None, None
hours_, minutes_, seconds_ = None, None, None
start_hours, start_minutes, start_seconds, start_time = None, None, None, None
day_counter = 0
time = 0.0
with open(fname, 'w') as outfile:
    for counter, line in enumerate(lines):
        words = line.split()

        # Skip column headings
        if counter < 3:
            continue

        # Convert time to seconds
        hours, minutes, seconds = [int(letters) for letters in words[1].split(':')]
        if start_time is None:
            start_hours = hours
            start_minutes = minutes
            start_seconds = seconds
            start_time = 3600*start_hours + 60*start_minutes + start_seconds
        time = 3600*hours + 60*minutes + seconds

        # Count date from start
        year, month, day = [int(letters) for letters in words[0].split('-')]
        if year_ is not None:
            if year == year_:
                if month == month_:
                    day_counter += day - day_
                elif month == month_ + 1:
                    day_counter += day + num_days(month_, year_) - day_
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        time += day_counter*86400 - start_time

        # Get elevation
        elev = float(words[-1])

        # Write to output
        outfile.write("{:7d} {:8.5f}\n".format(time, elev))

        # Keep values to use next iteration
        year_ = year
        month_ = month
        day_ = day
        hours_ = hours
        minutes_ = minutes
        seconds_ = seconds

        # Debugging
        if debug:
            month = index_string(month, n=2)
            day = index_string(day, n=2)
            hours = index_string(hours, n=2)
            minutes = index_string(minutes, n=2)
            seconds = index_string(seconds, n=2)
            print(msg.format(year, month, day, hours, minutes, seconds, elev))
