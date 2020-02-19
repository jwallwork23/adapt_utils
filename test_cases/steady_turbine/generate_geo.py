from adapt_utils.swe.turbine.meshgen import *
from adapt_utils.test_cases.steady_turbine.options import *

import argparse


def make_geo(level, offset):
    assert level in levels
    if offset is None:
        generate_geo_file(Steady2TurbineOptions(), level=level)
        generate_geo_file(Steady2TurbineOffsetOptions(), level=level, tag='offset')
    else:
        op = Steady2TurbineOffsetOptions() if offset else Steady2TurbineOptions()
        generate_geo_file(op, level=level, tag='offset' if offset else None)

parser = argparse.ArgumentParser()
parser.add_argument('-level', help='Mesh resolution')
parser.add_argument('-offset', help='Toggle aligned or offset configuration')
args = parser.parse_args()

levels = ('xcoarse', 'coarse', 'medium', 'fine', 'xfine')
level = args.level
offset = bool(args.offset or False)

if level is not None:
    make_geo(level, offset)
else:
    for l in levels:
        make_geo(l, offset)
