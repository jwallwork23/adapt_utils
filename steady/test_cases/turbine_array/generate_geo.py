from adapt_utils.swe.turbine.meshgen import *
from adapt_utils.test_cases.steady_turbine.options import *

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-level', help='Mesh resolution')
args = parser.parse_args()

levels = ('xcoarse', 'coarse', 'medium', 'fine', 'xfine')


def make_geo(level, offset):
    assert level in levels
    generate_geo_file(Steady2TurbineOptions(offset=offset), level=level)


for offset in (0, 1, 2):
    if args.level is not None:
        make_geo(args.level, offset)
    else:
        for l in levels:
            make_geo(l, offset)
