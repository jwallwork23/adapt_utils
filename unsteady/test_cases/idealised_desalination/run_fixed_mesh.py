from thetis import print_output

import argparse
import numpy as np
import os
from time import perf_counter

from adapt_utils.tracer.desalination.solver import AdaptiveDesalinationProblem
from adapt_utils.unsteady.test_cases.idealised_desalination.options import *


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("-level", help="Mesh resolution level in turbine region")
parser.add_argument("-num_meshes", help="Number of meshes (for debugging)")
parser.add_argument("-plot_pvd", help="Toggle plotting to .pvd")
parser.add_argument("-debug", help="Print all debugging statements")
parser.add_argument("-debug_mode", help="Choose debugging mode from 'basic' and 'full'")

args = parser.parse_args()


# Create directories and check if spun-up solution exists
ramp_dir = os.path.join(os.path.dirname(__file__), "data", "ramp")
spun = np.all([os.path.isfile(os.path.join(ramp_dir, f + ".h5")) for f in ('velocity', 'elevation')])

# --- Set parameters

approach = 'fixed_mesh'
level = int(args.level or 0)
plot_pvd = bool(args.plot_pvd or False)
kwargs = {
    'approach': approach,
    'level': level,
    'num_meshes': int(args.num_meshes or 1),
    'plot_pvd': plot_pvd,
    'spun': bool(spun),
    'debug': bool(args.debug or False),
    'debug_mode': args.debug_mode or 'basic',
}
op = IdealisedDesalinationOutfallOptions(**kwargs)


# --- Run model

if not op.spun:
    raise ValueError("Spin-up data not found.")
swp = AdaptiveDesalinationProblem(op, ramp_dir=ramp_dir)

# Solve forward problem
cpu_timestamp = perf_counter()
swp.solve_forward()
cpu_time = perf_counter() - cpu_timestamp
logstr = "Total CPU time: {:.1f} seconds / {:.1f} minutes / {:.3f} hours\n"
logstr = logstr.format(cpu_time, cpu_time/60, cpu_time/3600)
print_output(logstr)
