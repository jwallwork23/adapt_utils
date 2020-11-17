from thetis import print_output

import argparse
import os
from time import perf_counter

from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.test_cases.desalination.options import DesalinationOutfallOptions


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("-level", help="Mesh resolution level in turbine region")
parser.add_argument("-num_meshes", help="Number of meshes (for debugging)")
parser.add_argument("-end_time", help="End time of simulation in seconds")
parser.add_argument("-extension", help="Optional extension for output directory")
parser.add_argument("-plot_pvd", help="Toggle plotting to .pvd")
parser.add_argument("-debug", help="Print all debugging statements")
parser.add_argument("-debug_mode", help="Choose debugging mode from 'basic' and 'full'")
args = parser.parse_args()


# --- Set parameters

approach = 'fixed_mesh'
level = int(args.level or 0)
plot_pvd = bool(args.plot_pvd or False)
kwargs = {
    'approach': approach,
    'level': level,
    'num_meshes': int(args.num_meshes or 1),
    'plot_pvd': plot_pvd,
    'debug': bool(args.debug or False),
    'debug_mode': args.debug_mode or 'basic',
}
if args.end_time is not None:
    kwargs['end_time'] = float(args.end_time)
op = DesalinationOutfallOptions(**kwargs)
op.end_time = op.T_ramp
op.di = os.path.join(os.path.dirname(__file__), "data", "ramp")


# --- Solve problem

swp = AdaptiveProblem(op)
cpu_timestamp = perf_counter()
swp.solve_forward()
cpu_time = perf_counter() - cpu_timestamp
logstr = "Total CPU time: {:.1f} seconds / {:.1f} minutes / {:.3f} hours\n"
logstr = logstr.format(cpu_time, cpu_time/60, cpu_time/3600)
print_output(logstr)
swp.export_state(0, op.di)
