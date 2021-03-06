from thetis import print_output

import argparse
from time import perf_counter

from adapt_utils.tracer.desalination.solver import AdaptiveDesalinationProblem
from adapt_utils.unsteady.test_cases.idealised_desalination.options import *


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("-level", help="Mesh resolution level in turbine region")
parser.add_argument('-num_meshes', help="Number of meshes in sequence")
parser.add_argument("-debug", help="Print all debugging statements")
parser.add_argument("-debug_mode", help="Choose debugging mode from 'basic' and 'full'")

args = parser.parse_args()


# --- Set parameters

approach = 'fixed_mesh'
level = int(args.level or 0)
kwargs = {
    'approach': approach,
    'level': level,
    'num_meshes': int(args.num_meshes or 1),
    'plot_pvd': True,
    'debug': bool(args.debug or False),
    'debug_mode': args.debug_mode or 'basic',
}
op = IdealisedDesalinationOutfallOptions(**kwargs)


# --- Run model

tp = AdaptiveDesalinationProblem(op)
cpu_timestamp = perf_counter()
tp.solve_adjoint()
cpu_time = perf_counter() - cpu_timestamp
logstr = "Total CPU time: {:.1f} seconds / {:.1f} minutes / {:.3f} hours\n"
logstr = logstr.format(cpu_time, cpu_time/60, cpu_time/3600)
print_output(logstr)
