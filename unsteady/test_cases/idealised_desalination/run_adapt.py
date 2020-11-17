import argparse

from adapt_utils.tracer.desalination.solver import AdaptiveDesalinationProblem
from adapt_utils.unsteady.test_cases.idealised_desalination.options import *


# --- Parse arguments

parser = argparse.ArgumentParser()

# Solver
parser.add_argument('-level', help="Number of uniform refinements to apply to the initial mesh.")

# Mesh adaptation
parser.add_argument('-approach', help="Mesh adaptation approach.")
parser.add_argument('-num_meshes', help="Number of meshes in sequence")
parser.add_argument('-target', help="Target complexity.")
parser.add_argument('-normalisation', help="Metric normalisation strategy.")
parser.add_argument('-norm_order', help="Metric normalisation order.")
parser.add_argument('-min_adapt', help="Minimum number of mesh adaptations.")
parser.add_argument('-max_adapt', help="Maximum number of mesh adaptations.")

# I/O and debugging
parser.add_argument('-offset', help="Toggle between aligned or offset region of interest.")
parser.add_argument('-debug', help="Toggle debugging mode.")
args = parser.parse_args()
p = 'inf' if args.norm_order == 'inf' else float(args.norm_order or 1)


# --- Set parameters

kwargs = {
    'level': int(args.level or 0),

    # Mesh adaptation
    'approach': args.approach or 'dwr',
    'num_meshes': int(args.num_meshes or 20),
    'target': float(args.target or 1.0e+05),
    'norm_order': p,
    'min_adapt': int(args.min_adapt or 3),
    'max_adapt': int(args.max_adapt or 4),  # TODO: Increase?

    # I/O and debugging
    'plot_pvd': True,
    'debug': bool(args.debug or 0),
}
op = IdealisedDesalinationOutfallOptions(**kwargs)
op.normalisation = args.normalisation or 'complexity'  # FIXME: error


# --- Solve

tp = AdaptiveDesalinationProblem(op)
tp.run()
