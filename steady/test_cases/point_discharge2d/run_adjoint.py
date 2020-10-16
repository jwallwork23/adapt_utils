from thetis import create_directory

import argparse
import os

from adapt_utils.io import export_field
from adapt_utils.steady.solver import AdaptiveSteadyProblem
from adapt_utils.steady.test_cases.point_discharge2d.options import PointDischarge2dOptions


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument('-level', help="Number of uniform refinements to apply to the initial mesh.")
parser.add_argument('-family', help="Finite element family")
parser.add_argument('-stabilisation', help="Stabilisation method to use")
parser.add_argument('-offset', help="Toggle between aligned or offset region of interest.")
parser.add_argument('-debug', help="Toggle debugging mode.")
args = parser.parse_args()


# --- Set parameters

family = args.family or 'dg'
assert family in ('cg', 'dg')
offset = bool(args.offset or False)
kwargs = {
    'level': int(args.level or 0),
    'aligned': not offset,
    'plot_pvd': True,
    'debug': bool(args.debug or False),
}
op = PointDischarge2dOptions(approach='fixed_mesh', **kwargs)
op.tracer_family = family
op.stabilisation = args.stabilisation
alignment = 'offset' if offset else 'aligned'
op.di = create_directory(os.path.join(op.di, args.stabilisation or family, alignment))
op.use_automatic_sipg_parameter = op.tracer_family == 'dg'

# TODO: Limiters?


# --- Solve

tp = AdaptiveSteadyProblem(op)
tp.solve_adjoint()
adj = tp.adj_solution_tracer

# Export to HDF5
op.plot_pvd = False
export_field(adj, "Adjoint tracer", "continuous_adjoint", fpath=op.di, plexname=None, op=op)

# TODO: discrete
