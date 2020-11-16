import argparse
import os

from adapt_utils.io import export_field, load_mesh
from adapt_utils.steady.solver import AdaptiveSteadyProblem
from adapt_utils.steady.test_cases.point_discharge2d.options import PointDischarge2dOptions


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument('-level', help="Number of uniform refinements to apply to the initial mesh.")
parser.add_argument('-family', help="Finite element family")
parser.add_argument('-stabilisation', help="Stabilisation method to use")
parser.add_argument('-anisotropic_stabilisation', help="Use anisotropic cell size measure?")
parser.add_argument('-load_mesh', help="Approach to load mesh for.")
parser.add_argument('-debug', help="Toggle debugging mode.")
args = parser.parse_args()


# --- Set parameters

family = args.family or 'cg'
assert family in ('cg', 'dg')
kwargs = {
    'level': int(args.level or 0),
    'anisotropic_stabilisation': bool(args.anisotropic_stabilisation or False),
    'plot_pvd': True,
    'debug': bool(args.debug or False),
}
op = PointDischarge2dOptions(approach=args.load_mesh or 'fixed_mesh', **kwargs)
op.tracer_family = family
stabilisation = args.stabilisation or 'supg'
op.stabilisation_tracer = None if stabilisation == 'none' else stabilisation
op.anisotropic_stabilisation = False if args.anisotropic_stabilisation == '0' else True
op.di = os.path.join(op.di, op.stabilisation_tracer or family)
op.use_automatic_sipg_parameter = op.tracer_family == 'dg'
if args.load_mesh is not None:
    op.default_mesh = load_mesh("mesh", fpath=op.di)

# TODO: Limiters?


# --- Solve

tp = AdaptiveSteadyProblem(op)
tp.solve_forward()

# Export to HDF5
op.plot_pvd = False
export_field(tp.fwd_solution_tracer, "Tracer", "finite_element", fpath=op.di, op=op)
