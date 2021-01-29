from thetis import *

import argparse

from adapt_utils.steady.test_cases.point_discharge3d.options import PointDischarge3dOptions


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-level', help="Number of uniform refinements to apply to the initial mesh.")
parser.add_argument('-debug', help="Toggle debugging mode.")
args = parser.parse_args()

# Set parameters
kwargs = {
    'level': int(args.level or 1),
    'plot_pvd': True,
    'debug': bool(args.debug or False),
}
op = PointDischarge3dOptions(approach='fixed_mesh', **kwargs)
op.tracer_family = 'cg'
op.stabilisation_tracer = 'supg'
op.anisotropic_stabilisation = True

# Get analytical solution
Q = FunctionSpace(op.default_mesh, "CG", 1)
analytical = op.analytical_solution(Q)
