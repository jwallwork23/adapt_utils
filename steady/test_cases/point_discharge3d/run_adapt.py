import argparse

from adapt_utils.io import save_mesh
from adapt_utils.steady.solver3d import AdaptiveSteadyProblem3d
from adapt_utils.steady.test_cases.point_discharge3d.options import PointDischarge3dOptions


# --- Parse arguments

parser = argparse.ArgumentParser()

# Solver
parser.add_argument('-level', help="Number of uniform refinements to apply to the initial mesh.")

# Mesh adaptation
parser.add_argument('-approach', help="Mesh adaptation approach.")
parser.add_argument('-target', help="Target complexity.")
parser.add_argument('-normalisation', help="Metric normalisation strategy.")
parser.add_argument('-norm_order', help="Metric normalisation order.")
parser.add_argument('-convergence_rate', help="Convergence rate for anisotropic DWR.")
parser.add_argument('-min_adapt', help="Minimum number of mesh adaptations.")
parser.add_argument('-max_adapt', help="Maximum number of mesh adaptations.")

# I/O and debugging
parser.add_argument('-offset', help="Toggle between aligned or offset region of interest.")
parser.add_argument('-debug', help="Toggle debugging mode.")
args = parser.parse_args()
p = 'inf' if args.norm_order == 'inf' else float(args.norm_order or 1)
alpha = float(args.convergence_rate or 10)


# --- Set parameters

kwargs = {
    'level': int(args.level or 1),

    # QoI
    'aligned': not bool(args.offset or False),

    # Mesh adaptation
    'approach': args.approach or 'weighted_hessian',
    'target': float(args.target or 1.0e+04),
    'norm_order': p,
    'convergence_rate': alpha,
    'min_adapt': int(args.min_adapt or 3),
    'max_adapt': int(args.max_adapt or 35),

    # I/O and debugging
    'plot_pvd': True,
    'debug': bool(args.debug or 0),
}
op = PointDischarge3dOptions(**kwargs)
op.tracer_family = 'cg'
op.stabilisation = 'supg'
op.anisotropic_stabilisation = True
op.normalisation = args.normalisation or 'complexity'  # FIXME: error
op.print_debug(op)


# --- Solve

tp = AdaptiveSteadyProblem3d(op)
tp.run()

# Export to HDF5
save_mesh(tp.mesh, "mesh", fpath=op.di)
