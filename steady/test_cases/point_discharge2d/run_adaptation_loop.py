from thetis import create_directory, RectangleMesh

import argparse
import h5py
import os

from adapt_utils.steady.solver import AdaptiveSteadyProblem
from adapt_utils.steady.test_cases.point_discharge2d.options import PointDischarge2dOptions


# --- Parse arguments

parser = argparse.ArgumentParser()

# Solver
parser.add_argument('-level', help="Number of uniform refinements to apply to the initial mesh.")
parser.add_argument('-family', help="Finite element family.")
parser.add_argument('-stabilisation', help="Stabilisation method to use.")

# Mesh adaptation
parser.add_argument('-approach', help="Mesh adaptation approach.")
parser.add_argument('-target', help="Target complexity.")
parser.add_argument('-normalisation', help="Metric normalisation strategy.")
parser.add_argument('-min_adapt', help="Minimum number of mesh adaptations.")
parser.add_argument('-max_adapt', help="Maximum number of mesh adaptations.")

# I/O and debugging
parser.add_argument('-offset', help="Toggle between aligned or offset region of interest.")
parser.add_argument('-debug', help="Toggle debugging mode.")
args = parser.parse_args()


# --- Set parameters

family = args.family or 'cg'
assert family in ('cg', 'dg')
target = float(args.target or 1.0e+03)
level = int(args.level or 0)
offset = bool(args.offset or False)
kwargs = {
    'level': level,

    # QoI
    'aligned': not offset,

    # Mesh adaptation
    'approach': args.approach or 'dwr',
    'norm_order': 1,
    'min_adapt': int(args.min_adapt or 3),
    'max_adapt': int(args.max_adapt or 35),

    # Convergence analysis
    'target_base': 2,
    'outer_iterations': 7,

    # I/O and debugging
    'plot_pvd': True,
    'debug': bool(args.debug or 0),
}
op = PointDischarge2dOptions(**kwargs)
op.tracer_family = family
op.stabilisation = args.stabilisation
op.use_automatic_sipg_parameter = op.tracer_family == 'dg'
op.normalisation = args.normalisation or 'complexity'  # FIXME: error
op.print_debug(op)
di = create_directory(os.path.join(os.path.dirname(__file__), 'outputs', op.approach, 'hdf5'))


# --- Solve

elements = []
qois = []
for n in range(op.outer_iterations):
    op.target = target*op.target_base**n
    op.default_mesh = RectangleMesh(100*2**level, 20*2**level, 50, 10)
    tp = AdaptiveSteadyProblem(op)
    tp.run()
    elements.append(tp.num_cells[-1][0])
    qois.append(tp.qois[-1])
    print("Element count: ", elements)
    print("QoIs:          ", qois)

# Get filenames
ext = args.family
assert ext in ('cg', 'dg')
if ext == 'dg':
    if args.stabilisation in ('lf', 'LF', 'lax_friedrichs'):
        ext += '_lf'
else:
    if args.stabilisation in ('su', 'SU'):
        ext += '_su'
    if args.stabilisation in ('supg', 'SUPG'):
        ext += '_supg'
fname = 'qoi_{:s}'.format(ext)

# Store element count and QoI to HDF5
alignment = 'offset' if offset else 'aligned'
with h5py.File(os.path.join(di, '{:s}_{:s}.h5'.format(fname, alignment)), 'w') as outfile:
    outfile.create_dataset('elements', data=elements)
    outfile.create_dataset('qoi', data=qois)
