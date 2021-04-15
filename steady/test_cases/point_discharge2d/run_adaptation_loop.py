<<<<<<< HEAD
from firedrake import RectangleMesh
=======
from thetis import create_directory, print_output, RectangleMesh
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a

import argparse
import h5py
import os
import sys

<<<<<<< HEAD
from adapt_utils.io import create_directory, print_output
=======
from adapt_utils.steady.solver import AdaptiveSteadyProblem
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
from adapt_utils.steady.test_cases.point_discharge2d.options import PointDischarge2dOptions


# --- Parse arguments

parser = argparse.ArgumentParser()

# Solver
parser.add_argument('-level', help="Number of uniform refinements to apply to the initial mesh.")
parser.add_argument('-family', help="Finite element family.")
parser.add_argument('-stabilisation', help="Stabilisation method to use.")
parser.add_argument('-anisotropic_stabilisation', help="Use anisotropic cell size measure?")
<<<<<<< HEAD
parser.add_argument('-discrete_adjoint', help="Use discrete adjoint method.")
=======
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a

# Mesh adaptation
parser.add_argument('-approach', help="Mesh adaptation approach.")
parser.add_argument('-target', help="Target complexity.")
parser.add_argument('-normalisation', help="Metric normalisation strategy.")
parser.add_argument('-norm_order', help="Metric normalisation order.")
parser.add_argument('-convergence_rate', help="Convergence rate for anisotropic DWR.")
parser.add_argument('-min_adapt', help="Minimum number of mesh adaptations.")
parser.add_argument('-max_adapt', help="Maximum number of mesh adaptations.")
<<<<<<< HEAD
parser.add_argument('-enrichment_method', help="Choose from {'GE_hp', 'GE_h', 'GE_p', 'PR', 'DQ'}.")
parser.add_argument('-outer_iterations', help="Number of targets to consider.")
=======
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a

# I/O and debugging
parser.add_argument('-offset', help="Toggle between aligned or offset region of interest.")
parser.add_argument('-debug', help="Toggle debugging mode.")
args = parser.parse_args()
p = 'inf' if args.norm_order == 'inf' else float(args.norm_order or 1)
<<<<<<< HEAD
alpha = float(args.convergence_rate or 2)

# discrete_adjoint = bool(args.discrete_adjoint or False)
discrete_adjoint = False if args.discrete_adjoint == "0" else True
if discrete_adjoint:
    from adapt_utils.steady.solver_adjoint import AdaptiveDiscreteAdjointSteadyProblem
    problem = AdaptiveDiscreteAdjointSteadyProblem
else:
    from adapt_utils.steady.solver import AdaptiveSteadyProblem
    problem = AdaptiveSteadyProblem
=======
alpha = float(args.convergence_rate or 10)
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a


# --- Set parameters

family = args.family or 'cg'
assert family in ('cg', 'dg')
target = float(args.target or 5.0e+02)
level = int(args.level or 0)
offset = bool(args.offset or False)
<<<<<<< HEAD
stabilisation = args.stabilisation or 'supg'
anisotropic_stabilisation = False if args.anisotropic_stabilisation == "0" else True
=======
anisotropic_stabilisation = bool(args.anisotropic_stabilisation or False)
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a

# Get filenames
ext = family
if ext == 'dg':
<<<<<<< HEAD
    if stabilisation in ('lf', 'LF', 'lax_friedrichs'):
        ext += '_lf'
else:
    if stabilisation in ('su', 'SU'):
        ext += '_su'
    if stabilisation in ('supg', 'SUPG'):
=======
    if args.stabilisation in ('lf', 'LF', 'lax_friedrichs'):
        ext += '_lf'
else:
    if args.stabilisation in ('su', 'SU'):
        ext += '_su'
    if args.stabilisation in ('supg', 'SUPG'):
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
        ext += '_supg'
    if anisotropic_stabilisation:
        ext += '_anisotropic'
approach = args.approach or 'dwr'
if approach == 'fixed_mesh':
    print_output("Nothing to run.")
    sys.exit(0)
<<<<<<< HEAD
elif 'isotropic_dwr' in approach:
=======
elif approach == 'anisotropic_dwr':
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
    ext += '_{:.0f}'.format(alpha)
else:
    ext += '_inf' if p == 'inf' else '_{:.0f}'.format(p)
fname = 'qoi_{:s}'.format(ext)

<<<<<<< HEAD
both = approach == 'dwr_both' or 'int' in approach or 'avg' in approach
adjoint = 'adjoint' in approach or both
=======
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
kwargs = {
    'level': level,

    # QoI
    'aligned': not offset,

    # Mesh adaptation
    'approach': approach,
    'norm_order': p,
    'convergence_rate': alpha,
    'min_adapt': int(args.min_adapt or 3),
    'max_adapt': int(args.max_adapt or 35),
<<<<<<< HEAD
    'enrichment_method': args.enrichment_method or ('GE_p' if adjoint else 'DQ'),

    # Convergence analysis
    'target_base': 2,
    'outer_iterations': int(args.outer_iterations or 8),
=======

    # Convergence analysis
    'target_base': 2,
    'outer_iterations': 8,
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a

    # I/O and debugging
    'plot_pvd': True,
    'debug': bool(args.debug or 0),
}
op = PointDischarge2dOptions(**kwargs)
op.tracer_family = family
<<<<<<< HEAD
op.stabilisation_tracer = None if stabilisation == 'none' else stabilisation
=======
op.stabilisation = args.stabilisation
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
op.anisotropic_stabilisation = anisotropic_stabilisation
op.use_automatic_sipg_parameter = op.tracer_family == 'dg'
op.normalisation = args.normalisation or 'complexity'  # FIXME: error
op.print_debug(op)
<<<<<<< HEAD
di = os.path.dirname(__file__)
di = create_directory(os.path.join(di, 'outputs', op.approach, op.enrichment_method, 'hdf5'))
=======
di = create_directory(os.path.join(os.path.dirname(__file__), 'outputs', op.approach, 'hdf5'))
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a


# --- Solve

elements = []
<<<<<<< HEAD
dofs = []
=======
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
qois = []
estimators = []
for n in range(op.outer_iterations):
    op.target = target*op.target_base**n
    op.default_mesh = RectangleMesh(100*2**level, 20*2**level, 50, 10)
<<<<<<< HEAD
    tp = problem(op)
    tp.run()
    elements.append(tp.num_cells[-1][0])
    print("Element count: ", elements)
    dofs.append(tp.num_vertices[-1][0])
    print("DoF count: ", dofs)
    qois.append(tp.qois[-1])
    print("QoIs:          ", qois)
    if op.approach in ('dwr', 'dwr_adjoint', 'dwr_both'):
        estimators.append(tp.estimators[op.approach][-1])
=======
    tp = AdaptiveSteadyProblem(op)
    tp.run()
    elements.append(tp.num_cells[-1][0])
    print("Element count: ", elements)
    qois.append(tp.qois[-1])
    print("QoIs:          ", qois)
    if 'dwr' in op.approach:
        estimators.append(tp.estimator[op.approach][-1])
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
        print("Estimators:    ", estimators)

# Store element count and QoI to HDF5
alignment = 'offset' if offset else 'aligned'
with h5py.File(os.path.join(di, '{:s}_{:s}.h5'.format(fname, alignment)), 'w') as outfile:
    outfile.create_dataset('elements', data=elements)
<<<<<<< HEAD
    outfile.create_dataset('dofs', data=dofs)
    outfile.create_dataset('qoi', data=qois)
    if op.approach in ('dwr', 'dwr_adjoint', 'dwr_both'):
=======
    outfile.create_dataset('qoi', data=qois)
    if 'dwr' in op.approach:
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
        outfile.create_dataset('estimators', data=estimators)
