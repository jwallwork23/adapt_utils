import argparse
import os

<<<<<<< HEAD
from adapt_utils.io import create_directory, File, save_mesh
=======
from adapt_utils.io import save_mesh
from adapt_utils.steady.solver import AdaptiveSteadyProblem
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
from adapt_utils.steady.test_cases.point_discharge2d.options import PointDischarge2dOptions


# --- Parse arguments

parser = argparse.ArgumentParser()
<<<<<<< HEAD
parser.add_argument('approach', help="Mesh adaptation approach.")
parser.add_argument('-discrete_adjoint', help="Use discrete adjoint method.")
=======
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a

# Solver
parser.add_argument('-level', help="Number of uniform refinements to apply to the initial mesh.")
parser.add_argument('-family', help="Finite element family.")
parser.add_argument('-stabilisation', help="Stabilisation method to use.")
parser.add_argument('-anisotropic_stabilisation', help="Use anisotropic cell size measure?")

# Mesh adaptation
<<<<<<< HEAD
=======
parser.add_argument('-approach', help="Mesh adaptation approach.")
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
parser.add_argument('-target', help="Target complexity.")
parser.add_argument('-normalisation', help="Metric normalisation strategy.")
parser.add_argument('-norm_order', help="Metric normalisation order.")
parser.add_argument('-convergence_rate', help="Convergence rate for anisotropic DWR.")
parser.add_argument('-min_adapt', help="Minimum number of mesh adaptations.")
parser.add_argument('-max_adapt', help="Maximum number of mesh adaptations.")
<<<<<<< HEAD
parser.add_argument('-enrichment_method', help="Choose from {'GE_hp', 'GE_h', 'GE_p', 'PR', 'DQ'}.")

# I/O and debugging
parser.add_argument('-plot_indicator', help="Plot error indicator to file.")
=======

# I/O and debugging
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
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
<<<<<<< HEAD
approach = args.approach
both = approach == 'dwr_both' or 'int' in approach or 'avg' in approach
adjoint = 'adjoint' in approach or both
=======
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
kwargs = {
    'level': int(args.level or 0),

    # QoI
    'aligned': not bool(args.offset or False),

    # Mesh adaptation
<<<<<<< HEAD
    'approach': approach,
    'target': float(args.target or 4000.0),
=======
    'approach': args.approach or 'dwr',
    'target': float(args.target or 1.0e+03),
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
    'norm_order': p,
    'convergence_rate': alpha,
    'min_adapt': int(args.min_adapt or 3),
    'max_adapt': int(args.max_adapt or 35),
<<<<<<< HEAD
    'enrichment_method': args.enrichment_method or ('GE_p' if adjoint else 'DQ'),
=======
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a

    # I/O and debugging
    'plot_pvd': True,
    'debug': bool(args.debug or 0),
}
op = PointDischarge2dOptions(**kwargs)
op.tracer_family = family
stabilisation = args.stabilisation or 'supg'
<<<<<<< HEAD
op.stabilisation_tracer = None if stabilisation == 'none' else stabilisation
op.anisotropic_stabilisation = False if args.anisotropic_stabilisation == '0' else True
op.use_automatic_sipg_parameter = op.tracer_family == 'dg'
op.di = create_directory(os.path.join(op.di, op.stabilisation_tracer or family, op.enrichment_method))
op.normalisation = args.normalisation or 'complexity'  # FIXME: error
op.print_debug(op)

# --- Solve

tp = problem(op)
tp.run()

if bool(args.plot_indicator or False):
    indicator_file = File(os.path.join(op.di, "indicator.pvd"))
    indicator_file.write(tp.indicator[op.enrichment_method])

=======
op.stabilisation = None if stabilisation == 'none' else stabilisation
anisotropic_stabilisation = args.anisotropic_stabilisation
op.anisotropic_stabilisation = False if anisotropic_stabilisation == 0 else True
op.use_automatic_sipg_parameter = op.tracer_family == 'dg'
op.di = os.path.join(op.di, args.stabilisation or family)
op.normalisation = args.normalisation or 'complexity'  # FIXME: error
op.print_debug(op)


# --- Solve

tp = AdaptiveSteadyProblem(op)
tp.run()

>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
# Export to HDF5
save_mesh(tp.mesh, "mesh", fpath=op.di)
