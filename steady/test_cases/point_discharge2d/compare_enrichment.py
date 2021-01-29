import argparse
import os
import pickle
from time import perf_counter

from adapt_utils.io import create_directory
from adapt_utils.steady.test_cases.point_discharge2d.options import PointDischarge2dOptions


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument('-offset', help="Toggle between aligned or offset region of interest.")
parser.add_argument('-debug', help="Toggle debugging mode.")
parser.add_argument('-discrete_adjoint', help="Use discrete adjoint method.")
args = parser.parse_args()


# --- Set parameters

offset = bool(args.offset or False)
kwargs = {
    'approach': 'dwr',
    'aligned': not offset,
    'plot_pvd': False,
    'debug': bool(args.debug or 0),
}
analytical_qoi = 0.06956861886754047 if offset else 0.1633864523747167

discrete_adjoint = bool(args.discrete_adjoint or False)
# discrete_adjoint = False if args.discrete_adjoint == "0" else True
if discrete_adjoint:
    from adapt_utils.steady.solver_adjoint import AdaptiveDiscreteAdjointSteadyProblem
    problem = AdaptiveDiscreteAdjointSteadyProblem
else:
    from adapt_utils.steady.solver import AdaptiveSteadyProblem
    problem = AdaptiveSteadyProblem


# --- Loop over enrichment methods

levels = 6
methods = ('GE_hp', 'GE_h', 'GE_p', 'DQ')
out = {method: {'effectivity': [], 'time': [], 'num_cells': [], 'dofs': []} for method in methods}
di = create_directory('outputs/dwr/enrichment')
fname = os.path.join(di, '{:s}.p'.format('offset' if offset else 'aligned'))
for method in methods:
    print(method)
    for level in range(levels):
        op = PointDischarge2dOptions(level=level, **kwargs)
        op.tracer_family = 'cg'
        op.stabilisation_tracer = 'supg'
        op.anisotropic_stabilisation = True
        op.use_automatic_sipg_parameter = False
        op.normalisation = 'complexity'
        op.enrichment_method = method

        tp = problem(op, print_progress=False)
        out[method]['num_cells'].append(tp.mesh.num_cells())
        out[method]['dofs'].append(tp.mesh.num_vertices())
        tp.solve_forward()
        tp.solve_adjoint()

        timestamp = perf_counter()
        tp.indicate_error('tracer')
        out[method]['time'].append(perf_counter() - timestamp)

        # Calculate effectivity
        estimator = tp.indicator[op.enrichment_method].vector().gather().sum()
        out[method]['effectivity'].append(estimator/analytical_qoi)
pickle.dump(out, open(fname, 'wb'))
for method in out.keys():
    print(out, out[method])
