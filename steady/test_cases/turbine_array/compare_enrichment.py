import argparse
import os
import pickle
from time import perf_counter

from adapt_utils.io import create_directory
from adapt_utils.swe.turbine.solver import AdaptiveSteadyTurbineProblem
from adapt_utils.steady.test_cases.turbine_array.options import TurbineArrayOptions


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument('-solve_enriched_forward', help="Prolong or solve forward problem?")
parser.add_argument('-offset', help="Toggle between aligned or offset region of interest.")
parser.add_argument('-debug', help="Toggle debugging mode.")
args = parser.parse_args()


# --- Set parameters

offset = bool(args.offset or False)
alignment = 'offset' if offset else 'aligned'
solve_forward = bool(args.solve_enriched_forward or False)
nonlinear_method = 'solve' if solve_forward else 'prolong'
kwargs = {
    'approach': 'dwr',
    'solve_enriched_forward': solve_forward,
    'offset': offset,
    'plot_pvd': False,
    'debug': bool(args.debug or 0),
}


# --- Loop over enrichment methods

levels = 3
methods = ('GE_hp', 'GE_h', 'GE_p')
out = {method: {'time': [], 'num_cells': [], 'dofs': []} for method in methods}
di = create_directory('outputs/dwr/enrichment')
fname = os.path.join(di, '{:s}_{:s}.p'.format(alignment, nonlinear_method))
for method in methods:
    print(method)
    for level in range(levels):
        op = TurbineArrayOptions(level=level, **kwargs)
        op.enrichment_method = method

        # Solve problems in base space
        tp = AdaptiveSteadyTurbineProblem(op, print_progress=False, discrete_adjoint=True)
        out[method]['num_cells'].append(tp.mesh.num_cells())
        out[method]['dofs'].append(sum(tp.V[0].dof_count))
        tp.solve_forward()
        tp.solve_adjoint()

        # Indicate error
        timestamp = perf_counter()
        tp.indicate_error('shallow_water')
        out[method]['time'].append(perf_counter() - timestamp)
pickle.dump(out, open(fname, 'wb'))
for method in out.keys():
    print(out, out[method])
