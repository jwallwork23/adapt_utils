from thetis import *

import argparse
import h5py
import os
from time import perf_counter

from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.test_cases.bubble_shear.options import BubbleOptions


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("-family", help="Choose finite element from 'cg' and 'dg'")
<<<<<<< HEAD
parser.add_argument("-conservative", help="Toggle conservative tracer equation")
parser.add_argument("-limiters", help="Toggle limiters for tracer equation")
parser.add_argument("-stabilisation", help="Stabilisation method")
parser.add_argument("-anisotropic_stabilisation", help="Use anisotropic stabilisation.")
=======
parser.add_argument("-limiters", help="Toggle limiters for tracer equation")
parser.add_argument("-stabilisation", help="Stabilisation method")
>>>>>>> origin/master

parser.add_argument("-debug", help="Toggle debugging mode.")
args = parser.parse_args()


# --- Set parameters

kwargs = {
    'tracer_family': args.family or 'cg',
    'stabilisation_tracer': args.stabilisation or 'supg',
<<<<<<< HEAD
    'anisotropic_stabilisation': False if args.anisotropic_stabilisation == "0" else True,
    'use_automatic_sipg_parameter': False,  # We have an inviscid problem
    'use_limiter_for_tracers': bool(args.limiters or False),
    'use_tracer_conservative_form': bool(args.conservative or False),  # FIXME?
=======
    'use_limiter_for_tracers': bool(args.limiters or False),
>>>>>>> origin/master
    'debug': bool(args.debug or False),
}
l2_error = []
cons_error = []
times = []
num_cells = []
dofs = []
for level in range(4):

    # Setup
    op = BubbleOptions(approach='fixed_mesh', n=level)
    op.update(kwargs)
    op.dt_per_export = 2**level
    tp = AdaptiveProblem(op)
    dofs.append(tp.Q[0].dof_count)
    num_cells.append(tp.mesh.num_cells())
    tp.set_initial_condition()
    init_l1_norm = norm(tp.fwd_solutions_tracer[0], norm_type='L1')
    init_l2_norm = norm(tp.fwd_solutions_tracer[0], norm_type='L2')
    init_sol = tp.fwd_solutions_tracer[0].copy(deepcopy=True)

    # Solve forward problem
    cpu_timestamp = perf_counter()
    tp.solve_forward()
    times.append(perf_counter() - cpu_timestamp)

    # Compare initial and final tracer concentrations
    final_l1_norm = norm(tp.fwd_solutions_tracer[0], norm_type='L1')
    final_l2_norm = norm(tp.fwd_solutions_tracer[0], norm_type='L2')
    final_sol = tp.fwd_solutions_tracer[0].copy(deepcopy=True)
    abs_l2_error = errornorm(init_sol, final_sol, norm_type='L2')
    cons_error.append(100*abs(init_l1_norm-final_l1_norm)/init_l1_norm)
    l2_error.append(100*abs_l2_error/init_l2_norm)

# Save to HDF5
with h5py.File(os.path.join(op.di, 'convergence.h5'), 'w') as outfile:
    outfile.create_dataset('elements', data=num_cells)
    outfile.create_dataset('dofs', data=dofs)
    outfile.create_dataset('time', data=times)
    outfile.create_dataset('l2_error', data=l2_error)
    outfile.create_dataset('cons_error', data=cons_error)
