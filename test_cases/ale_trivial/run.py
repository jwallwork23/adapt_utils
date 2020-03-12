from firedrake import norm, errornorm
from thetis import print_output

import numpy as np
import argparse

from adapt_utils.test_cases.ale_trivial.options import ALEAdvectionOptions
from adapt_utils.tracer.solver2d import UnsteadyTracerProblem2d


# Collect user specified arguments
parser = argparse.ArgumentParser()
parser.add_argument("interpretation", help="Choose from {'eulerian', 'lagrangian'}.")
parser.add_argument("-family", help="Choose from {'cg', 'dg'}.")
args = parser.parse_args()
approach = 'fixed_mesh' if args.interpretation == 'eulerian' else 'ale'
family = args.family or 'cg'

# Setup
op = ALEAdvectionOptions(approach=approach, prescribed_velocity='fluid', family=family)
tp = UnsteadyTracerProblem2d(op)

# Get initial solution and coordinates
init_sol = tp.solution.copy(deepcopy=True)
init_norm = norm(init_sol)
init_coords = tp.mesh.coordinates.dat.data.copy()

# Solve PDE
if approach == 'fixed_mesh':
    tp.solve()
elif approach == 'ale':
    tp.solve_ale()
else:
    raise NotImplementedError
final_sol = tp.solution
final_norm = norm(final_sol)
final_coords = tp.mesh.coordinates.dat.data

# Check final coords match initial coords
if approach == 'ale':
    final_coords[:] -= [10.0, 0.0]  # TODO: Implement periodicity
try:
    assert np.allclose(init_coords, final_coords)
except AssertionError:
    raise ValueError("Initial and final mesh coordinates do not match")

# Compute relative errors
print_output("Initial norm:        {:.4e}".format(init_norm))
print_output("Final norm:          {:.4e}".format(final_norm))
print_output("Relative difference: {:.2f}%".format(100*abs(1.0 - final_norm/init_norm)))
print_output("Relative error:      {:.2f}%".format(100*abs(errornorm(init_sol, final_sol)/init_norm)))
