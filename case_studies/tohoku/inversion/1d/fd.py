from thetis import *

import argparse
import numpy as np

from adapt_utils.case_studies.tohoku.options.radial_options import TohokuRadialBasisOptions
from adapt_utils.unsteady.solver import AdaptiveProblem


parser = argparse.ArgumentParser()
parser.add_argument("-level", help="Mesh resolution level")
parser.add_argument("-debug", help="Toggle debugging")
args = parser.parse_args()

# --- Setup

# Set parameters
level = int(args.level or 0)
kwargs = {
    'level': level,
    'save_timeseries': True,

    # Spatial discretisation
    # 'family': 'dg-cg',
    'family': 'cg-cg',
    # 'stabilisation': 'lax_friedrichs',
    'stabilisation': None,
    'use_automatic_sipg_parameter': False,  # the problem is inviscid

    # Adjoint
    'control_parameters': [10.0, ],
    'optimal_value': 5.0,
    'synthetic': True,
    # 'qoi_scaling': 1.0e-12,
    'qoi_scaling': 1.0,
    'nx': 1,
    'ny': 1,

    # Misc
    'plot_pvd': False,
    'debug': bool(args.debug or False),
}
nonlinear = False  # TODO
op = TohokuRadialBasisOptions(**kwargs)

# Solve the forward problem to get data with 'optimal' control parameter m = 5
op.control_parameters[0].assign(kwargs['optimal_value'])
swp = AdaptiveProblem(op, nonlinear=nonlinear, checkpointing=False)
swp.solve_forward()
for gauge in op.gauges:
    op.gauges[gauge]["data"] = op.gauges[gauge]["timeseries"]
op.save_timeseries = False

# Solve the forward problem with 'suboptimal' control parameter m = 10, checkpointing state
op.control_parameters[0].assign(kwargs['control_parameters'][0])
swp.solve_forward()
J = op.J
assert not np.allclose(J, 0.0)

# --- Finite differences

# Establish gradient using finite differences
epsilon = 1.0
converged = False
rtol = 1.0e-05
g_fd_ = None
while not converged:
    op.control_parameters[0].assign(kwargs['control_parameters'][0] + epsilon)
    swp.solve_forward(plot_pvd=False)
    J_step = op.J
    g_fd = (J_step - J)/epsilon
    print_output("J(epsilon=0) = {:.8e}  J(epsilon={:.1e}) = {:.8e}".format(J, epsilon, J_step))
    if g_fd_ is not None:
        print_output("gradient = {:.8e}  difference = {:.8e}".format(g_fd, abs(g_fd - g_fd_)))
        if abs(g_fd - g_fd_) < rtol*J:
            converged = True
        elif epsilon < 1.0e-10:
            raise ConvergenceError
    epsilon *= 0.5
    g_fd_ = g_fd

# Logging
logstr = "elements: {:d}\n".format(swp.meshes[0].num_cells())
logstr += "finite difference gradient (rtol={:.1e}): {:.4e}\n".format(rtol, g_fd)
print_output(logstr)
