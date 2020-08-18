from thetis import *
from firedrake_adjoint import *

import numpy as np
import os
import scipy

from adapt_utils.case_studies.tohoku.options.okada_options import TohokuOkadaOptions
from adapt_utils.case_studies.tohoku.options.box_options import TohokuBoxBasisOptions
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.solver_adjoint import AdaptiveDiscreteAdjointProblem
from adapt_utils.norms import vecnorm


class DiscreteAdjointTsunamiProblem(AdaptiveDiscreteAdjointProblem):
    """The subclass exists to pass the QoI as required."""
    def quantity_of_interest(self):
        return self.op.J


# TODO: argparse

level = 0
real_data = False
optimise = True
test_gradient = True
N = 51
kwargs = {

    # Resolution
    "level": level,

    # Model
    "family": "dg-cg",
    "stabilisation": None,

    # Inversion
    "synthetic": not real_data,

    # I/O and debugging
    "save_timeseries": True,
    "plot_pvd": False,
    "debug": False,
}
nonlinear = False

with stop_annotating():

    # Create Okada parameter class and set the default initial conditionm
    kwargs_okada = {"okada_grid_resolution": N}
    kwargs_okada.update(kwargs)
    op_okada = TohokuOkadaOptions(**kwargs_okada)
    swp = AdaptiveProblem(op_okada, nonlinear=nonlinear)
    f = op_okada.set_initial_condition(swp)

    # Create BoxBasis parameter class and establish the basis functions
    op = TohokuBoxBasisOptions(**kwargs)
    swp = AdaptiveProblem(op, nonlinear=nonlinear)
    op.get_basis_functions()

    # Construct 'optimal' control vector by projection
    for i, phi in op.basis_functions:
        op.control_parameters[i] = assemble(phi*f*dx)/assemble(phi*dx)
    swp.set_initial_condition()

import matplotlib.pyplot as plt
fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
f_box = project(swp.fwd_solutions[0].split()[1], swp.P1[0])
fig.colorbar(tricontourf(f, axes=axes[0], levels=50, cmap='coolwarm'), ax=axes[0])
axes[0].set_title("Okada basis")
fig.colorbar(tricontourf(f_box, axes=axes[1], levels=50, cmap='coolwarm'), ax=axes[1])
axes[1].set_title("Box basis")
plt.show()
exit(0)

# Setup output directories
dirname = os.path.dirname(__file__)
di = create_directory(os.path.join(dirname, 'outputs', 'realistic' if real_data else 'synthetic'))
op.di = create_directory(os.path.join(di, 'discrete'))

# Synthetic run to get timeseries data, with the default control values as the "optimal" case
if not real_data:
    with stop_annotating():
        swp = AdaptiveProblem(op, nonlinear=nonlinear)
        swp.solve_forward()
        for gauge in op.gauges:
            op.gauges[gauge]["data"] = op.gauges[gauge]["timeseries"]

# Set zero initial guess for the optimisation
kwargs['control_parameters'] = op.control_parameters
m_init = []
for control in op.active_controls:
    kwargs['control_parameters'][control] = np.zeros(np.shape(op.control_parameters[control]))
    m_init.append(kwargs['control_parameters'][control])
m_init = np.array(m_init)
J_progress = []

# Create discrete adjoint solver object for the optimisation
swp = DiscreteAdjointTsunamiProblem(op, nonlinear=nonlinear)

# TODO: Inversion
