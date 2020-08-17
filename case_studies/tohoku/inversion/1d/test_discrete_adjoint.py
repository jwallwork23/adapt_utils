from thetis import *
from firedrake_adjoint import *

import argparse

from adapt_utils.case_studies.tohoku.options.gaussian_options import TohokuGaussianBasisOptions
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.solver_adjoint import AdaptiveDiscreteAdjointProblem


parser = argparse.ArgumentParser()
parser.add_argument("-level", help="Mesh resolution level")
parser.add_argument("-family", help="Finite element pair")
parser.add_argument("-stabilisation", help="Stabilisation approach")
parser.add_argument("-nonlinear", help="Toggle nonlinear equations")
parser.add_argument("-initial_guess", help="Initial guess for control parameter")
parser.add_argument("-optimal_control", help="Artificially choose an optimum to invert for")
parser.add_argument("-debug", help="Toggle debugging")
args = parser.parse_args()

# Set parameters
level = int(args.level or 0)
kwargs = {
    'level': level,
    'save_timeseries': True,

    # Spatial discretisation
    'family': args.family or 'dg-cg',
    'stabilisation': args.stabilisation,
    'use_automatic_sipg_parameter': False,  # the problem is inviscid

    # Optimisation
    'control_parameters': [float(args.initial_guess or 3.2760), ],
    'synthetic': True,
    'qoi_scaling': 1.0,

    # Misc
    'plot_pvd': False,
    'debug': bool(args.debug or False),
}
nonlinear = bool(args.nonlinear or False)

# Toggle smoothed or discrete timeseries
timeseries_type = "timeseries"
use_smoothed_timeseries = True
if use_smoothed_timeseries:
    timeseries_type = "_".join([timeseries_type, "smooth"])

# Artifical run
with stop_annotating():
    op = TohokuGaussianBasisOptions(**kwargs)
    op.control_parameters[0].assign(float(args.optimal_control or 5.0))
    swp = AdaptiveProblem(op, nonlinear=nonlinear)
    swp.solve_forward()
    for gauge in op.gauges:
        op.gauges[gauge]["data"] = op.gauges[gauge][timeseries_type]


class DiscreteAdjointTsunamiProblem(AdaptiveDiscreteAdjointProblem):
    """The subclass exists to pass the QoI as required."""
    def quantity_of_interest(self):
        return self.op.J


# Run with 'suboptimal' control
kwargs['plot_pvd'] = True
op_opt = TohokuGaussianBasisOptions(fpath='discrete/' + kwargs['family'], **kwargs)
gauges = list(op_opt.gauges.keys())
for gauge in gauges:
    op_opt.gauges[gauge]["data"] = op.gauges[gauge]["data"]
swp = DiscreteAdjointTsunamiProblem(op_opt, nonlinear=nonlinear)
swp.solve_forward()
print_output("QoI: {:.4e}".format(op_opt.J))

# Solve adjoint and plot
swp.compute_gradient(Control(op_opt.control_parameters[0]))
swp.get_solve_blocks()
swp.save_adjoint_trajectory()
