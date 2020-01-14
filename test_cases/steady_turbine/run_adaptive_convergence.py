from thetis import *
from adapt_utils.test_cases.steady_turbine.options import *
from adapt_utils.turbine.solver import SteadyTurbineProblem
from adapt_utils.adapt.metric import *
from adapt_utils.adapt.p0_metric import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-approach', help="Mesh adaptation strategy")
parser.add_argument('-target', help="Scaling parameter for metric")
parser.add_argument('-offset', help="Toggle offset or aligned turbine configurations")
parser.add_argument('-initial_mesh', help="Resolution of initial mesh")
parser.add_argument('-adapt_field', help="Field(s) to compute Hessian w.r.t.")
parser.add_argument('-rtol', help="Relative tolerance for adaptation algorithm termination")
args = parser.parse_args()

op2.init(log_level=INFO)

# Problem setup
offset = bool(args.offset or False)
level = args.initial_mesh or 'xcoarse'
label = '_'.join([level, 'offset']) if offset else level
sol = None

# Adaptation parameters
approach = args.approach or 'carpio'
op = Steady2TurbineOffsetOptions(approach) if offset else Steady2TurbineOptions(approach)
op.target = float(args.target or 1000)
op.adapt_field = args.adapt_field or 'all_int'
op.normalisation = 'complexity'
op.convergence_rate = 1
op.norm_order = None
op.h_max = 500.0

op.num_adapt = 1

# Termination criteria
op.num_adapt = 35  # Maximum iterations
rtol = float(args.rtol or 0.002)
qoi_rtol = rtol
element_rtol = rtol
estimator_rtol = rtol

# Set initial mesh hierarchy
tp = SteadyTurbineProblem(op, discrete_adjoint=True, prev_solution=sol, levels=1)
tp.adaptation_loop()

# TODO: Outer adaptation loop
# TODO: Format output as table
# TODO: Plotting
