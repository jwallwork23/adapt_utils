from thetis import *
from adapt_utils.test_cases.steady_turbine.options import *
from adapt_utils.swe.turbine.solver import SteadyTurbineProblem

import argparse

parser = argparse.ArgumentParser()  # TODO: Run all using outer adaptation loop; remove argparse
parser.add_argument('-approach', help="Mesh adaptation strategy")
parser.add_argument('-target', help="Scaling parameter for metric")
parser.add_argument('-offset', help="Toggle offset or aligned turbine configurations")
args = parser.parse_args()

op2.init(log_level=INFO)

# Problem setup
offset = bool(args.offset or False)
label = 'xcoarse_offset' if offset else 'xcoarse'
sol = None

# Adaptation parameters
approach = args.approach or 'carpio'
op = Steady2TurbineOffsetOptions(approach) if offset else Steady2TurbineOptions(approach)
op.target = float(args.target or 1000)
op.adapt_field = 'all_int'
op.normalisation = 'complexity'
op.convergence_rate = 1
op.norm_order = None
op.h_max = 500.0

# Termination criteria
op.num_adapt = 35  # Maximum iterations
op.set_all_rtols(0.002)

# Set initial mesh hierarchy
tp = SteadyTurbineProblem(op, discrete_adjoint=True, prev_solution=sol, levels=1)
tp.adaptation_loop()

# TODO: Format output as table
# TODO: Plot QoI convergence with nice formatting. (Replaces jupyter notebook.)
