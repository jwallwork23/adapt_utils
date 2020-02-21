import argparse

from adapt_utils.test_cases.steady_turbine.options import *
from adapt_utils.swe.turbine.solver import *


parser = argparse.ArgumentParser()
parser.add_argument('-approach', help="Mesh adaptation approach.")
parser.add_argument('-target', help="Target complexity for adaptive approaches.")
parser.add_argument('-level', help="Number of uniform refinements to apply to the initial mesh.")
parser.add_argument('-offset', help="""
    Number of turbine diameters by which to offset turbines in y-direction.
    'Aligned' configuration given by offset=0, 'Offset' configuration given by offset=1.""")
args = parser.parse_args()

kwargs = {
    'approach': args.approach or 'fixed_mesh',
    'offset': int(args.offset or 0),
    'plot_pvd': True,
    'debug': True,

    # Problem parameters
    'inflow_velocity': [6.0, 0.0],
    'base_viscosity': 0.5,

    # Adaptation parameters
    'target': float(args.target or 400.0),
    'adapt_field': 'all_int',
    'normalisation': 'complexity',
    'convergence_rate': 1,
    'norm_order': None,
    'h_max': 500.0,

}
level = int(args.level or 4)
if kwargs['approach'] != 'fixed_mesh':
    level = 1
tp = SteadyTurbineProblem(Steady2TurbineOptions(**kwargs), discrete_adjoint=True, levels=level)
if tp.op.approach == 'fixed_mesh':  # TODO: Use 'uniform' approach
    for i in range(level):
        tp = tp.tp_enriched
    tp.solve()
    tp.op.print_debug("QoI: {:.4e}kW".format(tp.quantity_of_interest()/1000))
else:
    tp.adaptation_loop()
