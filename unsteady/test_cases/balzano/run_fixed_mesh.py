"""
Test case created in [Balzano].

[Balzano] A. Balzano, "Evaluation of methods for numerical simulation of wetting and drying in
shallow water flow models." Coastal Engineering 34.1-2 (1998): 83-107.
"""
from firedrake_adjoint import *
from firedrake import *
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.test_cases.balzano.options import BalzanoOptions

#import argparse
import os

fric_coef = Constant(0.025)
#parser = argparse.ArgumentParser()
#parser.add_argument("-bathymetry_type", help="""
#    Choose bathymetry type from {1, 2, 3}. Option 1 corresponds to a linear bathymetry, whereas 2
#    and 3have kinks. See [Balzano] for details.
#    """)
#parser.add_argument("-stabilisation", help="Stabilisation method")
#parser.add_argument("-family", help="Choose finite element from 'cg-cg', 'dg-cg' and 'dg-dg'")
#parser.add_argument("-debug", help="Toggle debugging")
#args = parser.parse_args()

stabilisation = 'lax_friedrichs'

kwargs = {
    'fric_coeff': fric_coef,
    'approach': 'fixed_mesh',
    'n': 2,
    # Geometry
    'bathymetry_type': int(1),

    # Spatial discretisation
    'family': 'dg-cg',
    'stabilisation': stabilisation,
    'use_automatic_sipg_parameter': True,

    # Misc
    'plot_pvd': True,
    'debug': bool(False),
}
if os.getenv('REGRESSION_TEST') is not None:
    kwargs['num_hours'] = 6

op = BalzanoOptions(**kwargs)
swp = AdaptiveProblem(op)
swp.solve_forward()
J = assemble(swp.fwd_solutions[0].split()[1]*dx)
rf = ReducedFunctional(J, Control(fric_coef))
J_0 = rf(Constant(0.025))
import ipdb; ipdb.set_trace()
