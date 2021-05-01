"""
Monge-Ampere mesh movement applied to the test case created in [Balzano].

[Balzano] A. Balzano, "Evaluation of methods for numerical simulation of wetting and drying in
shallow water flow models." Coastal Engineering 34.1-2 (1998): 83-107.
"""
from firedrake_adjoint import *
from thetis import *

#import argparse
import numpy as np
import os

from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.test_cases.balzano.options import BalzanoOptions

fric_coef = Constant(0.025)

#parser = argparse.ArgumentParser()
#parser.add_argument("-bathymetry_type", help="""
#    Choose bathymetry type from {1, 2, 3}. Option 1 corresponds to a linear bathymetry, whereas 2
#    and 3 have kinks. See [Balzano] for details.
#    """)
#parser.add_argument("-stabilisation", help="Stabilisation method")
#parser.add_argument("-family", help="Choose finite element from 'cg-cg', 'dg-cg' and 'dg-dg'")
#parser.add_argument("-debug", help="Toggle debugging")
#args = parser.parse_args()

stabilisation = 'lax_friedrichs'

kwargs = {
    'approach': 'monge_ampere',
    'n': 2,
    'num_hours': 1,
    'fric_coeff': fric_coef,

    # Geometry
    'bathymetry_type': int(1),

    # Mesh movement
    'nonlinear_method': 'relaxation',
    'r_adapt_rtol': 1.0e-2,

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
assert op.num_meshes == 1
swp = AdaptiveProblem(op)
# swp.shallow_water_options[0]['mesh_velocity'] = swp.mesh_velocities[0]
swp.shallow_water_options[0]['mesh_velocity'] = None

alpha = Constant(0.0)  # size of the dense region surrounding the coast
beta = 10.0  # level of refinement at coast


def wet_dry_interface_monitor(mesh, alpha=alpha, x = None):
    """
    Monitor function focused around the wet-dry interface.

    NOTES:
      * The monitor function is defined on the *computational* mesh.
      * For the first mesh movement iteration, the mesh coordinates coincide.
    """
    eta_old = swp.fwd_solutions[0].split()[1]
    b_old = swp.bathymetry[0]
    eta = Function(FunctionSpace(mesh, eta_old.ufl_element()))
    b = Function(FunctionSpace(mesh, b_old.ufl_element()))
    eta.project(eta_old)
    b.project(b_old)
    return 1.0 + alpha*pow(cosh(beta*(eta + b)), -2)


swp.set_monitor_functions(wet_dry_interface_monitor)
swp.solve_forward()

J = assemble(swp.fwd_solutions[0].split()[1]*dx)

rf = ReducedFunctional(J, Control(fric_coef))
print(rf(Constant(0.025)))

import ipdb; ipdb.set_trace()
