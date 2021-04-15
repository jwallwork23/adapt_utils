"""
Monge-Ampere mesh movement applied to the test case created in [Balzano].

[Balzano] A. Balzano, "Evaluation of methods for numerical simulation of wetting and drying in
shallow water flow models." Coastal Engineering 34.1-2 (1998): 83-107.
"""
from thetis import *

import argparse
import numpy as np
import os

from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.test_cases.balzano.options import BalzanoOptions


parser = argparse.ArgumentParser()
parser.add_argument("-bathymetry_type", help="""
    Choose bathymetry type from {1, 2, 3}. Option 1 corresponds to a linear bathymetry, whereas 2
    and 3 have kinks. See [Balzano] for details.
    """)
parser.add_argument("-stabilisation", help="Stabilisation method")
parser.add_argument("-family", help="Choose finite element from 'cg-cg', 'dg-cg' and 'dg-dg'")
parser.add_argument("-debug", help="Toggle debugging")
args = parser.parse_args()

stabilisation = args.stabilisation or 'lax_friedrichs'

kwargs = {
    'approach': 'monge_ampere',
    'n': 2,

    # Geometry
    'bathymetry_type': int(args.bathymetry_type or 1),

    # Mesh movement
    'nonlinear_method': 'relaxation',
    'r_adapt_rtol': 1.0e-2,

    # Spatial discretisation
    'family': args.family or 'dg-cg',
    'stabilisation': stabilisation,
    'use_automatic_sipg_parameter': True,

    # Misc
    'plot_pvd': True,
    'debug': bool(args.debug or False),
}
if os.getenv('REGRESSION_TEST') is not None:
    kwargs['num_hours'] = 6

op = BalzanoOptions(**kwargs)
assert op.num_meshes == 1
swp = AdaptiveProblem(op)

alpha = 1.0  # size of the dense region surrounding the coast
beta = 10.0  # level of refinement at coast


def wet_dry_interface_monitor(mesh):
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
    same_mesh = np.allclose(mesh.coordinates.dat.data, swp.meshes[0].coordinates.dat.data)
    if same_mesh:
        eta.dat.data[:] = eta_old.dat.data
        b.dat.data[:] = b_old.dat.data
    else:
        eta.project(eta_old)
        b.project(b_old)
    return 1.0 + alpha*pow(cosh(beta*(eta + b)), -2)


swp.set_monitor_functions(wet_dry_interface_monitor)
swp.solve_forward()
