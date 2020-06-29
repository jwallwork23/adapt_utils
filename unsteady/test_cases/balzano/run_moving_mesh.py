from thetis import *

import numpy as np

from adapt_utils.unsteady.test_cases.balzano.options import BalzanoOptions
from adapt_utils.unsteady.solver import AdaptiveProblem


kwargs = {
    'approach': 'monge_ampere',
    'n': 2,
    'plot_pvd': True,
    'debug': True,

    'nonlinear_method': 'relaxation',
    'r_adapt_rtol': 1.0e-2,

    'family': 'dg-cg',
    'stabilisation': 'lax_friedrichs',
    'use_automatic_sipg_parameter': True,
}

op = BalzanoOptions(**kwargs)
# op.solver_parameters['shallow_water'].update({
#     'ksp_monitor': None,
# })
assert op.num_meshes == 1
swp = AdaptiveProblem(op)
# swp.shallow_water_options[0]['mesh_velocity'] = swp.mesh_velocities[0]
swp.shallow_water_options[0]['mesh_velocity'] = None

alpha = 1.0  # size of the dense region surrounding the coast
beta = 1.0   # level of refinement at coast


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
