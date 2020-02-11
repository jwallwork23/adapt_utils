from thetis import *

import pylab as plt

from adapt_utils.test_cases.trench_test.options import TrenchOptions
from adapt_utils.swe.tsunami.solver import TsunamiProblem

op = TrenchOptions(approach='monge_ampere',
                    plot_timeseries=False,
                    plot_pvd=True,
                    debug=False,
                    nonlinear_method='relaxation',
                    # nonlinear_method='quasi_newton',  # FIXME
                    num_adapt=1,
                    qoi_mode='inundation_volume',
                    friction = 'nikuradse',
                    nx=1,
                    ny = 1,
                    r_adapt_rtol=1.0e-3)

tp = TsunamiProblem(op, levels=0)
tp.setup_solver()


def gradient_interface_monitor(mesh, alpha=1):
    """
    Monitor function focused around the steep_gradient (budd acta numerica)

    NOTE: Defined on the *computational* mesh.

    :kwarg alpha: controls the size of the dense region surrounding the coast.
    """
    P1 = FunctionSpace(mesh, "CG", 1)
    
    eta = tp.solution.split()[1]
    
    b = tp.fields['bathymetry']
    current_mesh = eta.function_space().mesh()
    P1_current = FunctionSpace(current_mesh, "CG", 1)
    bath_dx = interpolate(b.dx(0), P1_current)
    bath_dy = interpolate(b.dx(1), P1_current)
    norm = interpolate(pow(bath_dx, 2) + pow(bath_dy, 2), P1_current)
    norm_proj = project(norm, P1)


    return sqrt(1.0 + alpha*norm_proj)

tp.monitor_function = gradient_interface_monitor
tp.solve(uses_adjoint=False)
