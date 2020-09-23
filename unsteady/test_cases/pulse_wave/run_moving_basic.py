from thetis import *

import datetime
import time

from adapt_utils.io import export_bathymetry
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.test_cases.beach_pulse_wave.options import BeachOptions

nx = 0.5
ny = 0.5

alpha = 1
beta = 1
gamma = 1

kappa = 100  # 12.5

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
outputdir = 'outputs' + st

kwargs = {
    'approach': 'monge_ampere',
    'nx': nx,
    'ny': ny,
    'plot_pvd': True,
    'output_dir': outputdir,
    'nonlinear_method': 'relaxation',
    'r_adapt_rtol': 1.0e-3,
    # Spatial discretisation
    'family': 'dg-dg',
    'stabilisation': None,
    'use_automatic_sipg_parameter': True,
    'friction': 'manning'
}

op = BeachOptions(**kwargs)
assert op.num_meshes == 1
swp = AdaptiveProblem(op)
swp.shallow_water_options[0]['mesh_velocity'] = None


def velocity_monitor(mesh, alpha=alpha, beta=beta, gamma=gamma, K=kappa):
    P1 = FunctionSpace(mesh, "CG", 1)
    b = swp.fwd_solutions_bathymetry[0]

    if b is not None:
        abs_hor_vel_norm = Function(b.function_space()).project(conditional(b > 0.0, Constant(1.0), Constant(0.0)))
    else:
        abs_hor_vel_norm = Function(swp.bathymetry[0].function_space()).project(conditional(swp.bathymetry[0] > 0.0, Constant(2.0), Constant(0.0)))
    comp_new = project(abs_hor_vel_norm, P1)
    mon_init = project(1.0 + alpha * comp_new, P1)
    return mon_init


swp.set_monitor_functions(velocity_monitor)

t1 = time.time()
swp.solve_forward()
t2 = time.time()

print(t2-t1)

fpath = "hydrodynamics_beach_bath_new_{:d}_test".format(int(nx*220))
export_bathymetry(swp.fwd_solutions_bathymetry[0], fpath, op=op)
