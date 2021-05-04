from firedrake_adjoint import *
from thetis import *

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime

diff_flag = True

def eval_callback(functional_value, value):
    f.write(str(value.dat.data[:]))
    f.write('   ')
    f.write(str(functional_value))
    f.write('\n')
    print(value.dat.data[:])
    print(functional_value)

lx = 16
ly = 1.1
nx = 0.5 #lx*4
ny = 1 #5
mesh2d = RectangleMesh(np.int(16*5*nx), 5*ny, 16, 1.1)

x, y = SpatialCoordinate(mesh2d)

V = FunctionSpace(mesh2d, "CG", 1)

initialdepth = Constant(0.397)
depth_riv = Constant(initialdepth - 0.397)
depth_trench = Constant(depth_riv - 0.15)
depth_diff = depth_trench - depth_riv

trench = conditional(le(x, 5), depth_riv, conditional(le(x, 6.5), (1/1.5)*depth_diff*(x-6.5) + depth_trench,
                         conditional(le(x, 9.5), depth_trench, conditional(le(x, 11), -(1/1.5)*depth_diff*(x-11) + depth_riv, depth_riv))))

def forward(bathymetry_2d, ks, average_size, rhos, diffusivity, viscosity, sed_rate = None):

    # define function spaces
    V = FunctionSpace(mesh2d, "CG", 1)
    DG_2d = FunctionSpace(mesh2d, "DG", 1)
    vector_dg = VectorFunctionSpace(mesh2d, "DG", 1)
    R_1d = FunctionSpace(mesh2d, 'R', 0)

    # choose directory to output results
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    outputdir = 'outputs'+ st

    print_output('Exporting to '+outputdir)

    morfac = 100
    dt = 0.25
    end_time = 2.5*3600

    # initialise velocity and elevation
    chk = DumbCheckpoint("hydrodynamics_trench_0.5/elevation", mode=FILE_READ)
    elev = Function(DG_2d, name="elevation")
    chk.load(elev)
    chk.close()

    chk = DumbCheckpoint('hydrodynamics_trench_0.5/velocity', mode=FILE_READ)
    uv = Function(vector_dg, name="velocity")
    chk.load(uv)
    chk.close()

    # set up solver
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    options = solver_obj.options

    options.sediment_model_options.solve_suspended_sediment = True
    options.sediment_model_options.use_bedload = True
    options.sediment_model_options.solve_exner = True

    options.sediment_model_options.average_sediment_size = average_size
    options.sediment_model_options.bed_reference_height = Function(R_1d).assign(ks)
    options.sediment_model_options.morphological_acceleration_factor = Constant(morfac)
    options.sediment_model_options.sediment_density = rhos
    options.sediment_model_options.morphological_viscosity = Function(R_1d).assign(viscosity)

    options.simulation_end_time = end_time/morfac
    options.simulation_export_time = options.simulation_end_time/45

    options.output_directory = outputdir
    options.check_volume_conservation_2d = True

    options.fields_to_export = ['sediment_2d', 'uv_2d', 'elev_2d', 'bathymetry_2d']  # note exporting bathymetry must be done through export func
    options.sediment_model_options.check_sediment_conservation = True

    # using nikuradse friction
    options.nikuradse_bed_roughness = Function(R_1d).assign(3*average_size)

    # set horizontal diffusivity parameter
    options.horizontal_diffusivity = diffusivity
    options.horizontal_viscosity = Function(R_1d).assign(1e-6)

    # crank-nicholson used to integrate in time system of ODEs resulting from application of galerkin FEM
    options.timestepper_type = 'CrankNicolson'
    options.timestepper_options.implicitness_theta = 1.0
    options.norm_smoother = Constant(0.1)

    if not hasattr(options.timestepper_options, 'use_automatic_timestep'):
        options.timestep = dt

    # set boundary conditions

    left_bnd_id = 1
    right_bnd_id = 2

    swe_bnd = {}

    swe_bnd[left_bnd_id] = {'flux': Constant(-0.22)}
    swe_bnd[right_bnd_id] = {'elev': Constant(0.397)}

    solver_obj.bnd_functions['shallow_water'] = swe_bnd

    if sed_rate is not None:
        solver_obj.bnd_functions['sediment'] = {
            left_bnd_id: {'flux': Constant(-0.22), 'value': sed_rate},
            right_bnd_id: {'elev': Constant(0.397)}}

        # set initial conditions
        solver_obj.assign_initial_conditions(uv=uv, elev=elev, sediment=sed_rate)
    else:
        solver_obj.bnd_functions['sediment'] = {
            left_bnd_id: {'flux': Constant(-0.22), 'equilibrium': None},
            right_bnd_id: {'elev': Constant(0.397)}}

        # set initial conditions
        solver_obj.assign_initial_conditions(uv=uv, elev=elev)

    # run model
    solver_obj.iterate()

    return solver_obj.fields.bathymetry_2d


test_derivative = False
taylor_test_flag = False
minimize_flag = True

ks = Constant(0.025)
average_size = Constant(160e-6)
rhos = Constant(2650)
diffusivity = Constant(0.15)
diffusivity_diff = Constant(0.15 + 10**(-4))

viscosity = Constant(1e-6)

#bath1 = Function(V).interpolate(-trench)

#old_bath = forward(bath1, ks2, average_size2, rhos2, diffusivity2, viscosity2, sed_rate2)

#tape = get_working_tape()
#tape.clear_tape()

bath2 = Function(V).interpolate(-trench)

new_bath = forward(bath2, ks, average_size, rhos, diffusivity, viscosity, sed_rate=None)

J = assemble(new_bath*dx)
rf = ReducedFunctional(J, Control(diffusivity))

print(J)
print(rf(Constant(0.15)))
import ipdb; ipdb.set_trace()

if taylor_test_flag == True:
    rf = ReducedFunctional(J, Control(diffusivity), eval_cb_post = eval_callback)
    h = Constant(5e-3)
    conv_rate = taylor_test(rf, diffusivity, h)

    if conv_rate > 1.9:
        print('*** test passed ***')
    else:
        print('*** ERROR: test failed ***')

    f.close()
    stop
