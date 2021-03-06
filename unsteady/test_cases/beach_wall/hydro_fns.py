#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:05:02 2019

@author: mc4117
"""

import thetis as th
import time
import datetime
import numpy as np


def hydrodynamics_only(boundary_conditions_fn, mesh2d, bathymetry_2d, uv_init,
                       elev_init, average_size, dt, t_end, wetting_and_drying=False,
                       wetting_alpha=0.1, friction='nikuradse', friction_coef=0,
                       fluc_bcs=False, viscosity=10**(-6), sponge_viscosity=th.Constant(1.0)):
    """
    Sets up a simulation with only hydrodynamics until a quasi-steady state when it can be used as an initial
    condition for the full morphological model. We update the bed friction at each time step.
    The actual run of the model are done outside the function

    Inputs:
    boundary_consditions_fn - function defining boundary conditions for problem
    mesh2d - define mesh working on
    bathymetry2d - define bathymetry of problem
    uv_init - initial velocity of problem
    elev_init - initial elevation of problem
    average_size - average sediment size
    dt - timestep
    t_end - end time
    wetting_and_drying - wetting and drying switch
    wetting_alpha - wetting and drying parameter
    friction - choice of friction formulation - nikuradse and manning
    friction_coef - value of friction coefficient used in manning
    viscosity - viscosity of hydrodynamics. The default value should be 10**(-6)

    Outputs:
    solver_obj - solver which we need to run to solve the problem
    update_forcings_hydrodynamics - function defining the updates to the model performed at each timestep
    outputdir - directory of outputs
    """
    def update_forcings_hydrodynamics(t_new):
        # update boundary conditions if have fluctuating conditions
        if fluc_bcs:
            in_fn, out_fn = boundary_conditions_fn(bathymetry_2d, t_new=t_new, state='update')
            for j in range(len(in_fn)):
                exec('constant_in' + str(j) + '.assign(' + str(in_fn[j]) + ')')

            for k in range(len(out_fn)):
                exec('constant_out' + str(k) + '.assign(' + str(out_fn[k]) + ')')
        # update bed friction
        if friction == 'nikuradse':
            uv1, elev1 = solver_obj.fields.solution_2d.split()

            if wetting_and_drying:
                wd_bath_displacement = solver_obj.depth.wd_bathymetry_displacement
                depth.project(elev1 + wd_bath_displacement(elev1) + bathymetry_2d)
            else:
                depth.interpolate(elev1 + bathymetry_2d)

        # calculate skin friction coefficient
        cfactor.interpolate(2*(0.4**2)/((th.ln(11.036*depth/(ksp)))**2))

    # choose directory to output results
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    outputdir = 'outputs' + st

    # export interval in seconds
    if t_end < 40:
        t_export = 0.1
    else:
        t_export = np.round(t_end/40, 0)

    th.print_output('Exporting to '+outputdir)

    # define function spaces
    V = th.FunctionSpace(mesh2d, 'CG', 1)
    P1_2d = th.FunctionSpace(mesh2d, 'DG', 1)

    # define parameters
    ksp = th.Constant(3*average_size)

    # define depth
    if wetting_and_drying:
        H = th.Function(V).project(elev_init + bathymetry_2d)
        depth = th.Function(V).project(H + (0.5 * (th.sqrt(H ** 2 + wetting_alpha ** 2) - H)))
    else:
        depth = th.Function(V).project(elev_init + bathymetry_2d)

    # define bed friction
    cfactor = th.Function(P1_2d).interpolate(2*(0.4**2)/((th.ln(11.036*depth/(ksp)))**2))

    # set up solver
    solver_obj = th.solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    options = solver_obj.options
    options.simulation_export_time = t_export
    options.simulation_end_time = t_end
    options.output_directory = outputdir

    options.check_volume_conservation_2d = True

    options.fields_to_export = ['uv_2d', 'elev_2d']
    options.solve_tracer = False
    options.use_lax_friedrichs_tracer = False
    if friction == 'nikuradse':
        options.quadratic_drag_coefficient = cfactor
    elif friction == 'manning':
        if friction_coef == 0:
            friction_coef = 0.02
        options.manning_drag_coefficient = th.Constant(friction_coef)
    else:
        print('Undefined friction')
    options.horizontal_viscosity = th.Function(P1_2d).interpolate(sponge_viscosity*th.Constant(viscosity))

    # crank-nicholson used to integrate in time system of ODEs resulting from application of galerkin FEM
    options.timestepper_type = 'CrankNicolson'
    options.timestepper_options.implicitness_theta = 1.0
    options.use_wetting_and_drying = wetting_and_drying
    options.wetting_and_drying_alpha = th.Function(V).interpolate(th.Constant(wetting_alpha))
    options.norm_smoother = th.Constant(1.0)

    if not hasattr(options.timestepper_options, 'use_automatic_timestep'):
        options.timestep = dt

    # set boundary conditions

    swe_bnd, left_bnd_id, right_bnd_id, in_constant, out_constant, left_string, right_string = boundary_conditions_fn(bathymetry_2d, flag='hydro')

    for j in range(len(in_constant)):
        exec('constant_in' + str(j) + ' = th.Constant(' + str(in_constant[j]) + ')', globals())

    str1 = '{'
    if len(left_string) > 0:
        for i in range(len(left_string)):
            str1 += "'" + str(left_string[i]) + "': constant_in" + str(i) + ","
        str1 = str1[0:len(str1)-1] + "}"
        exec('swe_bnd[left_bnd_id] = ' + str1)

    for k in range(len(out_constant)):
        exec('constant_out' + str(k) + '= th.Constant(' + str(out_constant[k]) + ')', globals())

    str2 = '{'
    if len(right_string) > 0:
        for i in range(len(right_string)):
            str2 += "'" + str(right_string[i]) + "': constant_out" + str(i) + ","
        str2 = str2[0:len(str2)-1] + "}"
        exec('swe_bnd[right_bnd_id] = ' + str2)

    solver_obj.bnd_functions['shallow_water'] = swe_bnd
    solver_obj.assign_initial_conditions(uv=uv_init, elev=elev_init)

    return solver_obj, update_forcings_hydrodynamics, outputdir
