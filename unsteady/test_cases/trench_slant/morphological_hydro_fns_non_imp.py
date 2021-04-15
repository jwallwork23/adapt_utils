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
import firedrake as fire
import math
import os
from firedrake.petsc import PETSc


def hydrodynamics_only(boundary_conditions_fn, mesh2d, bathymetry_2d, uv_init, elev_init, ks, average_size, dt, t_end, friction='nikuradse', friction_coef=0, fluc_bcs=False, viscosity=10**(-6), diffusivity=0.15):
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
    ks - bottom friction coefficient for quadratic drag coefficient
    average_size - average sediment size
    dt - timestep
    t_end - end time
    viscosity - viscosity of hydrodynamics. The default value should be 10**(-6)
    friction - choice of friction formulation - nikuradse and manning
    friction_coef - value of friction coefficient used in manning

    Outputs:
    solver_obj - solver which we need to run to solve the problem
    update_forcings_hydrodynamics - function defining the updates to the model performed at each timestep
    """
    def update_forcings_hydrodynamics(t_new):
        # update bed friction
        uv1, elev1 = solver_obj.fields.solution_2d.split()
        depth.interpolate(elev1 + bathymetry_2d)
        # calculate skin friction coefficient
        cfactor.interpolate(2*(0.4**2)/((th.ln(11.036*depth/(ksp)))**2))

    # choose directory to output results
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    outputdir = 'outputs' + st

    # export interval in seconds
    if t_end < 40:
        t_export = 1
    else:
        t_export = np.round(t_end/40, 0)

    th.print_output('Exporting to '+outputdir)

    # define function spaces
    V = th.FunctionSpace(mesh2d, 'CG', 1)
    P1_2d = th.FunctionSpace(mesh2d, 'DG', 1)

    # define parameters
    ksp = th.Constant(3*average_size)

    # define depth
    depth = th.Function(V).interpolate(elev_init + bathymetry_2d)

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
    # set horizontal diffusivity parameter
    options.horizontal_diffusivity = th.Constant(diffusivity)
    options.horizontal_viscosity = th.Constant(viscosity)

    # crank-nicholson used to integrate in time system of ODEs resulting from application of galerkin FEM
    options.timestepper_type = 'CrankNicolson'
    options.timestepper_options.implicitness_theta = 1.0

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
    return solver_obj, update_forcings_hydrodynamics



def export_final_state(inputdir, uv, elev,):
    """
    Export fields to be used in a subsequent simulation
    """
    if not os.path.exists(inputdir):
        os.makedirs(inputdir)
    th.print_output("Exporting fields for subsequent simulation")
    chk = th.DumbCheckpoint(inputdir + "/velocity", mode=th.FILE_CREATE)
    chk.store(uv, name="velocity")
    th.File(inputdir + '/velocityout.pvd').write(uv)
    chk.close()
    chk = th.DumbCheckpoint(inputdir + "/elevation", mode=th.FILE_CREATE)
    chk.store(elev, name="elevation")
    th.File(inputdir + '/elevationout.pvd').write(elev)
    chk.close()
    
    plex = elev.function_space().mesh()._plex
    viewer = PETSc.Viewer().createHDF5(inputdir + '/myplex.h5', 'w')
    viewer(plex)


def initialise_fields(mesh2d, inputdir, outputdir,):
    """
    Initialise simulation with results from a previous simulation
    """
    DG_2d = th.FunctionSpace(mesh2d, 'DG', 1)
    # elevation
    with th.timed_stage('initialising elevation'):
        chk = th.DumbCheckpoint(inputdir + "/elevation", mode=th.FILE_READ)
        elev_init = th.Function(DG_2d, name="elevation")
        chk.load(elev_init)
        th.File(outputdir + "/elevation_imported.pvd").write(elev_init)
        chk.close()
    # velocity
    with th.timed_stage('initialising velocity'):
        chk = th.DumbCheckpoint(inputdir + "/velocity", mode=th.FILE_READ)
        V = th.VectorFunctionSpace(mesh2d, 'DG', 1)
        uv_init = th.Function(V, name="velocity")
        chk.load(uv_init)
        th.File(outputdir + "/velocity_imported.pvd").write(uv_init)
        chk.close()
        return elev_init, uv_init,
