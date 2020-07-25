#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 14:59:04 2020

@author: mc4117
"""

import thetis as th
import morphological_hydro_mud_source_sink as morph

import numpy as np
import os

def boundary_conditions_shift_fn_balzano(bathymetry_2d, flag = None, morfac = 1, t_new = 0, state = 'initial'):
    """
    Define boundary conditions for problem to be used in morphological section.
    
    Inputs:
    morfac - morphological scale factor used when calculating time dependent boundary conditions
    t_new - timestep model currently at used when calculating time dependent boundary conditions
    state - when 'initial' this is the initial boundary condition set; when 'update' these are the boundary
            conditions set during update forcings (ie. if fluc_bcs = True, this will be called)
    """
    left_bnd_id = 1
    right_bnd_id = 2
    left_string = ['uv', 'elev']
    right_string = []
    
    
    # set boundary conditions
    swe_bnd = {}
   
    # boundary conditions
    h_amp = 0.5     # ocean boundary forcing amplitude
    omega = 0.5    # ocean boundary forcing period
    
    vel_amp = 1

    ocean_elev_func = lambda t: (h_amp * np.cos(-omega *(t+(100.0))))
    ocean_vel_func = lambda t: (vel_amp * np.cos(-omega *(t+(100.0))))
    
    if state == 'initial':
        elev_const = ocean_elev_func(0.0)
        vel_const = ocean_vel_func(0.0) 
                
        inflow_constant = [th.as_vector((vel_const, 0.0)), elev_const]
        outflow_constant = []
        return swe_bnd, left_bnd_id, right_bnd_id, inflow_constant, outflow_constant, left_string, right_string
    elif state == 'update':
        elev_const = ocean_elev_func(t_new)
        vel_const = ocean_vel_func(t_new)

        inflow_constant = [th.as_vector((vel_const, 0.0)), elev_const]
        outflow_constant = []

        return inflow_constant, outflow_constant

# define mesh
lx = 220
ly = 10
nx = lx
ny = 10
mesh2d = th.RectangleMesh(nx, ny, lx, ly)

x,y = th.SpatialCoordinate(mesh2d)

# define function spaces
V = th.FunctionSpace(mesh2d, 'CG', 1)
P1_2d = th.FunctionSpace(mesh2d, 'CG', 1)

# define underlying bathymetry
bathymetry_2d = th.Function(V, name='Bathymetry')
x,y = th.SpatialCoordinate(mesh2d)

beach_profile = -4+ x/25

bathymetry_2d.interpolate(-beach_profile)

value = 8/25

sponge_fn = th.Function(V).interpolate(th.conditional(x>=100, -99 + x, th.Constant(1.0)))

solver_obj, update_forcings_tracer, diff_bathy, diff_bathy_file, outputdir = morph.morphological(boundary_conditions_fn = boundary_conditions_shift_fn_balzano, morfac = 50, morfac_transport = True, suspendedload = True, convectivevel = True,\
                    bedload = True, angle_correction = False, slope_eff = True, seccurrent = False, wetting_and_drying = True, wetting_alpha = value, fluc_bcs = True, viscosity_hydro = 2*10**(-1), friction = 'manning',\
                 mesh2d = mesh2d, bathymetry_2d = bathymetry_2d, input_dir = 'hydrodynamics_beach_l_sep', ks = 0.025, average_size = 0.0002, dt = 0.05, final_time = 6*3600, cons_tracer = True, depth_integrated = True, sponge_viscosity = sponge_fn)

# User-defined output: moving bathymetry and eta_tilde
eta_tildefile = th.File(os.path.join(outputdir, 'eta_tilde.pvd'))
eta_tilde = th.Function(P1_2d, name="eta_tilde")

# user-specified export function
def export_func():
    wd_bath_displacement = solver_obj.depth.wd_bathymetry_displacement
    eta = solver_obj.fields.elev_2d
    eta_tilde.project(eta+wd_bath_displacement(eta))
    eta_tildefile.write(eta_tilde)

# run model
solver_obj.iterate(update_forcings = update_forcings_tracer, export_func = export_func)