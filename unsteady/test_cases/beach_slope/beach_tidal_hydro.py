"""
Beach Profile Test case
=======================

Solves the initial hydrodynamics simulation of a beach profile

"""

import thetis as th
import hydro_fns as hydro

import numpy as np
import os

plot = True

from adapt_utils.io import export_hydrodynamics


def boundary_conditions_fn_balzano(bathymetry_2d, flag=None, morfac=1, t_new=0, state='initial'):
    """
    Define boundary conditions for problem.

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
    h_amp = 0.25     # ocean boundary forcing amplitude
    omega = 0.5    # ocean boundary forcing period
    ocean_elev_func = lambda t: (h_amp * np.cos(-omega*t))

    vel_amp = 0.5
    ocean_vel_func = lambda t: (vel_amp * np.cos(-omega*t))

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

fac_x = 0.5
fac_y = 1

# define mesh
lx = 220
ly = 10
nx = np.int(lx*fac_x)
ny = np.int(10*fac_y)
mesh2d = th.RectangleMesh(nx, ny, lx, ly)

# define function spaces
V = th.FunctionSpace(mesh2d, 'CG', 1)
P1_2d = th.FunctionSpace(mesh2d, 'CG', 1)

# define underlying bathymetry
bathymetry_2d = th.Function(V, name='Bathymetry')
x, y = th.SpatialCoordinate(mesh2d)

beach_profile = (-180/40 + x/40)

bathymetry_2d.interpolate(-beach_profile)

# define initial elevation
elev_init = th.Function(P1_2d).interpolate(th.Constant(0.0))


uv_init = th.Constant((10**(-7), 0.))

value = 1/40


sponge_fn = th.Function(V).interpolate(th.conditional(x >= 100, -399 + 4*x, th.Constant(1.0)))

solver_obj, update_forcings_hydrodynamics, outputdir = hydro.hydrodynamics_only(boundary_conditions_fn_balzano, mesh2d, bathymetry_2d, uv_init, elev_init, wetting_and_drying=True, wetting_alpha=value, fluc_bcs=True, average_size=200*(10**(-6)), dt=0.05, t_end=100, friction='manning', sponge_viscosity=sponge_fn, viscosity=0.5)

# run model

solver_obj.iterate(update_forcings=update_forcings_hydrodynamics)

uv, elev = solver_obj.fields.solution_2d.split()

fpath = "hydrodynamics_beach_l_sep_nx_{:d}_{:d}".format(nx, ny)

if plot == False:
    export_hydrodynamics(uv, elev, fpath)
else:
    import pylab as plt

    x = np.linspace(0, 220, 221)

    bath = [-(4.5 - i/40) for i in x]

    # change t_end = 30
    wd_bath_displacement = solver_obj.depth.wd_bathymetry_displacement
    eta = solver_obj.fields.elev_2d
    eta_tilde = th.Function(P1_2d).project(eta+wd_bath_displacement(eta))

    xaxisthetis1 = []
    elevthetis1 = []

    for i in np.linspace(0, 219, 220):
        xaxisthetis1.append(i)
        elevthetis1.append(eta_tilde.at([i, 5]))



    plt.plot(xaxisthetis1, elevthetis1, label = 'water surface')
    plt.plot(x, bath, label = 'bed height')
    plt.xlabel(r'$x$ (m)')
    plt.ylabel(r'height (m)')
    plt.legend(loc = 3)
