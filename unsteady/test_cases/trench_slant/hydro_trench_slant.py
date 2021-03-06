"""
Migrating Trench 2D Test case
=======================

Solves initial hydrodynamics simulation of a 2D migrating trench

"""
import thetis as th
import hydro_fns as hydro
import numpy as np

from adapt_utils.io import export_hydrodynamics

timestep = 0.25
fac_x = 0.5
fac_y = 0.5


def boundary_conditions_fn_trench(bathymetry_2d, flag, morfac=1, t_new=0, state='initial'):
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
    left_string = ['flux']
    right_string = ['elev']

    # set boundary conditions

    swe_bnd = {}

    flux_constant = -0.22
    elev_constant2 = 0.397
    inflow_constant = [flux_constant]
    outflow_constant = [elev_constant2]
    return swe_bnd, left_bnd_id, right_bnd_id, inflow_constant, outflow_constant, left_string, right_string


# define mesh
lx = 16
ly = 1.1
nx = np.int(lx*5*fac_x)
ny = np.int(np.ceil(5*fac_y))

mesh2d = th.RectangleMesh(nx, ny, lx, ly)

x, y = th.SpatialCoordinate(mesh2d)

# define function spaces
V = th.FunctionSpace(mesh2d, 'CG', 1)
P1_2d = th.FunctionSpace(mesh2d, 'DG', 1)
vectorP1_2d = th.VectorFunctionSpace(mesh2d, 'DG', 1)

# define underlying bathymetry
bathymetry_2d = th.Function(V, name='Bathymetry')
initialdepth = th.Constant(0.297)
depth_riv = th.Constant(initialdepth - 0.397)
depth_trench = th.Constant(depth_riv - 0.15)
depth_diff = depth_trench - depth_riv

trench = th.conditional(th.le(x, 5), (0.1*(y-0.55)) + depth_riv, th.conditional(th.le(x, 6.5), (0.1*(y-0.55)) + (1/1.5)*depth_diff*(x-6.5) + depth_trench,
                        th.conditional(th.le(x, 9.5), (0.1*(y-0.55)) + depth_trench, th.conditional(th.le(x, 11), (0.1*(y-0.55)) - (1/1.5)*depth_diff*(x-11) + depth_riv, (0.1*(y-0.55)) + depth_riv))))
bathymetry_2d.interpolate(-trench)

# define initial elevation
elev_init = th.Function(P1_2d).interpolate(th.Constant(0.4))
uv_init = th.as_vector((0.51, 0.0))

solver_obj, update_forcings_hydrodynamics = hydro.hydrodynamics_only(boundary_conditions_fn_trench, mesh2d, bathymetry_2d, uv_init, elev_init, average_size=160 * (10**(-6)), dt=timestep, t_end=500)

# run model
solver_obj.iterate(update_forcings=update_forcings_hydrodynamics)


uv, elev = solver_obj.fields.solution_2d.split()
fpath = "hydrodynamics_trench_slant_" + str(fac_x)
export_hydrodynamics(uv, elev, fpath)
