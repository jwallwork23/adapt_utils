"""
Migrating Trench Test case
=======================

Solves the initial hydrodynamics simulation of a migrating trench.
<<<<<<< HEAD

[1] Clare et al. 2020. “Hydro-morphodynamics 2D Modelling Using a Discontinuous
    Galerkin Discretisation.” EarthArXiv. January 9. doi:10.31223/osf.io/tpqvy.

=======
>>>>>>> origin/master
"""
from thetis import *
from firedrake.petsc import PETSc

import argparse
import time


<<<<<<< HEAD
def export_final_state(inputdir, uv, elev,):
    """
    Export fields to be used in a subsequent simulation
    """
    if not os.path.exists(inputdir):
        os.makedirs(inputdir)
    print_output("Exporting fields for subsequent simulation")
    chk = DumbCheckpoint(inputdir + "/velocity", mode=FILE_CREATE)
    chk.store(uv, name="velocity")
    File(inputdir + '/velocityout.pvd').write(uv)
    chk.close()
    chk = DumbCheckpoint(inputdir + "/elevation", mode=FILE_CREATE)
    chk.store(elev, name="elevation")
    File(inputdir + '/elevationout.pvd').write(elev)
    chk.close()

    plex = elev.function_space().mesh()._plex
    viewer = PETSc.Viewer().createHDF5(inputdir + '/myplex.h5', 'w')
    viewer(plex)

res = 0.4
=======
>>>>>>> origin/master

# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("-res", help="Mesh resolution factor (default 0.5).")
args = parser.parse_args()

res = float(args.res or 0.5)


# --- Setup

# Define mesh
lx = 16
ly = 1.1
nx = np.int(16*5*res)
ny = 5 if res < 4 else 10
mesh2d = RectangleMesh(nx, ny, lx, ly)
x, y = SpatialCoordinate(mesh2d)

# Define function spaces
V = FunctionSpace(mesh2d, "CG", 1)
P1_2d = FunctionSpace(mesh2d, "DG", 1)

# Define underlying bathymetry
bathymetry_2d = Function(V, name='Bathymetry')
initialdepth = Constant(0.397)
depth_riv = Constant(initialdepth - 0.397)
depth_trench = Constant(depth_riv - 0.15)
depth_diff = depth_trench - depth_riv
trench = conditional(
    le(x, 5), depth_riv, conditional(
        le(x, 6.5), (1/1.5)*depth_diff*(x-6.5) + depth_trench,
        conditional(le(x, 9.5), depth_trench, conditional(
            le(x, 11), -(1/1.5)*depth_diff*(x-11) + depth_riv, depth_riv))))
bathymetry_2d.interpolate(-trench)


# --- Simulate initial hydrodynamics

# Define initial elevation
elev_init = Function(P1_2d).interpolate(Constant(0.4))
uv_init = as_vector((0.51, 0.0))

# Choose directory to output results
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
<<<<<<< HEAD
outputdir = 'outputs' + st

print_output('Exporting to '+outputdir)
=======
outputdir = os.path.join(os.path.dirname(__file__), 'outputs' + st)
print_output('Exporting to ' + outputdir)
>>>>>>> origin/master

# Define parameters
t_end = 500
t_export = np.round(t_end/40, 0)  # Export interval in seconds
average_size = 160*1.0e-06
ksp = Constant(3*average_size)

# Setup solver
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.output_directory = outputdir
options.check_volume_conservation_2d = True
options.fields_to_export = ['uv_2d', 'elev_2d']
options.solve_tracer = False
options.use_lax_friedrichs_tracer = False
options.nikuradse_bed_roughness = ksp
options.horizontal_viscosity = Constant(1e-6)
options.timestepper_type = 'CrankNicolson'
<<<<<<< HEAD
options.timestepper_options.implicitness_theta = 1.0
options.norm_smoother = Constant(0.1)

=======
options.timestepper_options.implicitness_theta = 1.0  # i.e. Implicit Euler
>>>>>>> origin/master
if not hasattr(options.timestepper_options, 'use_automatic_timestep'):
    options.timestep = 0.25 if res < 4 else 0.125
left_bnd_id = 1
right_bnd_id = 2
swe_bnd = {}
swe_bnd[left_bnd_id] = {'flux': Constant(-0.22)}
swe_bnd[right_bnd_id] = {'elev': Constant(0.397)}
solver_obj.bnd_functions['shallow_water'] = swe_bnd
solver_obj.assign_initial_conditions(uv=uv_init, elev=elev_init)

# Run model
solver_obj.iterate()
uv, elev = solver_obj.fields.solution_2d.split()
<<<<<<< HEAD
export_final_state("hydrodynamics_trench" + str(res), uv, elev)
=======
fpath = "hydrodynamics_trench_{:.4f}".format(res)
export_hydrodynamics(uv, elev, fpath)
>>>>>>> origin/master
