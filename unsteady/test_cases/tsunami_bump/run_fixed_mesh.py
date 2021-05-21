"""
Tsunami Bump Test case
=======================

Solves the hydro-morphodynamic simulation of a tsunami-like wave with an obstacle in the profile 
on a fixed uniform mesh

"""


from thetis import *
import firedrake as fire
from firedrake.petsc import PETSc

import pylab as plt
import pandas as pd
import numpy as np
import time
import datetime

from adapt_utils.unsteady.test_cases.tsunami_bump.options import BeachOptions
from adapt_utils.unsteady.solver import AdaptiveProblem

def export_final_state(inputdir, bathymetry_2d):
    """
    Export bathymetry and mesh
    """
    if not os.path.exists(inputdir):
        os.makedirs(inputdir)
    print_output("Exporting fields for subsequent simulation")

    chk = DumbCheckpoint(inputdir + "/bathymetry", mode=FILE_CREATE)
    chk.store(bathymetry_2d, name="bathymetry")
    File(inputdir + '/bathout.pvd').write(bathymetry_2d)
    chk.close()
    plex = bathymetry_2d.function_space().mesh()._topology_dm
    viewer = PETSc.Viewer().createHDF5(inputdir + '/myplex.h5', 'w')
    viewer(plex)

def initialise_fields(mesh2d, inputdir):
    """
    Initialise true value bathymetry
    """
    V = FunctionSpace(mesh2d, 'CG', 1)
    # elevation
    with timed_stage('initialising bathymetry'):
        chk = DumbCheckpoint(inputdir + "/bathymetry", mode=FILE_READ)
        bath = Function(V, name="bathymetry")
        chk.load(bath)
        chk.close()
    return bath

t1 = time.time()

nx = 4
ny = 4

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
outputdir = 'outputs' + st

kwargs = {
    'approach': 'fixed_mesh',
    'nx': nx,
    'ny': ny,
    'plot_pvd': True,
    'output_dir': outputdir,
    # Spatial discretisation
    'family': 'dg-dg',
    'stabilisation': None,
    'use_automatic_sipg_parameter': True,
    'friction': 'quadratic'
}

op = BeachOptions(**kwargs)
swp = AdaptiveProblem(op)

t1 = time.time()
swp.solve_forward()
t2 = time.time()

print(t2-t1)

new_mesh = RectangleMesh(600, 160, 30, 8)

bath = Function(FunctionSpace(new_mesh, "CG", 1)).project(swp.fwd_solutions_bathymetry[0])

export_final_state("fixed_output/bath_fixed_"+str(int(nx*30)) + '_' + str(int(ny*8)), bath)

# export and save bathymetry in readable format
bath_real = initialise_fields(new_mesh, 'fixed_output/bath_fixed_600_160')

# calculate error to true value
print('L2')
print(fire.errornorm(bath, bath_real))
