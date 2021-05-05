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
    Export fields to be used in a subsequent simulation
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
    Initialise simulation with results from a previous simulation
    """
    V = FunctionSpace(mesh2d, 'CG', 1)
    # elevation
    with timed_stage('initialising bathymetry'):
        chk = DumbCheckpoint(inputdir + "/bathymetry", mode=FILE_READ)
        bath = Function(V, name="bathymetry")
        chk.load(bath)
        chk.close()
    return bath

new_mesh = RectangleMesh(600, 160, 30, 8)

bath_new = initialise_fields(new_mesh, 'fixed_output/bath_fixed_90_24')

new_new_mesh = RectangleMesh(600, 160, 30, 8)

V = FunctionSpace(new_new_mesh, 'CG', 1)

bath_new_new = Function(V).project(bath_new)

bath_real = initialise_fields(new_new_mesh, 'fixed_output/bath_fixed_600_160')

print('L2')
print(fire.errornorm(bath_new_new, bath_real))

