#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 16:37:44 2020

@author: mc4117
"""

from thetis import *
import firedrake as fire
import pylab as plt
import numpy as np

def initialise_fields(inputdir):
    """
    Initialise simulation with results from a previous simulation
    """

    from firedrake.petsc import PETSc
    try:
        import firedrake.cython.dmplex as dmplex
    except:
        import firedrake.dmplex as dmplex  # Older version        
    # mesh
    with timed_stage('mesh'):
            # Load
            newplex = PETSc.DMPlex().create()
            newplex.createFromFile(inputdir + '/myplex.h5')
            mesh = Mesh(newplex)
    V = FunctionSpace(mesh, 'CG', 1)  

    # elevation
    with timed_stage('initialising bathymetry'):
        chk = DumbCheckpoint(inputdir + "/bathymetry", mode=FILE_READ)
        bath = Function(V, name="bathymetry")
        chk.load(bath)
        chk.close()

    return bath

bath0 = initialise_fields('hydrodynamics_beach_bath_new_fixed_180_110')
bath1 = initialise_fields('adapt_output/hydrodynamics_beach_bath_mov_new_180_110_3_1_1')
bath2 = initialise_fields('adapt_output/hydrodynamics_beach_bath_new_next_mon_180_110_1_1_0')
bath3 = initialise_fields('hydrodynamics_beach_bath_new_110_basic')

x, y = SpatialCoordinate(bath0.function_space().mesh())
bath_orig0 = Function(bath0.function_space()).interpolate(Constant(160/40) - x/40)

x, y = SpatialCoordinate(bath1.function_space().mesh())
bath_orig1 = Function(bath1.function_space()).interpolate(Constant(160/40) - x/40)

x, y = SpatialCoordinate(bath2.function_space().mesh())
bath_orig2 = Function(bath2.function_space()).interpolate(Constant(160/40) - x/40)

x, y = SpatialCoordinate(bath3.function_space().mesh())
bath_orig3 = Function(bath3.function_space()).interpolate(Constant(160/40) - x/40)

diff_bath0 = Function(bath0.function_space()).interpolate(bath_orig0 - bath0)
diff_bath1 = Function(bath1.function_space()).interpolate(bath_orig1 - bath1)
diff_bath2 = Function(bath2.function_space()).interpolate(bath_orig2 - bath2)
diff_bath3 = Function(bath3.function_space()).interpolate(bath_orig3 - bath3)

diff0_file = File("diff_bath_zero.pvd").write(diff_bath0)
diff1_file = File("diff_bath_one.pvd").write(diff_bath1)
diff2_file = File("diff_bath_two.pvd").write(diff_bath2)
diff3_file = File("diff_bath_three.pvd").write(diff_bath3)
