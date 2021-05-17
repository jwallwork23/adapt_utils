#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 20:00:01 2021

@author: mc4117
"""
from thetis import *
import firedrake as fire
import pylab as plt
import numpy as np

import matplotlib
font = {'size'   : 12}
matplotlib.rc('font', **font)

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

new_mesh = RectangleMesh(880, 20, 220, 10)

bath_new_whole = initialise_fields(new_mesh, '/Volumes/mc4117/hydrodynamics_beach_bath_mov_648_110_5_1_0')

bath_real = initialise_fields(new_mesh, '/Volumes/mc4117/hydrodynamics_beach_bath_fixed_440_10')
bath_coarse = initialise_fields(new_mesh, '/Volumes/mc4117/hydrodynamics_beach_bath_fixed_110_10')

print('L2')
print(fire.errornorm(bath_coarse, bath_real))

V = FunctionSpace(new_mesh, 'CG', 1)
x, y = SpatialCoordinate(new_mesh)
init_bath = Function(V).interpolate((Constant(180/40) - x/40))

xaxis = []
bath_real_x = []
bath_mm_x = []
bath_coarse_x = []
for i in np.linspace(0, 219, 221):
    xaxis.append(i)
    bath_real_x.append(-bath_real.at([i, 5])+init_bath.at([i, 5]))
    bath_mm_x.append(-bath_new_whole.at([i, 5])+init_bath.at([i, 5]))
    bath_coarse_x.append(-bath_coarse.at([i, 5])+init_bath.at([i, 5]))
    
plt.plot(xaxis, bath_real_x, label = '"True" value')
plt.plot(xaxis, bath_mm_x, label = 'Mesh movement whole (110 elem)')
plt.plot(xaxis, bath_coarse_x, '--', label = 'Fixed mesh (110 elem)')
plt.legend()