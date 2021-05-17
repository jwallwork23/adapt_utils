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

new_mesh = RectangleMesh(600, 160, 30, 8)

bath_new = initialise_fields(new_mesh, '/Volumes/mc4117/bath_fixed_20_60_7_1_0_1')

new_new_mesh = RectangleMesh(300, 80, 30, 8)

bath_real = initialise_fields(new_mesh, '/Volumes/mc4117/bath_fixed_600_160')
bath_coarse = initialise_fields(new_new_mesh, '/Volumes/mc4117/bath_fixed_60_16')

print('L2')
print(fire.errornorm(bath_new, bath_real))

xaxis = []
bath_real_x = []
bath_mm_x = []
bath_mm_x2 = []
bath_coarse_x = []
for i in np.linspace(0, 29.8, 299):
    xaxis.append(i)
    bath_real_x.append(-bath_real.at([i, 4]))
    bath_mm_x.append(-bath_new.at([i,4]))
    #bath_mm_x2.append(-bath_new200.at([i,4]))    
    bath_coarse_x.append(-bath_coarse.at([i, 4]))
    
plt.plot(xaxis, bath_real_x, 'k:', linewidth = 2, label = '"True" value')
plt.plot(xaxis, bath_mm_x, label = 'Mesh movement')
#plt.plot(xaxis, bath_mm_x2, label = 'Mesh movement (60 elem) 400')
plt.plot(xaxis, bath_coarse_x, '--', label = 'Fixed mesh')
plt.xlabel('x (m)')
plt.ylabel('Bedlevel (m)')
plt.xlim([4, 13])
plt.ylim([-0.8, 0.01])
plt.legend()