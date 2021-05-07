#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 16:37:44 2020

@author: mc4117
"""

import thetis as th
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
    V = th.FunctionSpace(mesh2d, 'CG', 1)
    # elevation
    with th.timed_stage('initialising bathymetry'):
        chk = th.DumbCheckpoint(inputdir + "/bathymetry", mode=th.FILE_READ)
        bath = th.Function(V, name="bathymetry")
        chk.load(bath)
        chk.close()

    return bath

new_mesh = th.RectangleMesh(16*5*4, 5*4, 16, 1.1)

bath1 = initialise_fields(new_mesh, 'hydrodynamics_trench_slant_bath_new_0.1')
bath2 = initialise_fields(new_mesh, 'hydrodynamics_trench_slant_bath_new_0.125')
bath3 = initialise_fields(new_mesh, 'hydrodynamics_trench_slant_bath_new_0.25')
bath4 = initialise_fields(new_mesh, 'hydrodynamics_trench_slant_bath_new_0.4')
bath5 = initialise_fields(new_mesh, 'hydrodynamics_trench_slant_bath_new_0.5')
bath6 = initialise_fields(new_mesh, 'hydrodynamics_trench_slant_bath_new_0.8')
#bath4b = initialise_fields(new_mesh, 'hydrodynamics_trench_slant_bath_new_1.6')
#bath5 =  initialise_fields(new_mesh, 'hydrodynamics_trench_slant_bath_new_2.0')
bath_real = initialise_fields(new_mesh, 'hydrodynamics_trench_slant_bath_new_code_4.0')

errorlist = []
errorlist.append(fire.errornorm(bath1, bath_real))
errorlist.append(fire.errornorm(bath2, bath_real))
errorlist.append(fire.errornorm(bath3, bath_real))
errorlist.append(fire.errornorm(bath4, bath_real))
errorlist.append(fire.errornorm(bath5, bath_real))
errorlist.append(fire.errornorm(bath6, bath_real))

print(errorlist)

logx = np.log([8, 10, 20, 32, 40, 64])
log_error = np.log(errorlist)
poly = np.polyfit(logx, log_error, 1)
print(poly[0])

fig, ax = plt.subplots()
ax.loglog([8, 10, 20, 32, 40, 64], errorlist, '-o', label = '__no_label__')
ax.loglog([8, 10, 20, 32, 40, 64], [i**(-1) for i in [8, 10, 20, 32, 40, 64]], '--', label = '1st order')
ax.set_xlabel(r'Number of mesh elements in $x$-direction')
ax.set_ylabel("L2 error norm")
ax.set_xticks([10, 20, 30, 40, 60])
ax.set_xticklabels([10, 20, 30, 40, 60])
ax.set_yticks([0.005, 0.01, 0.02, 0.05])
ax.set_yticklabels([0.005, 0.01, 0.02, 0.05])
plt.legend()
plt.show()
