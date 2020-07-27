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

new_mesh = th.RectangleMesh(880, 5*4, 220, 10)

bath1 = initialise_fields(new_mesh, 'hydrodynamics_beach_bath_new_55')
bath2 = initialise_fields(new_mesh, 'hydrodynamics_beach_bath_new_110')
bath3 = initialise_fields(new_mesh, 'hydrodynamics_beach_bath_new_165')
bath4 = initialise_fields(new_mesh, 'hydrodynamics_beach_bath_new_220')
bath5 = initialise_fields(new_mesh, 'hydrodynamics_beach_bath_new_330')
bath6 = initialise_fields(new_mesh, 'hydrodynamics_beach_bath_new_440')
bath_real = initialise_fields(new_mesh, 'hydrodynamics_beach_bath_new_880')

errorlist = []
errorlist.append(fire.errornorm(bath6, bath_real))
errorlist.append(fire.errornorm(bath5, bath_real))
errorlist.append(fire.errornorm(bath4, bath_real))
errorlist.append(fire.errornorm(bath3, bath_real))
errorlist.append(fire.errornorm(bath2, bath_real))
errorlist.append(fire.errornorm(bath1, bath_real))


print(errorlist)

plt.loglog([0.5, 2/3, 1, 4/3, 2, 4], errorlist, '-o')
plt.ylabel('Error norm (m)')
plt.xlabel(r'$\Delta x$ (m)')
plt.show()

logx = np.log([0.5, 2/3, 1, 4/3, 2, 4])
log_error = np.log(errorlist)
poly = np.polyfit(logx, log_error, 1)
print(poly[0])

