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
V = th.FunctionSpace(new_mesh, 'CG', 1)

x,y = th.SpatialCoordinate(new_mesh)

bath_init = th.Function(V).interpolate(th.Constant(180/40) - x/40)

bath0 = initialise_fields(new_mesh, 'fixed_output/hydrodynamics_beach_bath_fixed_44_0.5')
bath1 = initialise_fields(new_mesh, 'fixed_output/hydrodynamics_beach_bath_fixed_55_0.5')
bath2 = initialise_fields(new_mesh, 'fixed_output/hydrodynamics_beach_bath_fixed_110_1')
bath3 = initialise_fields(new_mesh, 'fixed_output/hydrodynamics_beach_bath_fixed_132_1')
bath4 = initialise_fields(new_mesh, 'fixed_output/hydrodynamics_beach_bath_fixed_165_1')
bath5 = initialise_fields(new_mesh, 'fixed_output/hydrodynamics_beach_bath_fixed_220_1')
bath6 = initialise_fields(new_mesh, 'fixed_output/hydrodynamics_beach_bath_fixed_330_1')
bath_real = initialise_fields(new_mesh, 'fixed_output/hydrodynamics_beach_bath_fixed_440_1')

errorlist = []
errorlist.append(fire.errornorm(bath6, bath_real))
errorlist.append(fire.errornorm(bath5, bath_real))
errorlist.append(fire.errornorm(bath4, bath_real))
errorlist.append(fire.errornorm(bath3, bath_real))
errorlist.append(fire.errornorm(bath2, bath_real))
errorlist.append(fire.errornorm(bath1, bath_real))
errorlist.append(fire.errornorm(bath0, bath_real))

print(errorlist)


plt.loglog([2/3, 1, 4/3, 5/3, 2, 4, 5], errorlist, '-o')
plt.ylabel('Error norm (m)')
plt.xlabel(r'$\Delta x$ (m)')
plt.show()

logx = np.log([2/3, 1, 4/3, 5/3, 2, 4, 5])
log_error = np.log(errorlist)
poly = np.polyfit(logx, log_error, 1)
print(poly[0])


diff_bath_real = th.Function(V).interpolate(-bath_real+bath_init)

xaxisthetis1 = []
baththetis1 = []

for i in np.linspace(0, 219.9, 2200):
    xaxisthetis1.append(i)
    baththetis1.append(diff_bath_real.at([i, 5]))

plt.plot(xaxisthetis1, baththetis1)
plt.xlim([0, 220])
plt.xlabel('x (m)') 
plt.ylabel('Bed evolution (m)') 
plt.show()

plt.plot(xaxisthetis1, baththetis1)
plt.xlabel('x (m)') 
plt.ylabel('Bed evolution (m)') 
plt.xlim([65, 220])
plt.ylim([-0.15, 0.1])
plt.show()

bath0_mod = th.Function(V).interpolate(th.conditional(x >= 70, bath0, th.Constant(0.0)))
bath1_mod = th.Function(V).interpolate(th.conditional(x >= 70, bath1, th.Constant(0.0)))
bath2_mod = th.Function(V).interpolate(th.conditional(x >= 70, bath2, th.Constant(0.0)))
bath3_mod = th.Function(V).interpolate(th.conditional(x >= 70, bath3, th.Constant(0.0)))
bath4_mod = th.Function(V).interpolate(th.conditional(x >= 70, bath4, th.Constant(0.0)))
bath5_mod = th.Function(V).interpolate(th.conditional(x >= 70, bath5, th.Constant(0.0)))
bath6_mod = th.Function(V).interpolate(th.conditional(x >= 70, bath6, th.Constant(0.0)))
bath_real_mod = th.Function(V).interpolate(th.conditional(x >= 70, bath_real, th.Constant(0.0)))

errorlist = []
errorlist.append(fire.errornorm(bath6_mod, bath_real_mod))
errorlist.append(fire.errornorm(bath5_mod, bath_real_mod))
errorlist.append(fire.errornorm(bath4_mod, bath_real_mod))
errorlist.append(fire.errornorm(bath3_mod, bath_real_mod))
errorlist.append(fire.errornorm(bath2_mod, bath_real_mod))
errorlist.append(fire.errornorm(bath1_mod, bath_real_mod))
errorlist.append(fire.errornorm(bath0_mod, bath_real_mod))

print(errorlist)

plt.loglog([2/3, 1, 4/3, 5/3, 2, 4, 5], errorlist, '-o')
plt.ylabel('Error norm (m)')
plt.xlabel(r'$\Delta x$ (m)')
plt.show()

logx = np.log([2/3, 1, 4/3, 5/3, 2, 4, 5])
log_error = np.log(errorlist)
poly = np.polyfit(logx, log_error, 1)
print(poly[0])

