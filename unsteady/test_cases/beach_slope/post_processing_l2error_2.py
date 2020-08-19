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

bath0 = initialise_fields(new_mesh, 'adapt_output/hydrodynamics_beach_bath_mov_1296_110_7_0_1')
bath1 = initialise_fields(new_mesh, 'adapt_output/hydrodynamics_beach_bath_mov_648_110_7_0_1')
bath2 = initialise_fields(new_mesh, 'adapt_output/hydrodynamics_beach_bath_mov_324_110_7_0_1')
bath3 = initialise_fields(new_mesh, 'adapt_output/hydrodynamics_beach_bath_mov_96_110_7_0_1')
bath4 = initialise_fields(new_mesh, 'adapt_output/hydrodynamics_beach_bath_mov_72_110_7_0_1')
bath5 = initialise_fields(new_mesh, 'adapt_output/hydrodynamics_beach_bath_mov_54_110_7_0_1')
bath6 = initialise_fields(new_mesh, 'adapt_output/hydrodynamics_beach_bath_mov_48_110_7_0_1')
bath7 = initialise_fields(new_mesh, 'adapt_output/hydrodynamics_beach_bath_mov_32_110_7_0_1')
bath_real = initialise_fields(new_mesh, 'fixed_output/hydrodynamics_beach_bath_fixed_440_1')

bath0_mod = th.Function(V).interpolate(th.conditional(x >= 0, bath0, th.Constant(0.0)))
bath1_mod = th.Function(V).interpolate(th.conditional(x >= 0, bath1, th.Constant(0.0)))
bath2_mod = th.Function(V).interpolate(th.conditional(x >= 0, bath2, th.Constant(0.0)))
bath3_mod = th.Function(V).interpolate(th.conditional(x >= 0, bath3, th.Constant(0.0)))
bath4_mod = th.Function(V).interpolate(th.conditional(x >= 0, bath4, th.Constant(0.0)))
bath5_mod = th.Function(V).interpolate(th.conditional(x >= 0, bath5, th.Constant(0.0)))
bath6_mod = th.Function(V).interpolate(th.conditional(x >= 0, bath6, th.Constant(0.0)))
bath7_mod = th.Function(V).interpolate(th.conditional(x >= 0, bath7, th.Constant(0.0)))
bath_real_mod = th.Function(V).interpolate(th.conditional(x >= 0, bath_real, th.Constant(0.0)))

errorlist = []
errorlist.append(fire.errornorm(bath7_mod, bath_real_mod))
errorlist.append(fire.errornorm(bath6_mod, bath_real_mod))
errorlist.append(fire.errornorm(bath5_mod, bath_real_mod))
errorlist.append(fire.errornorm(bath4_mod, bath_real_mod))
errorlist.append(fire.errornorm(bath3_mod, bath_real_mod))
errorlist.append(fire.errornorm(bath2_mod, bath_real_mod))
errorlist.append(fire.errornorm(bath1_mod, bath_real_mod))
errorlist.append(fire.errornorm(bath0_mod, bath_real_mod))

print(errorlist)

"""
plt.loglog([0.5, 2/3, 1, 4/3, 2, 4], errorlist, '-o')
plt.ylabel('Error norm (m)')
plt.xlabel(r'$\Delta x$ (m)')
plt.show()

logx = np.log([0.5, 2/3, 1, 4/3, 2, 4])
log_error = np.log(errorlist)
poly = np.polyfit(logx, log_error, 1)
print(poly[0])
"""
