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

new_mesh = th.RectangleMesh(16*5*4, 5*4, 16, 1.1)

bath1 = initialise_fields(new_mesh, 'adapt_output/hydrodynamics_trench_slant_bath_3_0_1-0.8')
bath2 = initialise_fields(new_mesh, 'adapt_output/hydrodynamics_trench_slant_bath_3_1_0-0.8')
bath3 = initialise_fields(new_mesh, 'adapt_output/hydrodynamics_trench_slant_bath_3_1_1-0.8')
bath4 = initialise_fields(new_mesh, 'adapt_output/hydrodynamics_trench_slant_bath_5_1_0-0.8')
bath5 = initialise_fields(new_mesh, 'adapt_output/hydrodynamics_trench_slant_bath_5_0_1-0.8')
bath6 = initialise_fields(new_mesh, 'adapt_output/hydrodynamics_trench_slant_bath_5_1_1-0.8')
bath7 = initialise_fields(new_mesh, 'adapt_output/hydrodynamics_trench_slant_bath_1_1_0-0.8')
bath8 = initialise_fields(new_mesh, 'adapt_output/hydrodynamics_trench_slant_bath_1_0_1-0.8')
bath9 = initialise_fields(new_mesh, 'adapt_output/hydrodynamics_trench_slant_bath_1_1_1-0.8')
bath_real = initialise_fields(new_mesh, 'hydrodynamics_trench_slant_bath_new_code_4.0')

errorlist = []
errorlist.append(fire.errornorm(bath1, bath_real))
errorlist.append(fire.errornorm(bath2, bath_real))
errorlist.append(fire.errornorm(bath3, bath_real))
errorlist.append(fire.errornorm(bath4, bath_real))
errorlist.append(fire.errornorm(bath5, bath_real))
errorlist.append(fire.errornorm(bath6, bath_real))
errorlist.append(fire.errornorm(bath7, bath_real))
errorlist.append(fire.errornorm(bath8, bath_real))
errorlist.append(fire.errornorm(bath9, bath_real))

print(errorlist)
