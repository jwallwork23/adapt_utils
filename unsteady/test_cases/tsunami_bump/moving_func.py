from firedrake.petsc import PETSc
import firedrake as fire
from thetis import *

import numpy as np


from adapt_utils.io import initialise_bathymetry, export_bathymetry
from adapt_utils.unsteady.test_cases.tsunami_bump.options import BeachOptions
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.adapt import recovery
from adapt_utils.norms import local_frobenius_norm, local_norm

import pandas as pd
import time
import datetime

def main(filename, alpha, mod, beta, gamma, nx, ny):

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

    beta_mod = 1
    outputdir = 'outputs' + str(alpha)

    r_tol = 1e-3

    kwargs = {
        'approach': 'monge_ampere',
        'nx': nx,
        'ny': ny,
        'plot_pvd': True,
        'output_dir': outputdir,
        'nonlinear_method': 'relaxation',
    	'r_adapt_rtol': r_tol,
    	# Spatial discretisation
    	'family': 'dg-dg',
    	'stabilisation': None,
    	'use_automatic_sipg_parameter': True,
    	'friction': 'quadratic'
         }

    op = BeachOptions(**kwargs)
    assert op.num_meshes == 1
    swp = AdaptiveProblem(op)
    swp.shallow_water_options[0]['mesh_velocity'] = None

    def gradient_interface_monitor(mesh, mod = mod, beta_mod = beta_mod, alpha=alpha, beta=beta, gamma=gamma):

        """
        Monitor function focused around the steep_gradient (budd acta numerica)

        NOTE: Defined on the *computational* mesh.

        """
        P1 = FunctionSpace(mesh, "CG", 1)

        eta = swp.fwd_solutions[0].split()[1]
        b = swp.fwd_solutions_bathymetry[0]

        bath_gradient = recovery.construct_gradient(b)
        bath_hess = recovery.construct_hessian(b, op=op)

        frob_bath_hess = Function(b.function_space()).project(local_frobenius_norm(bath_hess))

        if max(abs(frob_bath_hess.dat.data[:]))<1e-10:
            frob_bath_norm = Function(b.function_space()).project(frob_bath_hess)
        else:
            frob_bath_norm = Function(b.function_space()).project(frob_bath_hess/max(frob_bath_hess.dat.data[:]))

        current_mesh = b.function_space().mesh()
        l2_bath_grad = Function(b.function_space()).project(abs(local_norm(bath_gradient)))

        bath_dx_l2_norm = Function(b.function_space()).interpolate(l2_bath_grad/max(l2_bath_grad.dat.data[:]))

        alpha_mod = alpha*mod

        elev_abs_norm = Function(eta.function_space()).interpolate(alpha_mod*pow(cosh(beta_mod*(eta+b)), -2))

        comp_int = conditional(alpha*beta*bath_dx_l2_norm > alpha*gamma*frob_bath_norm, alpha*beta*bath_dx_l2_norm, alpha*gamma*frob_bath_norm)
        comp = interpolate(comp_int + elev_abs_norm, b.function_space())
        comp_new = project(comp, P1)
        comp_new2 = interpolate(conditional(comp_new > Constant(0.0), comp_new, Constant(0.0)), P1)
        mon_init = project(Constant(1.0) + comp_new2, P1)

        return mon_init

    swp.set_monitor_functions(gradient_interface_monitor)

    t1 = time.time()
    swp.solve_forward()
    t2 = time.time()

    f = open(filename, 'a')
    f.write(str(t2-t1))
    f.write('\n')
    f.write(str(nx))
    f.write('\n')
    f.write(str(alpha))
    f.write('\n')
    f.write(str(mod))
    f.write('; ')
    f.write(str(beta))
    f.write('; ')
    f.write(str(gamma))

    new_mesh = RectangleMesh(600, 160, 30, 8)

    bath = Function(FunctionSpace(new_mesh, "CG", 1)).project(swp.fwd_solutions_bathymetry[0])

    export_final_state("adapt_output/bath_fixed_" + str(op.dt_per_export) + "_" +str(int(nx*30)) + "_" + str(alpha) +'_' + str(beta) + '_' + str(gamma) + '_' + str(mod), bath)

    bath_real = initialise_fields(new_mesh, 'fixed_output/bath_fixed_600_160')

    f.write('\n')
    f.write(str(fire.errornorm(bath, bath_real)))
    f.write('\n')
