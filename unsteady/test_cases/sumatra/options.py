from thetis import *
from thetis.configuration import *

from adapt_utils.unsteady.options import CoupledOptions
from adapt_utils.unsteady.swe.utils import heaviside_approx
from thetis.options import ModelOptions2d
from adapt_utils.unsteady.sediment.sediments_model import SedimentModel

import scipy.interpolate as ip
import os
import time
import datetime
import numpy as np
import pandas as pd
from matplotlib import rc
import pylab as plt

rc('text', usetex=True)


__all__ = ["SumatraOptions"]


class SumatraOptions(CoupledOptions):

    def __init__(self, friction='nik_solver', plot_timeseries=False, nx=1, ny=1, mesh = None, input_dir = None, output_dir = None, **kwargs):

        super(SumatraOptions, self).__init__(**kwargs)

        try:
            assert friction in ('nikuradse', 'manning', 'nik_solver')
        except AssertionError:
            raise ValueError("Friction parametrisation '{:s}' not recognised.".format(friction))
        self.friction = friction

        if output_dir is not None:
            self.di = output_dir

        self.plot_timeseries = plot_timeseries

        self.default_mesh = RectangleMesh(367, 5, 4404, 60)

        self.set_up_morph_model(self.default_mesh)

        self.plot_pvd = True

        self.grad_depth_viscosity = True

        # Stabilisation
        self.stabilisation = 'lax_friedrichs'

        self.morphological_acceleration_factor = Constant(1)
        
        self.ksp = Constant(0.03)

        # Time integration

        self.dt = 0.5
        self.end_time = 1000 #float(7140/self.morphological_acceleration_factor)
        self.dt_per_mesh_movement = 7140
        self.dt_per_export = 40
        self.timestepper = 'CrankNicolson'
        self.implicitness_theta = 1.0

    def set_up_morph_model(self, mesh = None):

        # Physical
        self.base_diffusivity = 1
        self.gravity = Constant(9.81)
        self.porosity = Constant(0.4)
        self.ks = Constant(0.05)
        self.average_size = 5e-4  # Average sediment size

        self.wetting_and_drying = True
        #self.depth_integrated = False
        #self.use_tracer_conservative_form = False
        #self.slope_eff = False
        #self.angle_correction = False
        #self.suspended = False
        #self.convective_vel_flag = False
        #self.bedload = False
        #self.solve_sediment = False
        #self.solve_exner = False
        
        self.water_sumatra = pd.read_csv('waterlevel.csv', header = None)
        self.water_level = ip.interpolate.interp1d(self.water_sumatra[0]*60, self.water_sumatra[1])

        self.norm_smoother = Constant(5)

        self.P1 = FunctionSpace(mesh, "CG", 1)
        bath = self.set_bathymetry(self.P1)

    #def create_sediment_model(self, mesh, bathymetry):
    #    self.P1DG = FunctionSpace(mesh, "DG", 1)
    #    self.P1_vec_dg = VectorFunctionSpace(mesh, "DG", 1)

    #    self.uv_d = Function(self.P1_vec_dg).project(as_vector((1e-10, 0.0)))
    #    self.eta_d = Function(self.P1DG).project(Constant(self.water_sumatra[1][0]))

    #    self.sediment_model = SedimentModel(ModelOptions2d, suspendedload=self.suspended, convectivevel=self.convective_vel_flag,
    #        bedload=self.bedload, angle_correction=self.angle_correction, slope_eff=self.slope_eff, seccurrent=False,
    #        mesh2d=mesh, bathymetry_2d=bathymetry,
    #                        uv_init = self.uv_d, elev_init = self.eta_d, ks=self.ks, average_size=self.average_size,
    #                        cons_tracer = self.use_tracer_conservative_form, wetting_and_drying = self.wetting_and_drying, wetting_alpha = self.wetting_and_drying_alpha)

    def set_bathymetry(self, fs, **kwargs):
        # interpolate bathymetry
        def interpolate_onto(interp_func, output_func, coords):
            bvector = output_func.dat.data
            mesh_xy = coords.dat.data
        
            assert mesh_xy.shape[0] == bvector.shape[0]
            for i, (node_x, node_y) in enumerate(mesh_xy):
                bvector[i] = interp_func((np.round(node_x, 10), np.round(node_y, 10)))
        
        bed = pd.read_csv('sumatra_bed.csv', header = None)
        x = bed[0]-min(bed[0])
        x[199] = x[199]-1
        x[195] = x[195]-1
        x = x + 360
        ext = np.linspace(0, 360, 31)
        x_ext = np.concatenate([ext[:-1], x])
        y = np.array([0, 12, 24, 36, 48, 60])
        ext_bed = [-bed[1][0] for i in range(30)]
        bath = np.concatenate([ext_bed, -bed[1]])
        bath_array = np.zeros((len(x_ext), len(y)))
        for i in range(len(y)):
            bath_array[:, i] = bath
        bath_interpolator = ip.interpolate.RegularGridInterpolator((x_ext, y), bath_array)

        self.bathymetry = Function(fs, name="Bathymetry")
        interpolate_onto(bath_interpolator, self.bathymetry, fs.mesh().coordinates)
        return self.bathymetry

    def set_viscosity(self, fs):
        x, y = SpatialCoordinate(fs.mesh())
        self.viscosity = Constant(1) #Function(fs)
        #self.viscosity.interpolate(conditional(x> 2000, conditional(x < 2860, ((100-1e-6)/(860)*(x-2000))+1e-6, Constant(100)), Constant(1e-6)))
        return self.viscosity

    
    def set_boundary_conditions(self, prob, i):
        if not hasattr(self, 'elev_in'):
            self.elev_in = Constant(0.0)
        self.elev_in.assign(self.water_sumatra[1][0])

        inflow_tag = 1
        boundary_conditions = {
            'shallow_water': {
                inflow_tag: {'elev': self.elev_in},
            },
	   'sediment': {
            }
        }
        return boundary_conditions    

    def update_boundary_conditions(self, solver_obj, t=0.0):
        if t < min(self.water_sumatra[0]*60):
            self.elev_in.assign(self.water_sumatra[1][0])
        else:
            self.elev_in.assign(self.water_level(t))
        print(t)
        print(self.elev_in.dat.data[:])

    def set_initial_condition(self, prob):
        u, eta = prob.fwd_solutions[0].split()
        u.project(as_vector((1e-10, 0.0)))
        eta.project(Constant(self.water_sumatra[1][0]))

    def set_sediment_source(self, fs):
        if self.suspended and not self.depth_integrated:
            return self.sediment_model.ero_term
        else:
            return None

    def set_sediment_sink(self, fs):
        if self.suspended and not self.depth_integrated:
            return self.sediment_model.depo_term
        else:
            return None

    def set_sediment_depth_integ_sink(self, fs):
        if self.suspended and self.depth_integrated:
            return self.sediment_model.depo_term
        else:
            return None

    def set_sediment_depth_integ_source(self, fs):
        if self.suspended and self.depth_integrated:
            return self.sediment_model.ero
        else:
            return None
    
    def set_wetting_and_drying_alpha(self, bathymetry, fs):

        #mesh = fs.mesh()
        #wetting_fn = Function(fs).interpolate(abs(bathymetry.dx(0)))
        #tau = TestFunction(fs)
        #n = FacetNormal(mesh)

        #K = 400*(12**2)/4
        #a = (inner(tau, wetting_fn)*dx)+(K*inner(grad(tau), grad(wetting_fn))*dx) - (K*(tau*inner(grad(wetting_fn), n)))*ds
        #a -= inner(tau, bathymetry_dx)*dx
        #solve(a == 0, wetting_fn)

        #self.wetting_and_drying_alpha = Function(fs).interpolate(conditional(wetting_fn > 0, dot(get_cell_widths_2d(mesh)[0], wetting_fn)+Constant(1), Constant(1)))
        print('wetting fn')
        return Function(self.P1).interpolate(Constant(2)) #self.wetting_and_drying_alpha

    #def set_advective_velocity_factor(self, fs):
    #    if self.convective_vel_flag:
    #        return self.sediment_model.corr_factor_model.corr_vel_factor
    #    else:
    #        return Constant(1.0)

    def set_initial_condition_sediment(self, prob):
        prob.fwd_solutions_sediment[0].interpolate(Constant(0.0))

    def set_initial_condition_bathymetry(self, prob):
        prob.fwd_solutions_bathymetry[0].interpolate(self.set_bathymetry(prob.fwd_solutions_bathymetry[0].function_space()))    

    def get_update_forcings(self, prob, i, adjoint):

        def update_forcings(t):
            self.update_boundary_conditions(prob, t=t)
            if t < 2:
                if t> 1:
                    u, eta = prob.fwd_solutions[i].split()
                    chk = DumbCheckpoint("mesh_new", mode=FILE_CREATE)
                    chk.store(eta.function_space().mesh().coordinates, name="mesh")
                    chk.close()

        return update_forcings


    def get_export_func(self, prob, i):

        def export_func():

            if not hasattr(self, 'eta_tilde_file'):
                self.eta_tilde_file = File(self.di + "/eta_tilde.pvd")
                eta_tilde_file = File(self.di + "/eta_tilde_orig.pvd")
                bath = prob.depth[i].bathymetry_2d
                uv, eta = prob.fwd_solutions[0].split()
                H = bath + eta
                wd_b = 0.5 * (sqrt(H ** 2 + prob.depth[i].wetting_and_drying_alpha ** 2) - H)
                tmp_function = Function(prob.depth[i].bathymetry_2d.function_space()).interpolate(prob.depth[i].get_total_depth(eta)) #eta + wd_b)
                eta_tilde_file.write(tmp_function)
            else:
                if not hasattr(self, 'eta_tilde_new'):
                    self.eta_tilde_new = Function(prob.depth[i].bathymetry_2d.function_space()).interpolate(prob.depth[i].bathymetry_2d) #depth[i].get_total_depth(eta))
                bath = prob.depth[i].bathymetry_2d #prob.bathymetry[0]
                uv, eta = prob.fwd_solutions[0].split()
                H = bath + eta
                wd_b = 0.5 * (sqrt(H ** 2 + prob.depth[i].wetting_and_drying_alpha ** 2) - H)
                self.eta_tilde_new.project(prob.depth[i].get_total_depth(eta)) #eta+wd_b)
                self.eta_tilde_file.write(self.eta_tilde_new)


        return export_func


    def set_boundary_surface(self):
        """Set the initial displacement of the boundary elevation."""
        pass
