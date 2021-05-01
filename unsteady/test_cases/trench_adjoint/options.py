from thetis import *
from thetis.configuration import *

from adapt_utils.unsteady.options import CoupledOptions
from adapt_utils.unsteady.swe.utils import heaviside_approx
from thetis.options import ModelOptions2d
from adapt_utils.unsteady.sediment.sediments_model import SedimentModel
from adapt_utils.io import initialise_hydrodynamics

import os
import numpy as np
import matplotlib
# import matplotlib.pyplot as plt


matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif')


__all__ = ["TrenchSedimentOptions"]


class TrenchSedimentOptions(CoupledOptions):

    def __init__(self, diff_coeff, fric_coeff, friction='nik_solver', plot_timeseries=False, nx=1, ny=1, input_dir = None, output_dir = None, **kwargs):
        super(TrenchSedimentOptions, self).__init__(**kwargs)
        self.plot_timeseries = plot_timeseries
        self.default_mesh = RectangleMesh(np.int(16*5*nx), 5*ny, 16, 1.1)
        self.plot_pvd = True
        self.num_hours = 2.5 #15

        if output_dir is not None:
            self.di = output_dir

        # Physical
        self.base_viscosity = Constant(1e-6)
        self.wetting_and_drying = False
        self.solve_sediment = True
        self.solve_exner = True

        try:
            assert friction in ('nikuradse', 'manning', 'nik_solver')
        except AssertionError:
            raise ValueError("Friction parametrisation '{:s}' not recognised.".format(friction))
        self.friction = friction
        self.average_size = Constant(160e-6)  # Average sediment size
        self.friction_coeff = fric_coeff #0.025
        self.ksp = fric_coeff #Constant(3*self.average_size)
        self.norm_smoother = Constant(0.1)

        # Stabilisation
        self.stabilisation = 'lax_friedrichs'

        # Initial
        self.uv_init_tmp, self.elev_init_tmp = initialise_hydrodynamics(
            input_dir, outputdir=self.di, op=self, variant = None,
        )

        self.set_up_morph_model(input_dir, diff_coeff, self.default_mesh)

        self.morphological_acceleration_factor = Constant(100)

        # Time integration
        self.dt = 0.25
        self.end_time = self.num_hours*3600.0/float(self.morphological_acceleration_factor)
        self.dt_per_mesh_movement = 40
        self.dt_per_export = 40
        self.timestepper = 'CrankNicolson'
        self.implicitness_theta = 1.0
        self.family = 'dg-dg'


    def set_up_morph_model(self, input_dir, diff_coeff, mesh = None):

        # Physical
        self.base_diffusivity = diff_coeff #0.18011042551606954
        self.porosity = Constant(0.4)
        self.ks = self.friction_coeff #Constant(0.025)
        self.average_size = Constant(160*(10**(-6)))  # Average sediment size

        self.wetting_and_drying = False
        self.conservative = False
        self.slope_eff = True
        self.angle_correction = True
        self.suspended = True
        self.convective_vel_flag = False #True
        self.bedload = True


    def create_sediment_model(self, mesh, bathymetry):
        self.P1DG = FunctionSpace(mesh, "DG", 1)
        self.P1_vec_dg = VectorFunctionSpace(mesh, "DG", 1)

        self.uv_d = Function(self.P1_vec_dg)
        self.uv_d.assign(self.uv_init_tmp)

        self.eta_d = Function(self.P1DG)
        self.eta_d.assign(self.elev_init_tmp)

        self.sediment_model = SedimentModel(ModelOptions2d, suspendedload=self.suspended, convectivevel=self.convective_vel_flag,
            bedload=self.bedload, angle_correction=self.angle_correction, slope_eff=self.slope_eff, seccurrent=False,
            mesh2d=mesh, bathymetry_2d=bathymetry,
                            uv_init = self.uv_d, elev_init = self.eta_d, ks=self.ks, average_size=self.average_size,
                            cons_tracer = self.conservative, wetting_and_drying = self.wetting_and_drying)

    def set_quadratic_drag_coefficient(self, fs):
        self.depth = Function(fs).interpolate(self.set_bathymetry(fs) + Constant(0.397))
        if self.friction == 'nikuradse':
            return interpolate(self.get_cfactor(self.depth), fs)

    def get_cfactor(self, depth):
        try:
            assert hasattr(self, 'depth')
        except AssertionError:
            raise ValueError("Depth is undefined.")
        ksp = Constant(3*self.average_size)
        hc = conditional(depth > Constant(0.001), depth, Constant(0.001))
        aux = max_value(11.036*hc/ksp, 1.001)
        return 2*(0.4**2)/(ln(aux)**2)

    def set_bathymetry(self, fs):

        initial_depth = Constant(0.397)
        depth_riv = Constant(initial_depth - 0.397)
        depth_trench = Constant(depth_riv - 0.15)
        depth_diff = depth_trench - depth_riv
        x, y = SpatialCoordinate(fs.mesh())
        trench = conditional(le(x, 5), depth_riv, conditional(le(x, 6.5), (1/1.5)*depth_diff*(x-6.5) + depth_trench,
                             conditional(le(x, 9.5), depth_trench, conditional(le(x, 11), -(1/1.5)*depth_diff*(x-11) + depth_riv, depth_riv))))
        return interpolate(-trench, fs)

    #def set_viscosity(self, fs):
    #    self.viscosity = Function(fs)
    #    self.viscosity.assign(self.base_viscosity)
    #    return self.base_viscosity

    def set_boundary_conditions(self, prob, i):
        inflow_tag = 1
        outflow_tag = 2
        bottom_wall_tag = 3
        top_wall_tag = 4
        boundary_conditions = {
            'shallow_water': {
                inflow_tag: {'flux': Constant(-0.22)},
                outflow_tag: {'elev': Constant(0.397)},
            },
	   'sediment': {
                #inflow_tag: {'value': self.sediment_model.equiltracer}
            }
        }
        return boundary_conditions

    def set_initial_condition(self, prob):
        u, eta = prob.fwd_solutions[0].split()
        uv_init = Function(self.P1_vec_dg)
        uv_init.assign(self.uv_init_tmp)

        elev_init = Function(self.P1DG)
        elev_init.assign(self.elev_init_tmp)

        u.project(uv_init)
        eta.project(elev_init)

    def set_sediment_source(self, fs):
        if self.suspended:
            return self.sediment_model.ero_term
        else:
            return None

    def set_sediment_sink(self, fs):
        if self.suspended:
            return self.sediment_model.depo_term
        else:
            return None

    def set_advective_velocity_factor(self, fs):
        if self.convective_vel_flag:
            return self.sediment_model.corr_factor_model.corr_vel_factor
        else:
            return Constant(1.0)

    def set_initial_condition_sediment(self, prob):
        prob.fwd_solutions_sediment[0].interpolate(Constant(0.0)) #self.sediment_model.equiltracer)

    def set_initial_condition_bathymetry(self, prob):
        prob.fwd_solutions_bathymetry[0].interpolate(self.set_bathymetry(prob.fwd_solutions_bathymetry[0].function_space()))

    def get_update_forcings(self, prob, i, adjoint):
        return None

    def get_export_func(self, prob, i):
        eta_tilde = Function(prob.P1DG[i], name="Modified elevation")
        if self.plot_timeseries:
            u, eta = prob.fwd_solutions[i].split()
            b = prob.bathymetry[i]
            wd = Function(prob.P1DG[i], name="Heaviside approximation")

        def export_func():
            eta_tilde.project(self.get_eta_tilde(prob, i))
            u, eta = prob.fwd_solutions[i].split()

        return export_func

