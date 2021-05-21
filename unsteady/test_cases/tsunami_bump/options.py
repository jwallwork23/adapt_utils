from thetis import *
from thetis.configuration import *

from adapt_utils.unsteady.options import CoupledOptions
from adapt_utils.unsteady.swe.utils import heaviside_approx
from thetis.options import ModelOptions2d
from adapt_utils.unsteady.sediment.sediments_model import SedimentModel

import os
import time
import datetime
import numpy as np
from matplotlib import rc

rc('text', usetex=True)


__all__ = ["BeachOptions"]


class BeachOptions(CoupledOptions):
    """
    Parameters for tsunami bump test case adapted from test case in [1]
    
    [1] Kobayashi et al. "Cross-shore sediment transport under breaking solitary waves." Journal of Geophysical Research: Oceans 109 (2004).    
    """

    def __init__(self, friction='quadratic', nx=1, ny=1, mesh = None, input_dir = None, output_dir = None, **kwargs):

        super(BeachOptions, self).__init__(**kwargs)

        try:
            assert friction in ('nikuradse', 'manning', 'quadratic')
        except AssertionError:
            raise ValueError("Friction parametrisation '{:s}' not recognised.".format(friction))
        self.friction = friction

        self.lx = 30
        self.ly = 8

        if output_dir is not None:
            self.di = output_dir

        self.default_mesh = RectangleMesh(np.int(self.lx*nx), np.int(self.ly*ny), self.lx, self.ly)

        self.set_up_morph_model(self.default_mesh)
        # Initial
        self.elev_init = Constant(-0.0025)
        self.uv_init = as_vector((1e-10, 0.0))

        self.plot_pvd = True
        self.hessian_recovery = 'dL2'

        self.grad_depth_viscosity = True

        # Stabilisation
        self.stabilisation = 'lax_friedrichs'

        self.morphological_acceleration_factor = Constant(4)

        # Boundary conditions
        H = 0.216
        h = 0.8
        C = 3.16
        eta_down = -0.0025
        tmax = 3.9
        self.tsunami_elev_func = lambda t: H*(1/cosh(sqrt((3*H)/(4*h))*(C/h)*(t-tmax)))**2 + eta_down

        # Time integration
        if nx > 8:
            self.dt = 0.01
        else:
            self.dt = 0.025
        self.end_time = 20
        self.dt_per_mesh_movement = 20
        self.dt_per_export = 20
        self.timestepper = 'CrankNicolson'
        self.implicitness_theta = 1.0

        # Goal-Oriented
        self.qoi_mode = 'inundation_volume'

    def set_up_morph_model(self, mesh = None):

        # Physical
        self.base_diffusivity = 1
        self.gravity = Constant(9.81)
        self.porosity = Constant(0.4)
        self.ks = Constant(0.00054)
        self.average_size = 1.8e-4  # Average sediment size
        self.max_angle = 20
        self.meshgrid_size = 0.2

        self.wetting_and_drying = True
        self.sediment_slide = True
        self.depth_integrated = True
        self.use_tracer_conservative_form = True
        self.slope_eff = False
        self.angle_correction = False
        self.suspended = True
        self.convective_vel_flag = True
        self.bedload = False
        self.solve_sediment = True
        self.solve_exner = True

        self.norm_smoother = Constant(1/12*0.2)

        P1 = FunctionSpace(mesh, "CG", 1)
        bath = self.set_bathymetry(P1)
        self.wetting_and_drying_alpha = Constant(2/12*0.2)

    def create_sediment_model(self, mesh, bathymetry):
        self.P1DG = FunctionSpace(mesh, "DG", 1)
        self.P1_vec_dg = VectorFunctionSpace(mesh, "DG", 1)

        self.uv_d = Function(self.P1_vec_dg).project(self.uv_init)
        self.eta_d = Function(self.P1DG).project(self.elev_init)

        self.sediment_model = SedimentModel(ModelOptions2d, suspendedload=self.suspended, convectivevel=self.convective_vel_flag,
            bedload=self.bedload, angle_correction=self.angle_correction, slope_eff=self.slope_eff, seccurrent=False, sediment_slide = self.sediment_slide,
            mesh2d=mesh, bathymetry_2d=bathymetry, dt = self.dt,
                            uv_init = self.uv_d, elev_init = self.eta_d, ks=self.ks, average_size=self.average_size, porosity = self.porosity, max_angle = self.max_angle, meshgrid_size = self.meshgrid_size,
                            cons_tracer = self.use_tracer_conservative_form, morfac = self.morphological_acceleration_factor, wetting_and_drying = self.wetting_and_drying, wetting_alpha = self.wetting_and_drying_alpha)


    def set_quadratic_drag_coefficient(self, fs):
        self.quadratic_drag_coefficient = Constant(9.81/(65**2))
        return self.quadratic_drag_coefficient

    def set_bathymetry(self, fs, **kwargs):
        x, y = SpatialCoordinate(fs.mesh())
        self.bathymetry = Function(fs, name="Bathymetry")
        beach_profile = -x/12 + 131/120

        self.bathymetry.interpolate(conditional(y < 3, conditional(x<3.5, Constant(0.8), beach_profile),
                                    conditional(y < 5, conditional(x<3.5, Constant(0.8),
                                    conditional(x < 6, beach_profile, conditional(x < 8, Constant(0.2), beach_profile))),
                                                             conditional(x<3.5, Constant(0.8), beach_profile))))
        return self.bathymetry

    def set_viscosity(self, fs):
        x, y = SpatialCoordinate(fs.mesh())
        self.viscosity = Constant(0.8)
        return self.viscosity

    def set_boundary_conditions(self, prob, i):
        if not hasattr(self, 'elev_in'):
            self.elev_in = Constant(0.0)
        self.elev_in.assign(self.tsunami_elev_func(0.0))

        inflow_tag = 1
        outflow_tag = 2
        bottom_wall_tag = 3
        top_wall_tag = 4
        boundary_conditions = {
            'shallow_water': {
                inflow_tag: {'elev': self.elev_in},
            },
	   'sediment': {
            }
        }
        return boundary_conditions

    def update_boundary_conditions(self, solver_obj, t=0.0):
        self.elev_in.assign(self.tsunami_elev_func(t))

    def set_initial_condition(self, prob):
        u, eta = prob.fwd_solutions[0].split()
        u.project(self.uv_init)
        eta.project(self.elev_init)

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

    def set_advective_velocity_factor(self, fs):
        if self.convective_vel_flag:
            return self.sediment_model.corr_factor_model.corr_vel_factor
        else:
            return Constant(1.0)

    def set_initial_condition_sediment(self, prob):
        prob.fwd_solutions_sediment[0].interpolate(Constant(0.0))

    def set_initial_condition_bathymetry(self, prob):
        prob.fwd_solutions_bathymetry[0].interpolate(self.set_bathymetry(prob.fwd_solutions_bathymetry[0].function_space()))

    def get_update_forcings(self, prob, i, adjoint):

        def update_forcings(t):
            uv, elev = prob.fwd_solutions[0].split()
            self.update_boundary_conditions(prob, t=t)

        return update_forcings

    def get_export_func(self, prob, i):
        return None

    def set_boundary_surface(self):
        """Set the initial displacement of the boundary elevation."""
        pass
