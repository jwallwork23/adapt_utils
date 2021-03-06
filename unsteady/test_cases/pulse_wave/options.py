from thetis import *
from thetis.configuration import *
from thetis.options import ModelOptions2d

import numpy as np

from adapt_utils.options import CoupledOptions
from adapt_utils.sediment.sediments_model import SedimentModel


__all__ = ["BeachOptions"]


class BeachOptions(CoupledOptions):
    """
    Parameters for test case adapted from [1].

    [1] Roberts, W. et al. "Investigation using simple mathematical models of
    the effect of tidal currents and waves on the profile shape of intertidal
    mudflats." Continental Shelf Research 20.10-11 (2000): 1079-1097.
    """
    def __init__(self, plot_timeseries=False, nx=1, ny=1, input_dir=None, output_dir=None, **kwargs):
        super(BeachOptions, self).__init__(**kwargs)
        friction = kwargs.get('friction', 'manning')
        try:
            assert friction in ('nikuradse', 'manning')
        except AssertionError:
            raise ValueError("Friction parametrisation '{:s}' not recognised.".format(friction))
        self.friction = friction
        self.friction_coeff = 0.02

        # Mesh
        self.lx = 220
        self.ly = 10
        self.default_mesh = RectangleMesh(np.int(220*nx), np.int(10*ny), self.lx, self.ly)

        # I/O
        if output_dir is not None:
            self.di = output_dir
        self.plot_timeseries = plot_timeseries
        self.plot_pvd = True

        # Initialisation
        self.elev_init = Constant(0.0)
        self.uv_init = as_vector((10**(-4), 0.0))
        self.tracer_init = Constant(0.0)

        # Models
        self.set_up_morph_model(self.default_mesh)
        self.hessian_recovery = 'L2'
        self.grad_depth_viscosity = True
        self.stabilisation = 'lax_friedrichs'
        self.stabilisation_sediment = 'lax_friedrichs'

        # Boundary conditions
        h_amp = 0.25  # Ocean boundary forcing amplitude
        v_amp = 0.5   # Ocean boundary foring velocity
        omega = 0.5   # Ocean boundary forcing frequency
        self.morphological_acceleration_factor = Constant(125)
        self.ocean_elev_func = lambda t: (h_amp*np.cos(-omega*t)) if (t % (8*np.pi) > 4*np.pi and t % (8*np.pi) < 6*np.pi) else 0
        self.ocean_vel_func = lambda t: (v_amp*np.cos(-omega*t)) if (t % (8*np.pi) > 4*np.pi and t % (8*np.pi) < 6*np.pi) else 10**(-4)

        # Time integration
        self.num_hours = kwargs.get('num_hours', 48)  # 288*4
        self.dt = 0.05
        self.end_time = float(self.num_hours*3600.0/self.morphological_acceleration_factor)
        self.dt_per_mesh_movement = kwargs.get('dt_per_mesh_movement', 54)
        self.dt_per_export = kwargs.get('dt_per_export', 54)
        self.timestepper = 'CrankNicolson'
        self.implicitness_theta = 1.0

        # # Timeseries  # TODO: Remove?
        # self.wd_obs = []
        # self.trange = np.linspace(0.0, self.end_time, self.num_hours+1)
        # tol = 1e-8  # FIXME: Point evaluation hack
        # self.xrange = np.linspace(tol, 16-tol, 20)

    def set_up_morph_model(self, mesh=None):

        # Physics
        self.base_viscosity = 0.5
        self.base_diffusivity = 100
        self.gravity = Constant(9.81)
        self.porosity = Constant(0.4)
        self.ks = Constant(0.025)
        self.average_size = 0.0002  # Average sediment size

        self.wetting_and_drying = True
        self.depth_integrated = True
        self.use_tracer_conservative_form = True
        self.slope_eff = True
        self.angle_correction = False
        self.suspended = False
        self.convective_vel_flag = False
        self.bedload = True
        self.solve_sediment = False
        self.solve_exner = True

        self.norm_smoother = Constant(10/25)

        # P1 = FunctionSpace(mesh, "CG", 1)
        # bath = self.set_bathymetry(P1)
        self.wetting_and_drying_alpha = Constant(1/40)  # bath.dx(0)

    def create_sediment_model(self, mesh, bathymetry):
        self.P1DG = FunctionSpace(mesh, "DG", 1)
        self.P1_vec_dg = VectorFunctionSpace(mesh, "DG", 1)
        # TODO: Do these really need to be attributes? Would prefer to avoid stashing func spaces

        self.uv_d = Function(self.P1_vec_dg).project(self.uv_init)
        self.eta_d = Function(self.P1DG).project(self.elev_init)

        self.sediment_model = SedimentModel(  # TODO: Use kwargs dict format
            ModelOptions2d,
            suspendedload=self.suspended,
            convectivevel=self.convective_vel_flag,
            bedload=self.bedload,
            angle_correction=self.angle_correction,
            slope_eff=self.slope_eff,
            seccurrent=False,
            mesh2d=mesh,
            bathymetry_2d=bathymetry,
            uv_init=self.uv_d,
            elev_init=self.eta_d,
            ks=self.ks,
            average_size=self.average_size,
            cons_tracer=self.use_tracer_conservative_form,
            wetting_and_drying=self.wetting_and_drying,
            wetting_alpha=self.wetting_and_drying_alpha
        )

    def set_manning_drag_coefficient(self, fs):
        if self.friction == 'manning':
            if hasattr(self, 'friction_coeff'):
                self.manning_drag_coefficient = Constant(self.friction_coeff)
            else:
                self.manning_drag_coefficient = Constant(0.02)
        return self.manning_drag_coefficient

    def set_bathymetry(self, fs, **kwargs):
        x, y = SpatialCoordinate(fs.mesh())
        self.bathymetry = Function(fs, name="Bathymetry")
        self.bathymetry.interpolate(4 - x/40)
        return self.bathymetry

    def set_viscosity(self, fs):
        x, y = SpatialCoordinate(fs.mesh())
        sponge = conditional(x >= 10000, -39999 + 4*x, Constant(1.0))
        self.viscosity = interpolate(self.base_viscosity*sponge, fs)
        return self.viscosity  # Constant(self.base_viscosity)

    def set_boundary_conditions(self, prob, i):
        if not hasattr(self, 'elev_in'):
            self.elev_in = Constant(0.0)
        if not hasattr(self, 'vel_in'):
            self.vel_in = Constant(as_vector((0.0, 0.0)))
        self.elev_in.assign(self.ocean_elev_func(0.0))
        vel_const = Constant(self.ocean_vel_func(0.0))
        self.vel_in.assign(as_vector((vel_const, 0.0)))

        inflow_tag = 1
        # outflow_tag = 2
        # bottom_wall_tag = 3
        # top_wall_tag = 4
        boundary_conditions = {
            'shallow_water': {
                inflow_tag: {'elev': self.elev_in, 'uv': self.vel_in},
            },
            'sediment': {},
        }
        return boundary_conditions

    def update_boundary_conditions(self, solver_obj, t=0.0):
        self.elev_in.assign(self.ocean_elev_func(t))
        vel_const = Constant(self.ocean_vel_func(t))
        self.vel_in.assign(as_vector((vel_const, 0.0)))

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
        fs = prob.fwd_solutions_bathymetry[0].function_space()
        prob.fwd_solutions_bathymetry[0].interpolate(self.set_bathymetry(fs))

    def get_update_forcings(self, prob, i, adjoint):

        def update_forcings(t):
            self.update_boundary_conditions(prob, t=t)

        return update_forcings

    def get_export_func(self, prob, i):
        eta_tilde = Function(prob.P1DG[i], name="Modified elevation")
        # self.eta_tilde_file = File(self.di + "/eta_tilde.pvd").write(eta_tilde)
        # self.eta_tilde_file._topology = None
        if self.plot_timeseries:
            u, eta = prob.fwd_solutions[i].split()
            # b = prob.bathymetry[i]
            # wd = Function(prob.P1DG[i], name="Heaviside approximation")

        def export_func():
            eta_tilde.project(self.get_eta_tilde(prob, i))
            # self.eta_tilde_file.write(eta_tilde)
            u, eta = prob.fwd_solutions[i].split()
            # if self.plot_timeseries:

            #    # Store modified bathymetry timeseries
            #    wd.project(heaviside_approx(-eta-b, self.wetting_and_drying_alpha))
            #    self.wd_obs.append([wd.at([x, 0]) for x in self.xrange])

        return export_func

    def set_boundary_surface(self):
        """Set the initial displacement of the boundary elevation."""
        pass
