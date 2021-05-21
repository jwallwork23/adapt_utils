from thetis import *
from thetis.configuration import *
from thetis.options import ModelOptions2d

from adapt_utils.io import initialise_hydrodynamics
from adapt_utils.options import CoupledOptions
from adapt_utils.sediment.sediments_model import SedimentModel


__all__ = ["TrenchSedimentOptions"]


class TrenchSedimentOptions(CoupledOptions):
    """
    Parameters for the (effectively) 1D migrating trench test case introduced in [Van Rijn 1980].

    [Van Rijn 1980] Leo C Van Rijn, "Storm surge barrier Oosterschelde-computation of siltation in
        dredged trenches: Semi-empirical model for the flow in dredged trenches", Delft, the
        Netherlands, 1980.
    """
    def __init__(self, friction='nik_solver', nx=1, ny=1, input_dir=None, output_dir=None, **kwargs):
        self.timestepper = 'CrankNicolson'
        super(TrenchSedimentOptions, self).__init__(**kwargs)
        self.default_mesh = RectangleMesh(np.int(16*5*nx), 5*ny, 16, 1.1)
        self.plot_pvd = True
        self.num_hours = 15
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
        self.friction_coeff = Constant(0.025)
        self.ksp = Constant(3*self.average_size)
        self.norm_smoother = Constant(0.1)

        # Stabilisation
        self.stabilisation = 'lax_friedrichs'
        self.stabilisation_sediment = 'lax_friedrichs'

        # Initialisation
        self.uv_init, self.elev_init = initialise_hydrodynamics(
            input_dir, outputdir=output_dir, op=self,
        )
        self.set_up_morph_model(input_dir, self.default_mesh)
        self.morphological_acceleration_factor = Constant(100)

        # Time integration
        self.dt = 0.25 if nx < 4 else 0.1
        self.end_time = self.num_hours*3600.0/float(self.morphological_acceleration_factor)
        self.dt_per_mesh_movement = 40
        self.dt_per_export = 40
        self.implicitness_theta = 1.0
        self.family = 'dg-dg'

    def set_up_morph_model(self, input_dir, mesh=None):
        self.base_diffusivity = Constant(0.18011042551606954)
        self.porosity = Constant(0.4)
        self.ks = Constant(0.025)

        self.wetting_and_drying = False
        self.conservative = False
        self.slope_eff = True
        self.angle_correction = True
        self.suspended = True
        self.convective_vel_flag = True
        self.bedload = True

    def create_sediment_model(self, mesh, bathymetry):
        self.P1DG = FunctionSpace(mesh, "DG", 1)
        self.P1_vec_dg = VectorFunctionSpace(mesh, "DG", 1)

        self.uv_d = Function(self.P1_vec_dg).project(self.uv_init)

        self.eta_d = Function(self.P1DG).project(self.elev_init)
        self.sediment_model = SedimentModel(
            ModelOptions2d, suspendedload=self.suspended, convectivevel=self.convective_vel_flag,
            bedload=self.bedload, angle_correction=self.angle_correction,
            slope_eff=self.slope_eff, seccurrent=False, mesh2d=mesh, bathymetry_2d=bathymetry,
            uv_init=self.uv_d, elev_init=self.eta_d, ks=self.ks, average_size=self.average_size,
            cons_tracer=self.conservative, wetting_and_drying=self.wetting_and_drying
        )

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
        trench = conditional(
            le(x, 5), depth_riv, conditional(
                le(x, 6.5), (1/1.5)*depth_diff*(x-6.5) + depth_trench, conditional(
                    le(x, 9.5), depth_trench, conditional(
                        le(x, 11), -(1/1.5)*depth_diff*(x-11) + depth_riv, depth_riv))))
        return interpolate(-trench, fs)

    def set_boundary_conditions(self, prob, i):
        inflow_tag = 1
        outflow_tag = 2
        boundary_conditions = {
            'shallow_water': {
                inflow_tag: {'flux': Constant(-0.22)},
                outflow_tag: {'elev': Constant(0.397)},
            },
            'sediment': {
                inflow_tag: {'value': self.sediment_model.equiltracer}
            }
        }
        return boundary_conditions

    def set_initial_condition(self, prob):
        u, eta = prob.fwd_solutions[0].split()
        u.project(self.uv_init)
        eta.project(self.elev_init)

    def set_advective_velocity_factor(self, fs):
        if self.convective_vel_flag:
            return self.sediment_model.corr_factor_model.corr_vel_factor
        else:
            return Constant(1.0)

    def set_initial_condition_sediment(self, prob):
        prob.fwd_solutions_sediment[0].interpolate(Constant(0.0))

    def set_initial_condition_bathymetry(self, prob):
        prob.fwd_solutions_bathymetry[0].interpolate(
            self.set_bathymetry(prob.fwd_solutions_bathymetry[0].function_space())
        )

    def get_export_func(self, prob, i):
        eta_tilde = Function(prob.P1DG[i], name="Modified elevation")

        def export_func():
            eta_tilde.project(self.get_eta_tilde(prob, i))
            u, eta = prob.fwd_solutions[i].split()

        return export_func
