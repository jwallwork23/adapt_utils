from thetis import *
from thetis.configuration import *

from adapt_utils.options import Options


__all__ = ["ShallowWaterOptions"]


# TODO: Improve doc
class ShallowWaterOptions(Options):
    """Parameters for coupled shallow water - tracer transport model."""

    # Physics
    base_viscosity = NonNegativeFloat(0.0).tag(config=True)
    base_diffusivity = NonNegativeFloat(0.0).tag(config=True)
    g = FiredrakeScalarExpression(Constant(9.81)).tag(config=True)
    friction = Unicode(None, allow_none=True).tag(config=True)
    friction_coeff = NonNegativeFloat(None, allow_none=True).tag(config=True)

    # Common model
    implicitness_theta = NonNegativeFloat(0.5).tag(config=True)

    # Shallow water model
    solve_swe = Bool(True).tag(config=True)
    family = Enum(['dg-dg', 'rt-dg', 'dg-cg', 'cg-cg'], default_value='dg-dg').tag(config=True)
    grad_div_viscosity = Bool(False).tag(config=True)
    grad_depth_viscosity = Bool(False).tag(config=True)
    wetting_and_drying = Bool(False).tag(config=True)
    wetting_and_drying_alpha = FiredrakeScalarExpression(Constant(4.3)).tag(config=True)
    lax_friedrichs_velocity_scaling_factor = FiredrakeScalarExpression(None, allow_none=True).tag(config=True)
    sipg_parameter = FiredrakeScalarExpression(None, allow_none=True).tag(config=True)

    # Tracer transport model
    solve_tracer = Bool(False).tag(config=True)
    tracer_family = Enum(['dg', 'cg'], default_value='dg').tag(config=True)
    lax_friedrichs_tracer_scaling_factor = FiredrakeScalarExpression(None, allow_none=True).tag(config=True)
    sipg_parameter_tracer = FiredrakeScalarExpression(None, allow_none=True).tag(config=True)

    # Adaptation
    adapt_field = Unicode('all_avg', help="Adaptation field of interest.").tag(config=True)
    region_of_interest = List(default_value=[]).tag(config=True)

    def __init__(self, **kwargs):
        self.degree_increase = 0
        super(ShallowWaterOptions, self).__init__(**kwargs)

    def set_bathymetry(self, fs):
        raise NotImplementedError("Should be implemented in derived class.")

    def set_viscosity(self, fs):
        """Should be implemented in derived class."""
        return None if np.allclose(self.base_viscosity, 0.0) else Constant(self.base_viscosity)

    def set_source_tracer(self, fs):
        """Should be implemented in derived class."""
        ero = None
        depo = None
        return ero, depo

    def set_diffusivity(self, fs):
        """Should be implemented in derived class."""
        return None if np.allclose(self.base_diffusivity, 0.0) else Constant(self.base_diffusivity)

    def set_inflow(self, fs):
        """Should be implemented in derived class."""
        return

    def set_coriolis(self, fs):
        """Should be implemented in derived class."""
        return

    def set_quadratic_drag_coefficient(self, fs):
        """Should be implemented in derived class."""
        return

    def set_manning_drag_coefficient(self, fs):
        if self.friction == 'manning':
            return Constant(self.friction_coeff or 0.02)

    def get_eta_tilde(self, prob, i):
        u, eta = prob.fwd_solutions[i].split()
        if not self.wetting_and_drying:
            return eta
        bathymetry_displacement = prob.equations[i].shallow_water.depth.wd_bathymetry_displacement
        return eta + bathymetry_displacement(eta)

    def get_export_func(self, prob, i):
        if self.wetting_and_drying:
            eta_tilde = Function(prob.P1DG[i], name="Modified elevation")
            self.eta_tilde_file._topology = None

            def export_func():
                eta_tilde.project(self.get_eta_tilde(prob, i))
                self.eta_tilde_file.write(eta_tilde)

            return export_func
