from thetis import *
from thetis.configuration import *

from adapt_utils.options import Options


__all__ = ["ShallowWaterOptions"]


class ShallowWaterOptions(Options):
    """
    Parameters for shallow water solver.
    """
    solve_tracer = Bool(False).tag(config=True)

    # Physical
    bathymetry = FiredrakeScalarExpression(Constant(1.0)).tag(config=True)
    base_viscosity = NonNegativeFloat(0.0).tag(config=True)
    base_diffusivity = NonNegativeFloat(0.0).tag(config=True)
    viscosity = FiredrakeScalarExpression(Constant(0.0)).tag(config=True)
    diffusivity = FiredrakeScalarExpression(Constant(0.0)).tag(config=True)
    quadratic_drag_coefficient = FiredrakeScalarExpression(None, allow_none=True).tag(config=True)
    manning_drag_coefficient = FiredrakeScalarExpression(None, allow_none=True).tag(config=True)
    inflow = FiredrakeVectorExpression(None, allow_none=True).tag(config=True)
    g = FiredrakeScalarExpression(Constant(9.81)).tag(config=True)
    coriolis = FiredrakeScalarExpression(None, allow_none=True).tag(config=True)

    # Model
    grad_div_viscosity = Bool(False).tag(config=True)
    grad_depth_viscosity = Bool(False).tag(config=True)
    family = Enum(['dg-dg', 'rt-dg', 'dg-cg'], default_value='dg-dg').tag(config=True)
    wetting_and_drying = Bool(False).tag(config=True)
    wetting_and_drying_alpha = FiredrakeScalarExpression(Constant(0.0)).tag(config=True)

    # Adaptation
    adapt_field = Unicode('all_avg', help="Adaptation field of interest.").tag(config=True)
    region_of_interest = List(default_value=[]).tag(config=True)

    def __init__(self, **kwargs):
        super(ShallowWaterOptions, self).__init__(**kwargs)
        self.degree_increase = 0
        self.stabilisation = 'lax_friedrichs'
        self.stabilisation_parameter = Constant(1.0)

    def set_bathymetry(self, fs):
        raise NotImplementedError("Should be implemented in derived class.")

    def set_viscosity(self, fs):
        """Should be implemented in derived class."""
        self.viscosity = Constant(self.base_viscosity)
        return self.viscosity

    def set_diffusivity(self, fs):
        """Should be implemented in derived class."""
        self.diffusivity = Constant(self.base_diffusivity)
        return self.diffusivity

    def set_inflow(self, fs):
        """Should be implemented in derived class."""
        pass

    def set_coriolis(self, fs):
        """Should be implemented in derived class."""
        pass

    def set_quadratic_drag_coefficient(self, fs):
        """Should be implemented in derived class."""
        pass

    def set_manning_drag_coefficient(self, fs):
        """Should be implemented in derived class."""
        pass

    def get_initial_depth(self, fs):
        """Compute the initial total water depth, using the bathymetry and initial elevation."""
        if not hasattr(self, 'bathymetry'):
            self.set_bathymetry(fs.sub(1))
        if not hasattr(self, 'initial_value'):
            self.set_initial_value(fs)
        eta = self.initial_value.split()[1]
        self.depth = self.bathymetry + eta
        return self.depth

    def set_boundary_surface(self):
        """Set the initial displacement of the boundary elevation."""
        self.elev_in = Constant(0.0)
        self.elev_out = Constant(0.0)
