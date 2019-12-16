from thetis import *
from thetis.configuration import *

from adapt_utils.options import Options


__all__ = ["ShallowWaterOptions"]


class ShallowWaterOptions(Options):
    """
    Parameters for shallow water solver.
    """

    # Physical
    bathymetry = FiredrakeScalarExpression(Constant(1.0)).tag(config=True)
    base_viscosity = NonNegativeFloat(None, allow_none=True).tag(config=True)
    viscosity = FiredrakeScalarExpression(Constant(0.0)).tag(config=True)
    drag_coefficient = FiredrakeScalarExpression(None, allow_none=True).tag(config=True)
    g = FiredrakeScalarExpression(Constant(9.81)).tag(config=True)

    # Model
    grad_div_viscosity = Bool(False).tag(config=True)
    grad_depth_viscosity = Bool(False).tag(config=True)
    lax_friedrichs = Bool(True).tag(config=True)
    lax_friedrichs_scaling_factor = FiredrakeConstantTraitlet(Constant(1.0)).tag(config=True)
    family = Enum(['dg-dg', 'rt-dg', 'dg-cg'], default_value='dg-dg').tag(config=True)

    # Adaptation
    adapt_field = Unicode('speed_avg_elevation', help="Adaptation field of interest.").tag(config=True)
    region_of_interest = List(default_value=[]).tag(config=True)

    def set_viscosity(self, fs):
        pass

    def set_initial_condition(self, fs):
        pass

    def set_coriolis(self, fs):
        pass

    def set_bcs(self, fs):
        pass

    def set_qoi_kernel(self, fs):
        pass
