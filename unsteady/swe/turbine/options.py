from thetis import *
from thetis.configuration import *

from adapt_utils.unsteady.options import CoupledOptions


__all__ = ["TurbineOptions"]


class TurbineOptions(CoupledOptions):
    # TODO: doc

    # Turbine parameters
    turbine_length = PositiveFloat(18.0).tag(config=False)

    def __init__(self, **kwargs):
        super(TurbineOptions, self).__init__(**kwargs)

        # Timestepping
        self.timestepper = 'CrankNicolson'

        # Boundary forcing
        self.M2_tide_period = 12.4*3600.0
        self.T_tide = self.M2_tide_period
        self.dt_per_export = 10

    def set_quadratic_drag_coefficient(self, fs):
        return Constant(self.friction_coeff)

    def set_manning_drag_coefficient(self, fs):
        return

    def thrust_coefficient_correction(self):
        """
        Correction to account for the fact that the thrust coefficient is based on an upstream
        velocity whereas we are using a depth averaged at-the-turbine velocity (see Kramer and
        Piggott 2016, eq. (15))
        """
        if hasattr(self, 'turbine_diameter'):
            D = self.turbine_diameter
        else:
            D = max(self.turbine_length, self.turbine_width)
        A_T = pi*(D/2)**2
        correction = 4/(1+sqrt(1-A_T/(self.max_depth()*D)))**2
        self.thrust_coefficient *= correction
        # NOTE: We're not yet correcting power output here, so that will be overestimated

    def max_depth(self):
        """Compute maximum depth from bathymetry field."""
        if hasattr(self, 'bathymetry'):
            if isinstance(self.bathymetry, Constant):
                return self.bathymetry.values()[0]
            elif isinstance(self.bathymetry, Function):
                return self.bathymetry.vector().gather().max()
            else:
                raise ValueError("Bathymetry format cannot be understood.")
        else:
            assert hasattr(self, 'base_bathymetry')
            return self.base_bathymetry
