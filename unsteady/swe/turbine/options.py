from thetis import *
from thetis.configuration import *

from adapt_utils.unsteady.options import CoupledOptions


__all__ = ["TurbineOptions"]


class TurbineOptions(CoupledOptions):
    # TODO: doc

    # Turbine parameters
    turbine_length = PositiveFloat(18.0).tag(config=False)
    thrust_coefficient = NonNegativeFloat(7.6).tag(config=True)

    def __init__(self, **kwargs):
        super(TurbineOptions, self).__init__(**kwargs)

        # Timestepping
        self.timestepper = 'CrankNicolson'

        # Boundary forcing
        self.M2_tide_period = 12.4*3600.0
        self.T_tide = self.M2_tide_period
        self.dt_per_export = 10

    def set_qoi_kernel(self, prob, i):
        prob.kernels[i] = Function(prob.V[i])
        u, eta = prob.fwd_solutions[i].split()
        k_u, k_eta = prob.kernels[i].split()
        k_u.interpolate(Constant(1/3)*prob.turbine_densities[i]*sqrt(inner(u, u))*u)

    def set_quadratic_drag_coefficient(self, fs):
        return Constant(self.friction_coeff)

    def set_manning_drag_coefficient(self, fs):
        return

    def get_thrust_coefficient(self, correction=True):
        """
        Correction to account for the fact that the thrust coefficient is based on an upstream
        velocity whereas we are using a depth averaged at-the-turbine velocity (see Kramer and
        Piggott 2016, eq. (15))
        """
        if not correction:
            return self.thrust_coefficient
        if hasattr(self, 'turbine_diameter'):
            D = self.turbine_diameter
        else:
            D = max(self.turbine_length, self.turbine_width)
        A_T = pi*(D/2)**2
        correction = 4/(1+sqrt(1-A_T/(self.max_depth*D)))**2
        return self.thrust_coefficient*correction
        # NOTE: We're not yet correcting power output here, so that will be overestimated

    def get_max_depth(self, bathymetry=None):
        """Compute maximum depth from bathymetry field."""
        if bathymetry is not None:
            if isinstance(bathymetry, Constant):
                self.max_depth = bathymetry.values()[0]
            elif isinstance(bathymetry, Function):
                self.max_depth = self.bathymetry.vector().gather().max()
            else:
                raise ValueError("Bathymetry format cannot be understood.")
        else:
            assert hasattr(self, 'base_bathymetry')
            self.max_depth = self.base_bathymetry
