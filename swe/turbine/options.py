from thetis import *
from thetis.configuration import *

import numpy as np

from adapt_utils.swe.options import ShallowWaterOptions


__all__ = ["SteadyTurbineOptions", "UnsteadyTurbineOptions"]


# Default: Newton with line search; solve linear system exactly with LU factorisation
lu_params = {
    'mat_type': 'aij',
    'snes_type': 'newtonls',
    'snes_rtol': 1e-3,
    'snes_atol': 1e-16,
    'snes_max_it': 100,
    'snes_linesearch_type': 'bt',
    'snes_monitor': None,
    'snes_converged_reason': None,
    'ksp_type': 'preonly',
    'ksp_converged_reason': None,
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
}
# TODO: 'Physics based' fieldsplit approach
default_params = lu_params
keys = {key for key in default_params if not 'snes' in key}
default_adjoint_params = {}
for key in keys:
    default_adjoint_params[key] = default_params[key]


class SteadyTurbineOptions(ShallowWaterOptions):
    """
    Base class holding parameters for steady state tidal turbine problems.
    """

    # Turbine parametrisation
    turbine_diameter = PositiveFloat(18.).tag(config=True)
    thrust_coefficient = NonNegativeFloat(0.8).tag(config=True)

    def __init__(self, num_iterations=1, bathymetry_space=None, **kwargs):
        self.base_bathymetry = 40.0
        self.set_bathymetry(bathymetry_space)
        super(SteadyTurbineOptions, self).__init__(**kwargs)
        self.timestepper = 'SteadyState'
        self.dt = 20.
        self.end_time = num_iterations*self.dt - 0.2

        # Solver parameters
        self.params = default_params
        self.adjoint_params = default_adjoint_params

        # Adaptivity
        self.h_min = 1e-5
        self.h_max = 500.0

    def set_viscosity(self, fs):
        self.viscosity.assign(self.base_viscosity)
        return self.viscosity

    def set_inflow(self, fs):
        """Should be implemented in derived class."""
        pass

    def get_max_depth(self):
        if hasattr(self, 'bathymetry'):
            if isinstance(self.bathymetry, Constant):
                self.max_depth = self.bathymetry.values()[0]
            elif isinstance(self.bathymetry, Function):
                self.max_depth = self.bathymetry.vector().gather().max()
            else:
                raise ValueError("Bathymetry format cannot be understood.")
        else:
            assert hasattr(self, 'base_bathymetry')
            self.max_depth = self.base_bathymetry

    def set_bathymetry(self, fs):
        self.bathymetry = Constant(self.base_bathymetry)
        return self.bathymetry

    def set_quadratic_drag_coefficient(self, fs):
        self.quadratic_drag_coefficient = Constant(0.0025)
        return self.quadratic_drag_coefficient

    def thrust_coefficient_correction(self):
        """
        Correction to account for the fact that the thrust coefficient is based on an upstream
        velocity whereas we are using a depth averaged at-the-turbine velocity (see Kramer and
        Piggott 2016, eq. (15))
        """
        self.get_max_depth()
        D = self.turbine_diameter
        A_T = pi*(D/2)**2
        correction = 4/(1+sqrt(1-A_T/(self.max_depth*D)))**2
        self.thrust_coefficient *= correction
        # NOTE: We're not yet correcting power output here, so that will be overestimated


class UnsteadyTurbineOptions(SteadyTurbineOptions):
    def __init__(self, **kwargs):
        super(UnsteadyTurbineOptions, self).__init__(**kwargs)
        self.timestepper = 'CrankNicolson'
