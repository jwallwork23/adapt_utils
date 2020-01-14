from thetis import *
from thetis.configuration import *

import numpy as np

from adapt_utils.swe.options import ShallowWaterOptions


__all__ = ["SteadyTurbineOptions", "UnsteadyTurbineOptions"]


# Default: Newton with line search; solve linear system exactly with LU factorisation
default_params = {
    'mat_type': 'aij',
    'snes_type': 'newtonls',
    'snes_rtol': 1e-3,
    'snes_atol': 1e-16,
    'snes_max_it': 100,
    'snes_linesearch_type': 'bt',
    'snes_monitor': None,
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
}
keys = {key for key in default_params if not 'snes' in key}
default_adjoint_params = {}
for key in keys:
    default_adjoint_params[key] = default_params[key]


class SteadyTurbineOptions(ShallowWaterOptions):
    """
    Base class holding parameters for steady state tidal turbine problems.
    """

    # Solver parameters
    params = PETScSolverParameters(default_params).tag(config=True)
    adjoint_params = PETScSolverParameters(default_adjoint_params).tag(config=True)

    def __init__(self, approach='fixed_mesh', num_iterations=1):
        super(SteadyTurbineOptions, self).__init__(approach)
        self.timestepper = 'SteadyState'
        self.dt = 20.
        self.end_time = num_iterations*self.dt - 0.2
        self.bathymetry = Constant(40.0)
        self.viscosity = Constant(self.base_viscosity)
        self.drag_coefficient = Constant(0.0025)

        # Adaptivity
        self.h_min = 1e-5
        self.h_max = 500.0

    def set_viscosity(self, fs):
        self.viscosity.assign(self.base_viscosity)
        return self.viscosity

    def set_inflow(self, fs):
        self.inflow = interpolate(as_vector([3., 0.]), fs)
        return self.inflow

    def get_max_depth(self):
        assert hasattr(self, 'bathymetry')
        if isinstance(self.bathymetry, Constant):
            self.max_depth = self.bathymetry.values()[0]
        elif isinstance(self.bathymetry, Function):
            self.max_depth = self.bathymetry.vector().gather().max()
        else:
            raise ValueError("Bathymetry format cannot be understood.")

    def thrust_coefficient_correction(self):
        """
        Correction to account for the fact that the thrust coefficient is based on an upstream
        velocity whereas we are using a depth averaged at-the-turbine velocity (see Kramer and
        Piggott 2016, eq. (15))
        """
        if not hasattr(self, 'max_depth'):
            self.get_max_depth()
        D = self.turbine_diameter
        A_T = pi*(D/2)**2
        correction = 4/(1+sqrt(1-A_T/(self.max_depth*D)))**2
        self.thrust_coefficient *= correction
        # NOTE: We're not yet correcting power output here, so that will be overestimated


class UnsteadyTurbineOptions(SteadyTurbineOptions):
    def __init__(self, approach='fixed_mesh'):
        super(UnsteadyTurbineOptions, self).__init__(approach)
        self.timestepper = 'CrankNicolson'

    def set_boundary_surface(self, fs):
        self.elev_in = Function(fs)
        self.elev_out = Function(fs)
