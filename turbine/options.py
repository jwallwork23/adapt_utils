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
        self.lax_friedrichs = True
        self.drag_coefficient = Constant(0.0025)

        # Adaptivity
        self.h_min = 1e-5
        self.h_max = 500.0

    def set_viscosity(self, fs):
        self.viscosity.assign(self.base_viscosity)
        return self.viscosity

    def set_inflow(self, fs):
        self.inflow = Function(fs).interpolate(as_vector([3., 0.]))
        return self.inflow

    def thrust_coefficient_correction(self):
        """
        Correction to account for the fact that the thrust coefficient is based on an upstream
        velocity whereas we are using a depth averaged at-the-turbine velocity (see Kramer and
        Piggott 2016, eq. (15))
        """
        D = self.turbine_diameter
        A_T = pi*(D/2)**2
        correction = 4/(1+sqrt(1-A_T/(40.*D)))**2
        self.thrust_coefficient *= correction
        # NOTE: We're not yet correcting power output here, so that will be overestimated

    def set_bcs(self, fs):
        pass


# TODO: bring up to date
class UnsteadyTurbineOptions(SteadyTurbineOptions):
    def __init__(self, approach='fixed_mesh'):
        super(UnsteadyTurbineOptions, self).__init__(approach)

        # Time period and discretisation
        self.dt = 3
        self.timestepper = 'CrankNicolson'
        self.T_tide = 1.24*3600
        self.T_ramp = 1*self.T_tide
        self.end_time = self.T_ramp+2*self.T_tide
        self.dt_per_export = 10
        self.dt_per_remesh = 10  # FIXME: solver seems to go out of sync if this != dt_per_export

        # Boundary forcing
        self.hmax = 0.5
        self.omega = 2*pi/self.T_tide

        # Turbines
        self.base_viscosity = 3.
        self.thrust_coefficient = 7.6

    def set_boundary_surface(self, fs):
        self.elev_in = Function(fs)
        self.elev_out = Function(fs)
