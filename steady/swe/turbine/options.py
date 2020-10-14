from thetis import *
from thetis.configuration import *

from adapt_utils.unsteady.swe.turbine.options import TurbineOptions  # TODO: Don't refer to unsteady


__all__ = ["TurbineOptions"]


# Default: Newton with line search; solve linear system exactly with LU factorisation
lu_params = {
    'mat_type': 'aij',
    'snes_type': 'newtonls',
    'snes_rtol': 1e-8,
    'snes_max_it': 100,
    'snes_linesearch_type': 'bt',
    'snes_monitor': None,
    'snes_converged_reason': None,
    'ksp_type': 'preonly',
    'ksp_converged_reason': None,
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
}
default_params = lu_params


class SteadyTurbineOptions(TurbineOptions):
    """
    Base class holding parameters for steady state tidal turbine problems.
    """

    # Turbine parameters
    turbine_diameter = PositiveFloat(18.0).tag(config=True)
    thrust_coefficient = NonNegativeFloat(0.8).tag(config=True)

    # --- Setup

    def __init__(self, num_iterations=1, **kwargs):
        super(SteadyTurbineOptions, self).__init__(**kwargs)

        # Timestepping
        self.timestepper = 'SteadyState'
        self.dt = 20.0
        self.dt_per_export = 1
        self.end_time = num_iterations*self.dt - 0.2

        # Solver parameters
        self.solver_parameters = {'shallow_water': default_params}
        self.adjoint_solver_parameters = {'shallow_water': default_params}
