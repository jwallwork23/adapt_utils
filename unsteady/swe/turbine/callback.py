from thetis import *

from adapt_utils.unsteady.callback import TimeseriesCallback


__all__ = ["PowerOutputCallback"]


class PowerOutputCallback(TimeseriesCallback):
    # TODO: doc

    def __init__(self, prob, i, farm_id):
        self.name = "power output {:}".format(farm_id)
        u, eta = split(prob.fwd_solutions[i])
        # u, eta = prob.fwd_solutions[i].split()
        dt = prob.op.dt

        # Turbine farm object
        farm_options = prob.shallow_water_options[i].tidal_turbine_farms[farm_id]
        self.farm = turbines.TurbineFarm(farm_options, farm_id, u[0], u[1], dt)

        # Power output functional
        power_output = lambda t: self.farm.evaluate_timestep()[0]

        super(PowerOutputCallback, self).__init__(prob, power_output, i, "power_output")
