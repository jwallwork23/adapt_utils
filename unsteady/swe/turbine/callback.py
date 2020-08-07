from thetis import *

from adapt_utils.unsteady.callback import TimeseriesCallback


__all__ = ["PowerOutputCallback"]


class PowerOutputCallback(TimeseriesCallback):
    """
    Callback for evaluating the power output of all turbines in an array which have the label
    `farm_id`. In the discrete turbine case, this will typically correspond to individual turbines.
    However, in the continuous turbine case, the label 'everywhere' is used to refer to the entire
    array.

    Note that the integral of power is energy.
    """
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

        label = "power_output"
        if farm_id != "everywhere":
            label += "_{:}".format(farm_id)
        super(PowerOutputCallback, self).__init__(prob, power_output, i, label)
