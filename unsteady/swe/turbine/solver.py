from thetis import *

from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.swe.turbine.callback import PowerOutputCallback


__all__ = ["AdaptiveTurbineProblem"]


class AdaptiveTurbineProblem(AdaptiveProblem):
    """General solver object for adaptive tidal turbine problems."""

    # --- Setup

    def __init__(self, *args, remove_turbines=False, discrete_turbines=False, callback_dir=None, **kwargs):
        """
        :kwarg remove_turbines: toggle whether turbines are present in the flow or not.
        :kwarg discrete_turbines: toggle whether to use a discrete or continuous representation
            for the turbine array.
        :kwarg thrust_correction: toggle whether to correct the turbine thrust coefficient.
        """
        super(AdaptiveTurbineProblem, self).__init__(*args, **kwargs)
        self.callback_dir = callback_dir
        if remove_turbines:
            return
        op = self.op
        num_turbines = op.num_turbines

        # Set up tidal farm object
        if discrete_turbines:
            raise NotImplementedError  # TODO
        else:
            smooth_indicators = kwargs.get('smooth_indicators', True)
            shape = op.bump if smooth_indicators else op.box
            self.farm_options = [TidalTurbineFarmOptions() for mesh in self.meshes]
            self.turbine_densities = [None for mesh in self.meshes]
            self.turbine_drag_coefficients = [None for mesh in self.meshes]
            c_T = op.get_thrust_coefficient(correction=kwargs.get('thrust_correction', True))
            if hasattr(op, 'turbine_diameter'):
                D = op.turbine_diameter
                A_T = D**2
            else:
                D = max(op.turbine_length, op.turbine_width)
                A_T = op.turbine_length, op.turbine_width
                print_output("#### TODO: Account for non-square turbines")  # TODO
            for i, mesh in enumerate(self.meshes):
                area = assemble(shape(self.meshes[i])*dx)
                self.turbine_densities[i] = shape(self.meshes[i], scale=num_turbines/area)
                self.farm_options[i].turbine_density = self.turbine_densities[i]
                self.farm_options[i].turbine_options.diameter = D
                self.farm_options[i].turbine_options.thrust_coefficient = c_T
                self.turbine_drag_coefficients[i] = 0.5*c_T*A_T*self.turbine_densities[i]

                self.shallow_water_options[i].tidal_turbine_farms = {
                    farm_id: self.farm_options[i] for farm_id in op.farm_ids
                }

    # --- Quantity of Interest

    def add_callbacks(self, i):
        super(AdaptiveTurbineProblem, self).add_callbacks(i)
        di = self.callback_dir
        for farm_id in self.shallow_water_options[i].tidal_turbine_farms:
            self.callbacks[i].add(PowerOutputCallback(self, i, farm_id, callback_dir=di), 'timestep')

    def quantity_of_interest(self):
        self.qoi = 0.0
        for i in range(self.num_meshes):
            for farm_id in self.shallow_water_options[i].tidal_turbine_farms:
                self.qoi += self.callbacks[i]['timestep'][farm_id].time_integrate()
        return self.qoi

    def quantity_of_interest_form(self, i):
        u, eta = split(self.fwd_solutions[i])
        return self.turbine_drag_coefficients[i]*pow(inner(u, u), 1.5)*dx
