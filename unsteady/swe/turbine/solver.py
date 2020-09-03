from thetis import *

from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.swe.turbine.callback import PowerOutputCallback


__all__ = ["AdaptiveTurbineProblem"]


class AdaptiveTurbineProblem(AdaptiveProblem):
    """General solver object for adaptive tidal turbine problems."""

    # --- Setup

    def __init__(self, *args, remove_turbines=False, callback_dir=None, **kwargs):
        """
        :kwarg remove_turbines: toggle whether turbines are present in the flow or not.
        :kwarg callback_dir: directory to save power output data to.
        """
        super(AdaptiveTurbineProblem, self).__init__(*args, **kwargs)
        self.callback_dir = callback_dir
        if not remove_turbines:
            self.create_tidal_farms(**kwargs)

    def create_tidal_farms(self, discrete_turbines=False, thrust_correction=True, smooth_indicators=True):
        """
        Create tidal farm objects *on each mesh*.

        :kwarg discrete_turbines: toggle whether to use a discrete or continuous representation
            for the turbine array.
        :kwarg thrust_correction: toggle whether to correct the turbine thrust coefficient.
        :kwarg smooth_indicators: use continuous approximations to discontinuous indicator functions.
        """
        op = self.op
        op.print_debug("SETUP: Creating tidal turbine farms...")
        num_turbines = op.num_turbines
        self.farm_options = [TidalTurbineFarmOptions() for i in range(self.num_meshes)]
        self.turbine_densities = [None for i in range(self.num_meshes)]
        self.turbine_drag_coefficients = [None for i in range(self.num_meshes)]
        c_T = op.get_thrust_coefficient(correction=thrust_correction)
        if not discrete_turbines:
            shape = op.bump if smooth_indicators else op.box
        if hasattr(op, 'turbine_diameter'):
            D = op.turbine_diameter
            A_T = D**2
        else:
            D = max(op.turbine_length, op.turbine_width)
            A_T = op.turbine_length, op.turbine_width
            print_output("#### TODO: Account for non-square turbines")  # TODO
        for i, mesh in enumerate(self.meshes):
            if discrete_turbines:  # TODO: Use length and width
                self.turbine_densities[i] = Constant(1.0/D**2, domain=self.meshes[i])
            else:
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
        if di is None:
            return
        for farm_id in self.shallow_water_options[i].tidal_turbine_farms:
            self.callbacks[i].add(PowerOutputCallback(self, i, farm_id, callback_dir=di), 'timestep')

    def quantity_of_interest(self):
        self.qoi = 0.0
        for i in range(self.num_meshes):
            for farm_id in self.shallow_water_options[i].tidal_turbine_farms:
                tag = 'power_output'
                if farm_id != 'everywhere':
                    tag += '_{:d}'.format(farm_id)
                self.qoi += self.callbacks[i]['timestep'][tag].time_integrate()
        return self.qoi

    def quantity_of_interest_form(self, i):
        u, eta = split(self.fwd_solutions[i])
        return self.turbine_drag_coefficients[i]*pow(inner(u, u), 1.5)*dx
