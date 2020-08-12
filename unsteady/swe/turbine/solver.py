from thetis import *

from adapt_utils.unsteady.solver import AdaptiveProblem
# from adapt_utils.unsteady.callback import QoICallback
from adapt_utils.unsteady.swe.turbine.callback import PowerOutputCallback


__all__ = ["AdaptiveTurbineProblem"]


class AdaptiveTurbineProblem(AdaptiveProblem):
    """General solver object for adaptive tidal turbine problems."""

    # --- Setup

    def __init__(self, *args, remove_turbines=False, discrete_turbines=False, **kwargs):
        """
        :kwarg remove_turbines: toggle whether turbines are present in the flow or not.
        :kwarg discrete_turbines: toggle whether to use a discrete or continuous representation
            for the turbine array.
        :kwarg thrust_correction: toggle whether to correct the turbine thrust coefficient.
        """
        super(AdaptiveTurbineProblem, self).__init__(*args, **kwargs)
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
                self.shallow_water_options[i].tidal_turbine_farms = {
                    'everywhere': self.farm_options[i],
                }
                self.turbine_drag_coefficients[i] = 0.5*c_T*A_T*self.turbine_densities[i]

    def set_fields(self):
        """Set velocity field, viscosity, etc *on each mesh*."""
        self.fields = [AttrDict() for mesh in self.meshes]
        for i, P1 in enumerate(self.P1):
            self.fields[i].update({
                'quadratic_drag_coefficient': self.op.set_quadratic_drag_coefficient(P1),
                'manning_drag_coefficient': self.op.set_manning_drag_coefficient(P1),
            })
        for i, P1DG in enumerate(self.P1DG):
            self.fields[i].update({
                'horizontal_viscosity': self.op.set_viscosity(P1),
            })
        self.bathymetry = [self.op.set_bathymetry(P1) for P1 in self.P1]
        self.depth = [None for bathymetry in self.bathymetry]
        for i, bathymetry in enumerate(self.bathymetry):
            # NOTE: DepthExpression is the modified version from `unsteady/swe/utils`.
            self.depth[i] = DepthExpression(
                bathymetry,
                use_nonlinear_equations=self.shallow_water_options[i].use_nonlinear_equations,
                use_wetting_and_drying=self.shallow_water_options[i].use_wetting_and_drying,
                wetting_and_drying_alpha=self.shallow_water_options[i].wetting_and_drying_alpha,
            )

    # --- Quantity of Interest

    def add_callbacks(self, i):
        super(AdaptiveTurbineProblem, self).add_callbacks(i)
        # self.get_qoi_kernels(i)
        # self.callbacks[i].add(QoICallback(self, i), 'timestep')
        for farm_id in self.shallow_water_options[i].tidal_turbine_farms:
            self.callbacks[i].add(PowerOutputCallback(self, i, farm_id), 'timestep')

    def quantity_of_interest(self):
        # self.qoi = sum(c['timestep']['qoi'].time_integrate() for c in self.callbacks)
        # self.qoi = sum(c['timestep']['power_output'].time_integrate() for c in self.callbacks)
        self.qoi = sum(c['timestep']['power_output_everywhere'].time_integrate() for c in self.callbacks)
        return self.qoi

    def quantity_of_interest_form(self, i):
        u, eta = split(self.fwd_solutions[i])
        return self.turbine_drag_coefficients[i]*pow(inner(u, u), 1.5)*dx
