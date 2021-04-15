from thetis import *

<<<<<<< HEAD
from adapt_utils.steady.swe.solver import *


__all__ = ["SteadyTurbineProblem"]


class SteadyTurbineProblem(SteadyShallowWaterProblem):
    """
    General solver object for stationary tidal turbine problems.
    """
    def extra_setup(self):
        """
        We haven't meshed the turbines with separate ids, so define a farm everywhere and make it
        have a density of 1/D^2 inside the DxD squares where the turbines are and 0 outside.
        """
        op = self.op
        num_turbines = len(op.region_of_interest)
        scaling = num_turbines/assemble(op.bump(self.mesh)*dx)
        self.turbine_density = op.bump(self.mesh, scale=scaling)
        # scaling = num_turbines/assemble(op.box(self.P0)*dx)
        # self.turbine_density = op.box(self.P0, scale=scaling)
        self.farm_options = TidalTurbineFarmOptions()
        self.farm_options.turbine_density = self.turbine_density
        self.farm_options.turbine_options.diameter = op.turbine_diameter
        self.farm_options.turbine_options.thrust_coefficient = op.thrust_coefficient

        A_T = pi*(op.turbine_diameter/2.0)**2  # TODO: Why circular area?
        self.C_D = op.thrust_coefficient*A_T*self.turbine_density/2.0

        # Turbine drag is applied everywhere (where the turbine density isn't zero)
        self.solver_obj.options.tidal_turbine_farms["everywhere"] = self.farm_options

        # Callback that computes average power
        self.cb = turbines.TurbineFunctionalCallback(self.solver_obj)
        self.solver_obj.add_callback(self.cb, 'timestep')

    def extra_residual_terms(self):
        u, eta = self.solution.split()
        z, zeta = self.adjoint_solution.split()
        H = self.op.bathymetry + eta
        return -self.C_D*sqrt(dot(u, u))*inner(z, u)/H

    def extra_strong_residual_terms_momentum(self):
        u, eta = self.solution.split()
        H = self.op.bathymetry + eta
        return -self.C_D*sqrt(dot(u, u))*u/H

    def get_qoi_kernel(self):
        self.kernel = Function(self.V)
        u = self.solution.split()[0]
        k_u = self.kernel.split()[0]
        k_u.interpolate(Constant(1/3)*self.turbine_density*sqrt(inner(u, u))*u)

    def quantity_of_interest(self):
        self.qoi = self.cb.average_power
        return self.qoi

    def quantity_of_interest_form(self):
        return self.C_D*pow(inner(split(self.solution)[0], split(self.solution)[0]), 1.5)*dx
=======
from adapt_utils.io import index_string
from adapt_utils.steady.solver import AdaptiveSteadyProblem
from adapt_utils.swe.turbine.callback import PowerOutputCallback


# NOTE: DUPLICATED IN adapt_utils/swe/turbine/solver.py

__all__ = ["AdaptiveSteadyTurbineProblem"]


class AdaptiveSteadyTurbineProblem(AdaptiveSteadyProblem):
    """
    General solver object for adaptive steady-state turbine problems.
    """

    # --- Setup

    def __init__(self, *args, **kwargs):
        """
        :kwarg discrete_turbines: toggle whether to use a discrete or continuous representation
            for the turbine array.
        :kwarg thrust_correction: toggle whether to correct the turbine thrust coefficient.
        :kwarg smooth_indicators: use continuous approximations to discontinuous indicator functions.
        :kwarg remove_turbines: toggle whether turbines are present in the flow or not.
        :kwarg callback_dir: directory to save power output data to.
        """
        self.discrete_turbines = kwargs.pop('discrete_turbines', False)
        self.thrust_correction = kwargs.pop('thrust_correction', True)
        self.smooth_indicators = kwargs.pop('smooth_indicators', True)
        self.remove_turbines = kwargs.pop('remove_turbines', False)
        self.load_mesh = kwargs.pop('load_mesh', None)
        self.callback_dir = kwargs.pop('callback_dir', None)
        super(AdaptiveSteadyTurbineProblem, self).__init__(*args, **kwargs)

    def setup_all(self):
        super(AdaptiveSteadyTurbineProblem, self).setup_all()
        self.create_tidal_farms()

    def create_tidal_farms(self):
        """
        Create tidal farm objects *on each mesh*.

        If :attr:`discrete_turbines` is set to `True` then turbines are interpreted via cell tags
        which must be defined on the mesh at the start of the simulation. If set to `False` then
        turbines are interpreted as a continuous density field. That is, the position of an
        individual turbine corresponds to a region of high turbine density. This functionality
        mainly exists to smoothen the fields used in a turbine array optimisation loop. It is no
        longer needed for mesh adaptation, because Pragmatic now preserves cell tags across a mesh
        adaptation step.
        """
        if self.remove_turbines:
            return
        op = self.op
        op.print_debug("SETUP: Creating tidal turbine farms...")
        num_turbines = op.num_turbines
        self.farm_options = TidalTurbineFarmOptions()
        c_T = op.get_thrust_coefficient(correction=self.thrust_correction)
        if not self.discrete_turbines:
            shape = op.bump if self.smooth_indicators else op.box
        if hasattr(op, 'turbine_diameter'):
            D = op.turbine_diameter
            A_T = D**2
        else:
            D = max(op.turbine_length, op.turbine_width)
            A_T = op.turbine_length, op.turbine_width
            print_output("#### TODO: Account for non-square turbines")  # TODO
        if self.discrete_turbines:  # TODO: Use length and width
            self.turbine_density = Constant(1.0/D**2, domain=self.mesh)
        else:
            area = assemble(shape(self.mesh)*dx)
            self.turbine_density = shape(self.mesh, scale=num_turbines/area)

        self.farm_options.turbine_density = self.turbine_density
        self.farm_options.turbine_options.diameter = D
        self.farm_options.turbine_options.thrust_coefficient = c_T
        self.turbine_drag_coefficient = 0.5*c_T*A_T*self.turbine_density

        self.shallow_water_options[0].tidal_turbine_farms = {
            farm_id: self.farm_options for farm_id in op.farm_ids
        }

    # --- Quantity of Interest

    def add_callbacks(self, i):
        super(AdaptiveSteadyTurbineProblem, self).add_callbacks(i)
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
                tag += '_{:5s}'.format(index_string(i))
                self.qoi += self.callbacks[i]['timestep'][tag].timeseries[-1]
        return self.qoi

    def quantity_of_interest_form(self, i):
        """
        Power output quantity of interest expressed as a UFL form.
        """
        u, eta = split(self.fwd_solutions[i])
        return self.turbine_drag_coefficients[i]*pow(inner(u, u), 1.5)*dx
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
