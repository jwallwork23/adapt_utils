from thetis import *

from adapt_utils.io import index_string
from adapt_utils.swe.turbine.callback import PowerOutputCallback
from adapt_utils.unsteady.solver import AdaptiveProblem


# NOTE: DUPLICATED IN adapt_utils/steady/swe/turbine/solver.py


__all__ = ["AdaptiveTurbineProblem"]


class AdaptiveTurbineProblem(AdaptiveProblem):
    """General solver object for adaptive tidal turbine problems."""
    # TODO: doc

    # --- Setup

    def __init__(self, op, **kwargs):
        """
        :kwarg discrete_turbines: toggle whether to use a discrete or continuous representation
            for the turbine array.
        :kwarg thrust_correction: toggle whether to correct the turbine thrust coefficient.
        :kwarg smooth_indicators: use continuous approximations to discontinuous indicator functions.
        :kwarg remove_turbines: toggle whether turbines are present in the flow or not.
        :kwarg callback_dir: directory to save power output data to.
        """
        self.discrete_turbines = kwargs.pop('discrete_turbines', True)
        self.thrust_correction = kwargs.pop('thrust_correction', True)
        self.smooth_indicators = kwargs.pop('smooth_indicators', True)
        self.remove_turbines = kwargs.pop('remove_turbines', False)
        self.load_mesh = kwargs.pop('load_mesh', None)
        self.callback_dir = kwargs.pop('callback_dir', None)
        self.ramp_dir = kwargs.pop('ramp_dir', None)
        if self.ramp_dir is None and not op.spun:
            raise ValueError("Spin-up data directory not found.")
        super(AdaptiveTurbineProblem, self).__init__(op, **kwargs)

    def setup_all(self):
        super(AdaptiveTurbineProblem, self).setup_all()
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
        self.farm_options = [TidalTurbineFarmOptions() for i in range(self.num_meshes)]
        self.turbine_densities = [None for i in range(self.num_meshes)]
        self.turbine_drag_coefficients = [None for i in range(self.num_meshes)]
        c_T = op.get_thrust_coefficient(correction=self.thrust_correction)
        W, D = op.turbine_width, op.turbine_diameter
        footprint_area = D*W      # area of (horizontal) footprint
        swept_area = pi*(D/2)**2  # area of (vertical) cross-section
        shape = op.bump if self.smooth_indicators else op.box  # only used in continuous case
        for i, mesh in enumerate(self.meshes):
            if self.discrete_turbines:
                self.turbine_densities[i] = Constant(1.0/footprint_area, domain=self.meshes[i])
            else:
                area = assemble(shape(self.meshes[i])*dx)
                self.turbine_densities[i] = shape(self.meshes[i], scale=num_turbines/area)

            self.farm_options[i].turbine_density = self.turbine_densities[i]
            self.farm_options[i].turbine_options.diameter = D
            self.farm_options[i].turbine_options.thrust_coefficient = c_T
            self.turbine_drag_coefficients[i] = 0.5*c_T*swept_area*self.turbine_densities[i]

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

    def load_power_output_timeseries(self, di=None):
        """
        Load power output timeseries data stored in directory `di` into the :attr:`timeseries`
        attributes of the :attr:`callbacks`.
        """
        di = di or self.callback_dir
        for farm_id in self.op.farm_ids:
            for i in range(self.num_meshes):
                tag = 'power_output'
                if farm_id != 'everywhere':
                    tag += '_{:d}'.format(farm_id)
                fname = os.path.join(di, tag + '_{:5s}.npy'.format(index_string(i)))
                if not os.path.exists(fname):
                    raise IOError("Need to run the model in order to get power output timeseries.")
                self.callbacks[i]['timestep'][tag].timeseries = np.load(fname)

    def energy_output(self):
        """
        :returns: the total energy output over the entire simulation.
        """
        energy = 0.0
        for i in range(self.num_meshes):
            for farm_id in self.op.farm_ids:
                tag = 'power_output'
                if farm_id != 'everywhere':
                    tag += '_{:d}'.format(farm_id)
                tag += '_{:5s}'.format(index_string(i))
                energy += self.callbacks[i]['timestep'][tag].time_integrate()
        return energy*self.op.sea_water_density

    def average_power_output(self):
        """
        :returns: the average power output over the entire simulation.
        """
        return self.energy_output()/self.op.end_time

    def get_turbine_power_output_timeseries(self, farm_id):
        """
        :returns: the power output timeseries for a single turbine, with tag `farm_id`.
        """
        timeseries = np.array([])
        for i in range(self.num_meshes):
            tag = 'power_output'
            if farm_id != 'everywhere':
                tag += '_{:d}'.format(farm_id)
            tag += '_{:5s}'.format(index_string(i))
            timeseries = np.append(timeseries, self.callbacks[i]['timestep'][tag].timeseries)
        return timeseries*self.op.sea_water_density

    def get_power_output_timeseries(self):
        """
        :returns: the power output timeseries for the entire array.
        """
        all_timeseries = []
        for farm_id in self.op.farm_ids:
            all_timeseries.append(get_turbine_power_output_timeseries(farm_id))
        return sum(all_timeseries)

    def peak_power_output(self):
        """
        :returns: the peak power output over the entire simulation.
        """
        timeseries = self.get_power_output_timeseries()
        i = np.argmax(self.qoi_timeseries)
        times = np.linspace(0, self.op.end_time, len(timeseries))
        return timeseries[i], times[i]

    def quantity_of_interest(self):
        """By default, the QoI is set to the average power output over the entire simulation."""
        self.qoi = self.energy_output()/self.op.end_time
        return self.qoi

    def quantity_of_interest_form(self, i):
        """Power output quantity of interest expressed as a UFL form."""
        u, eta = split(self.fwd_solutions[i])
        return self.turbine_drag_coefficients[i]*pow(inner(u, u), 1.5)*dx

    # --- Restarts

    def set_initial_condition(self):
        if self.op.spun:
            self.load_state(0, self.ramp_dir)
            if self.load_mesh is not None:
                tmp = self.fwd_solutions[0].copy(deepcopy=True)
                u_tmp, eta_tmp = tmp.split()
                self.set_meshes(self.load_mesh)
                self.setup_all()
                u, eta = self.fwd_solutions[0].split()
                u.project(u_tmp)
                eta.project(eta_tmp)
        else:
            super(AdaptiveTurbineProblem, self).set_initial_condition()
