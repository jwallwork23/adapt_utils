from thetis import AttrDict

from adapt_utils.io import index_string
from adapt_utils.tracer.desalination.callback import DesalinationOutfallCallback
from adapt_utils.unsteady.solver import AdaptiveProblem


__all__ = ["AdaptiveDesalinationProblem"]


class AdaptiveDesalinationProblem(AdaptiveProblem):
    """General solver object for adaptive desalination outfall problems."""
    # TODO: doc

    # --- Setup

    def __init__(self, *args, **kwargs):
        super(AdaptiveDesalinationProblem, self).__init__(*args, **kwargs)
        self.callback_dir = kwargs.get('callback_dir', self.op.di)
        self.ramp_dir = kwargs.get('ramp_dir')
        self.load_mesh = kwargs.get('load_mesh')

    def _get_fields_for_tracer_timestepper(self, i):
        u, eta = self.fwd_solutions[i].split()  # FIXME: Not fully annotated
        fields = AttrDict({
            'elev_2d': eta,
            'uv_2d': u,
            'diffusivity_h': self.fields[i].horizontal_diffusivity,
            'source': self.fields[i].tracer_source_2d,
            'tracer_advective_velocity_factor': self.fields[i].tracer_advective_velocity_factor,
            'lax_friedrichs_tracer_scaling_factor': self.tracer_options[i].lax_friedrichs_tracer_scaling_factor,
        })
        if self.stabilisation_tracer == 'lax_friedrichs':
            fields['lax_friedrichs_tracer_scaling_factor'] = self.tracer_options[i].lax_friedrichs_tracer_scaling_factor
        return fields

    # --- Quantity of Interest

    def add_callbacks(self, i):
        super(AdaptiveDesalinationProblem, self).add_callbacks(i)
        cb = DesalinationOutfallCallback(self, i, callback_dir=self.callback_dir)
        self.callbacks[i].add(cb, 'timestep')

    def load_qoi_timeseries(self, di=None):
        """
        Load power output timeseries data stored in directory `di` into the :attr:`timeseries`
        attributes of the :attr:`callbacks`.
        """
        di = di or self.callback_dir
        tag = 'inlet_salinity_diff'
        for i in range(self.num_meshes):
            fname = os.path.join(di, '{:s}_{:s}.npy'.format(tag, index_string(i)))
            if not os.path.exists(fname):
                raise IOError("Need to run the model in order to get QoI timeseries.")
            self.callbacks[i]['timestep'][tag].timeseries = np.load(fname)

    def quantity_of_interest(self):
        """
        Time integrate inlet salinty differential.
        """
        self.qoi = 0.0
        for i in range(self.num_meshes):
            tag = 'inlet_salinity_diff_{:s}'.format(index_string(i))
            self.qoi += self.callbacks[i]['timestep'][tag].time_integrate()
        return self.qoi

    # --- Restarts

    def set_initial_condition(self):
        if self.op.spun:

            # Load hydrodynamics from file
            self.op.solve_tracer = False
            self.load_state(0, self.ramp_dir)
            if self.load_mesh is not None:
                tmp = self.fwd_solutions[0].copy(deepcopy=True)
                u_tmp, eta_tmp = tmp.split()
                self.set_meshes(self.load_mesh)
                self.setup_all()
                u, eta = self.fwd_solutions[0].split()
                u.project(u_tmp)
                eta.project(eta_tmp)

            # Set background salinity
            self.op.solve_tracer = True
            self.fwd_solution_tracer.assign(self.op.background_salinity)
        else:
            super(AdaptiveDesalinationProblem, self).set_initial_condition()
