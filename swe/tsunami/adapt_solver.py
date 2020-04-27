from thetis import *

import os

from adapt_utils.swe.adapt_solver import AdaptiveShallowWaterProblem


__all__ = ["AdaptiveTsunamiProblem"]


class AdaptiveTsunamiProblem(AdaptiveShallowWaterProblem):
    """General solver object for adaptive tsunami propagation problems."""

    # -- Setup

    def __init__(self, *args, extension=None, **kwargs):
        self.extension = extension
        super(AdaptiveTsunamiProblem, self).__init__(*args, **kwargs)
        self.callbacks = [{} for mesh in self.meshes]

        # Use linearised equations
        self.shallow_water_options['use_nonlinear_equations'] = False

        # Don't bother plotting velocity
        self.io_options['fields_to_export'] = ['elev_2d'] if self.op.plot_pvd else []
        self.io_options['fields_to_export_hdf5'] = ['elev_2d'] if self.op.save_hdf5 else []

    def set_fields(self):
        self.fields = []
        self.bathymetry = []
        for i in range(self.num_meshes):
            self.fields.append({
                'horizontal_viscosity': self.op.set_viscosity(self.P1[i]),
                'coriolis_frequency': self.op.set_coriolis(self.P1[i]),
                'quadratic_drag_coefficient': self.op.set_quadratic_drag_coefficient(self.P1DG[i]),
                'manning_drag_coefficient': self.op.set_manning_drag_coefficient(self.P1[i]),
            })
            self.bathymetry.append(self.op.set_bathymetry())

    # FIXME: Continuity of callbacks across mesh steps
    def add_callbacks(self, i):
        op = self.op

        # --- Callbacks

        # Gauge timeseries
        names = [g for g in op.gauges]
        locs = [op.gauges[g]["coords"] for g in names]
        fname = "gauges"
        if self.extension is not None:
            fname = '_'.join([fname, self.extension])
        fname = '_'.join([fname, str(self.num_cells[-1])])
        self.callbacks[i]['gauges'] = callback.DetectorsCallback(
            self.fwd_solvers[i], locs, ['elev_2d'], fname, names)
        self.fwd_solvers[i].add_callback(self.callbacks[i]['gauges'], 'export')
        # for g in names:
        #     x, y = op.gauges[g]["coords"]
        #     self.callbacks[i][g] = callback.TimeSeriesCallback2D(
        #         self.fwd_solvers[i], ['elev_2d'], x, y, g, self.di)
        #     self.fwd_solvers[i].add_callback(self.callbacks[i][g], 'export')

        # Quantity of interest
        if not hasattr(self, 'kernel'):
            self.get_qoi_kernels()
        kernel_file = File(os.path.join(self.di, 'kernel_mesh{:d}.pvd'.format(i)))
        kernel_file.write(self.kernels[i].split()[1])
        kt = Constant(0.0)  # Kernel in time

        def qoi(sol):
            t = self.fwd_solvers[i].simulation_time
            kt.assign(1.0 if t >= op.start_time else 0.0)
            return assemble(kt*inner(self.kernels[i], sol)*dx)

        self.callbacks[i]["qoi"] = callback.TimeIntegralCallback(
            qoi, self.fwd_solvers[i], self.fwd_solvers[i].timestepper, name="qoi", append_to_log=op.debug)
        self.fwd_solvers[i].add_callback(self.callbacks[i]["qoi"], 'timestep')
