import numpy as np
import os
import h5py

from adapt_utils.callback import *
from adapt_utils.swe.adapt_solver import AdaptiveShallowWaterProblem


__all__ = ["AdaptiveTsunamiProblem"]


class AdaptiveTsunamiProblem(AdaptiveShallowWaterProblem):
    """
    General solver object for adaptive tsunami propagation problems which exists to hook up
    callbacks appropriately. The default callbacks are:
      * Time-integrate the functional quantity of interest;
      * Extract free surface elevation timeseries for all gauges stored in the :class:`Options`
        parameter class.
    """
    # TODO: Is `extension` redundant?
    def __init__(self, *args, extension=None, nonlinear=False, **kwargs):
        self.extension = extension
        super(AdaptiveTsunamiProblem, self).__init__(*args, nonlinear=nonlinear, **kwargs)

    def add_callbacks(self, i):
        for gauge in self.op.gauges:
            self.callbacks[i].add(GaugeCallback(self, i, gauge), 'export')
        self.get_qoi_kernels(i)
        self.callbacks[i].add(QoICallback(self, i), 'timestep')

    def quantity_of_interest(self):
        self.qoi = sum(c['timestep']['qoi'].time_integrate() for c in self.callbacks)
        return self.qoi

    def save_gauge_data(self, fname):
        fname = "diagnostic_gauges_{:s}.hdf5".format(fname)
        with h5py.File(os.path.join(self.di, fname), 'w') as f:
            for gauge in self.op.gauges:
                timeseries = []
                for i in range(self.num_meshes):
                    timeseries.extend(self.callbacks[i]['export'][gauge].timeseries)
                f.create_dataset(gauge, data=np.array(timeseries))
            f.create_dataset("num_cells", data=np.array(self.num_cells[-1]))
