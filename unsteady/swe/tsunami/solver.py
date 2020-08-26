from thetis import PointNotInDomainError, print_output

import numpy as np
import os
import h5py

from adapt_utils.unsteady.callback import *
from adapt_utils.unsteady.solver import AdaptiveProblem


__all__ = ["AdaptiveTsunamiProblem"]


class AdaptiveTsunamiProblem(AdaptiveProblem):
    """
    General solver object for adaptive tsunami propagation problems which exists to hook up
    callbacks appropriately. The default callbacks are:
      * Time-integrate the functional quantity of interest;
      * Extract free surface elevation timeseries for all gauges stored in the :class:`Options`
        parameter class.
    """
    def __init__(self, *args, nonlinear=False, **kwargs):
        super(AdaptiveTsunamiProblem, self).__init__(*args, nonlinear=nonlinear, **kwargs)
        try:
            assert not (self.op.solve_tracer or self.op.solve_sediment or self.op.solve_exner)
        except AssertionError:
            raise ValueError("This class is for problems with hydrodynamics only.")

    def add_callbacks(self, i):
        super(AdaptiveTsunamiProblem, self).add_callbacks(i)
        # gauges = list(self.op.gauges.keys())
        # for gauge in gauges:
        #     self.callbacks[i].add(GaugeCallback(self, i, gauge), 'export')
        self.get_qoi_kernels(i)
        self.callbacks[i].add(QoICallback(self, i), 'timestep')

    def get_qoi_timeseries(self):
        self.qoi_timeseries = []
        for c in self.callbacks:
            self.qoi_timeseries.extend(c['timestep']['qoi'].timeseries)
        return self.qoi_timeseries

    def quantity_of_interest(self):
        self.get_qoi_timeseries()
        self.qoi = sum(self.qoi_timeseries)
        return self.qoi

    # def save_gauge_data(self, fname):
    #     fname = "diagnostic_gauges_{:s}.hdf5".format(fname)
    #     with h5py.File(os.path.join(self.di, fname), 'w') as f:
    #         for gauge in self.op.gauges:
    #             timeseries = []
    #             for i in range(self.num_meshes):
    #                 timeseries.extend(self.callbacks[i]['export'][gauge].timeseries)
    #             f.create_dataset(gauge, data=np.array(timeseries))
    #         f.create_dataset("num_cells", data=np.array(self.num_cells[-1]))
