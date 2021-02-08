from adapt_utils.io import index_string
from adapt_utils.unsteady.callback import QoICallback
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

    def add_callbacks(self, i, **kwargs):
        super(AdaptiveTsunamiProblem, self).add_callbacks(i, **kwargs)
        self.get_qoi_kernels(i)
        self.callbacks[i].add(QoICallback(self, i), 'timestep')

    def get_qoi_timeseries(self):
        self._qoi_timeseries = []
        for i in range(self.num_meshes):
            tag = 'qoi_{:5s}'.format(index_string(i))
            self._qoi_timeseries.extend(self.callbacks[i]['timestep'][tag].timeseries)

    @property
    def qoi_timeseries(self):
        if not hasattr(self, '_qoi_timeseries'):
            self.get_qoi_timeseries()
        return self._qoi_timeseries

    def quantity_of_interest(self):
        self.qoi = sum(self.qoi_timeseries)
        return self.qoi
