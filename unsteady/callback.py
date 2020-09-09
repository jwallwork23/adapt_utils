from thetis import *

import numpy as np
import os

from adapt_utils.io import index_string
from adapt_utils.unsteady.swe.utils import L2ProjectorVorticity


__all__ = ["VelocityNormCallback", "ElevationNormCallback", "TracerNormCallback",
           "SedimentNormCallback", "ExnerNormCallback", "QoICallback", "GaugeCallback"]


class TimeseriesCallback(object):
    """
    Generic callback object for storing timseries extracted during a
    simulation and integrating in time.
    """
    def __init__(self, prob, func, i, name, callback_dir=None):
        """
        :arg prob: :class:`AdaptiveProblem` object.
        :arg func: user-provided function to be evaluated.
        :arg i: mesh index.
        :arg name: name for the callback object.
        :kwarg callback_dir: if provided, timeseries is saved to that location every export.
        """
        self.prob = prob
        self.func = func
        self.name = '_'.join([name, index_string(i)])
        self.timeseries = []
        if not hasattr(self, 'label'):
            self.label = self.name
        self.msg = "    {:24s}".format(self.label) + " at time {:6.1f} = {:11.4e}"
        self.callback_dir = callback_dir

    def evaluate(self, **kwargs):
        t = self.prob.simulation_time
        value = self.func(t)
        self.prob.print(self.msg.format(t, value))
        self.timeseries.append(value)
        if self.callback_dir is not None:
            np.save(os.path.join(self.callback_dir, self.name), self.timeseries)

    def time_integrate(self):
        N = len(self.timeseries)
        op = self.prob.op
        assert N >= 2
        if op.timestepper != 'CrankNicolson':
            raise NotImplementedError  # TODO
        val = 0.5*op.dt*(self.timeseries[0] + self.timeseries[-1])
        for i in range(1, N-1):
            val += op.dt*self.timeseries[i]
        return val


class VelocityNormCallback(TimeseriesCallback):
    """Callback for evaluating the L2 norm of the velocity field at each timestep/export."""
    def __init__(self, prob, i):
        self.label = "velocity norm"
        u, eta = prob.fwd_solutions[i].split()
        super(VelocityNormCallback, self).__init__(prob, lambda t: norm(u), i, "velocity_norm")


class ElevationNormCallback(TimeseriesCallback):
    """Callback for evaluating the L2 norm of the elevation field at each timestep/export."""
    def __init__(self, prob, i):
        self.label = "elevation norm"
        u, eta = prob.fwd_solutions[i].split()
        super(ElevationNormCallback, self).__init__(prob, lambda t: norm(eta), i, "elevation_norm")


class TracerNormCallback(TimeseriesCallback):
    """Callback for evaluating the L2 norm of the tracer concentration at each timestep/export."""
    def __init__(self, prob, i):
        self.label = "tracer norm"
        c = prob.fwd_solutions_tracer[i]
        super(TracerNormCallback, self).__init__(prob, lambda t: norm(c), i, "tracer_norm")


class SedimentNormCallback(TimeseriesCallback):
    """Callback for evaluating the L2 norm of the sediment at each timestep/export."""
    def __init__(self, prob, i):
        self.label = "sediment norm"
        s = prob.fwd_solutions_sediment[i]
        super(SedimentNormCallback, self).__init__(prob, lambda t: norm(s), i, "sediment_norm")


class ExnerNormCallback(TimeseriesCallback):
    """Callback for evaluating the L2 norm of the modified bathymetry at each timestep/export."""
    def __init__(self, prob, i):
        self.label = "bathymetry norm"
        b = prob.fwd_solutions_bathymetry[i]
        super(ExnerNormCallback, self).__init__(prob, lambda t: norm(b), i, "bathymetry_norm")


class QoICallback(TimeseriesCallback):
    r"""
    Callback for evaluating functional quantities of interest of the form

  ..math::
        \int_0^T\int_\Omega\mathbf k\cdot\mathbf q \;\mathrm dx\;\mathrm dt,

    where :math:`\mathbf k=\mathbf k(\mathbf x,t)` is a kernel function and
    :math:`\mathbf q=\mathbf q(\mathbf x,t)` is the prognostic solution
    tuple for the forward problem.
    """
    def __init__(self, prob, i):
        """
        :arg prob: :class:`AdaptiveProblem` object.
        :arg i: mesh index.
        """
        self.label = "quantity of interest"
        ks = prob.kernels[i]  # Kernel in space
        kt = Constant(0.0)    # Kernel in time
        sol = prob.fwd_solutions[i]

        def functional(t):
            kt.assign(1.0 if t >= prob.op.start_time else 0.0)
            return assemble(kt*inner(ks, sol)*dx)

        super(QoICallback, self).__init__(prob, functional, i, "qoi")


class GaugeCallback(TimeseriesCallback):
    """
    Callback for evaluating the free surface elevation field at a specific
    spatial location stored in the :class:`Options` parameter class
    associated with the :class:`AdaptiveProblem` object.
    """
    def __init__(self, prob, i, gauge):
        """
        :arg prob: :class:`AdaptiveProblem` object.
        :arg i: mesh index.
        :arg gauge: name of gauge to be evaluated.
        """
        self.label = "gauge {:s}".format(gauge)
        u, eta = prob.fwd_solutions[i].split()
        gauge_location = prob.op.gauges[gauge]["coords"]

        def extract(t):
            return float(eta.at(gauge_location))

        super(GaugeCallback, self).__init__(prob, extract, i, gauge)
