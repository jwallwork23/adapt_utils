from firedrake import *


__all__ = ["QoICallback", "GaugeCallback"]


class TimeseriesCallback(object):
    def __init__(self, prob, func, i, name):
        self.prob = prob
        self.name = name
        self.func = func
        self.timeseries = []

    def evaluate(self, **kwargs):
        self.timeseries.append(self.func(self.prob.simulation_time))

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


class QoICallback(TimeseriesCallback):
    def __init__(self, prob, i):
        ks = prob.kernels[i]  # Kernel in space
        kt = Constant(0.0)    # Kernel in time
        sol = prob.fwd_solutions[i]

        def functional(t):
            kt.assign(1.0 if t >= prob.op.start_time else 0.0)
            return assemble(kt*inner(ks, sol)*dx)

        super(QoICallback, self).__init__(prob, functional, i, "qoi")


class GaugeCallback(TimeseriesCallback):
    def __init__(self, prob, i, gauge):
        u, eta = prob.fwd_solutions[i].split()
        gauge_location = prob.op.gauges[gauge]["coords"]

        def extract(t):
            return eta.at(gauge_location)

        super(GaugeCallback, self).__init__(prob, extract, i, gauge)
