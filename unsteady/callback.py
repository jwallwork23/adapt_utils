from thetis import *


__all__ = ["VelocityNormCallback", "ElevationNormCallback", "TracerNormCallback",
           "SedimentNormCallback", "ExnerNormCallback", "QoICallback", "GaugeCallback"]


class TimeseriesCallback(object):
    """
    Generic callback object for storing timseries extracted during a
    simulation and integrating in time.
    """
<<<<<<< HEAD
    def __init__(self, prob, func, i, name):
=======
    def __init__(self, prob, func, i, name, callback_dir=None, **kwargs):
>>>>>>> origin/master
        """
        :arg prob: :class:`AdaptiveProblem` object.
        :arg func: user-provided function to be evaluated.
        :arg i: mesh index.
        :arg name: name for the callback object.
        """
        self.prob = prob
        self.name = name
        self.func = func
        self.timeseries = []
        self.msg = "    {:16s}".format(self.name) + " at time {:6.1f} = {:11.4e}"

    def evaluate(self, **kwargs):
        """
        Evaluate callback at current time and append it to timeseries.
        """
        t = self.prob.simulation_time
        value = self.func(t)
        print_output(self.msg.format(t, value))
        self.timeseries.append(value)

    def time_integrate(self):
        """
        Integrate the timeseries using a quadrature routine appropriate for the timestepper.
        """
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
<<<<<<< HEAD
    """Callback for evaluating the L2 norm of the velocity field at each timestep/export."""
    def __init__(self, prob, i):
        self.name = "velocity norm"
        u, eta = prob.fwd_solutions[i].split()
        super(VelocityNormCallback, self).__init__(prob, lambda t: norm(u), i, "velocity_norm")


class ElevationNormCallback(TimeseriesCallback):
    """Callback for evaluating the L2 norm of the elevation field at each timestep/export."""
    def __init__(self, prob, i):
        self.name = "elevation norm"
        u, eta = prob.fwd_solutions[i].split()
        super(ElevationNormCallback, self).__init__(prob, lambda t: norm(eta), i, "elevation_norm")


class TracerNormCallback(TimeseriesCallback):
    """Callback for evaluating the L2 norm of the tracer concentration at each timestep/export."""
    def __init__(self, prob, i):
        self.name = "tracer norm"
        c = prob.fwd_solutions_tracer[i]
        super(TracerNormCallback, self).__init__(prob, lambda t: norm(c), i, "tracer_norm")


class SedimentNormCallback(TimeseriesCallback):
    """Callback for evaluating the L2 norm of the sediment at each timestep/export."""
    def __init__(self, prob, i):
        self.name = "sediment norm"
        s = prob.fwd_solutions_sediment[i]
        super(SedimentNormCallback, self).__init__(prob, lambda t: norm(s), i, "sediment_norm")


class ExnerNormCallback(TimeseriesCallback):
    """Callback for evaluating the L2 norm of the modified bathymetry at each timestep/export."""
    def __init__(self, prob, i):
        self.name = "bathymetry norm"
        b = prob.fwd_solutions_bathymetry[i]
        super(ExnerNormCallback, self).__init__(prob, lambda t: norm(b), i, "bathymetry_norm")


=======
    """
    Callback for evaluating the L2 norm of the velocity field at each timestep/export.
    """
    def __init__(self, prob, i, adjoint=False):
        self.label = "adjoint " if adjoint else ""
        self.label += "velocity norm"
        name = self.label.replace(" ", "_")
        u, eta = prob.get_solutions("shallow_water", adjoint=adjoint)[i].split()
        super(VelocityNormCallback, self).__init__(prob, lambda t: norm(u), i, name)


class ElevationNormCallback(TimeseriesCallback):
    """
    Callback for evaluating the L2 norm of the elevation field at each timestep/export.
    """
    def __init__(self, prob, i, adjoint=False):
        self.label = "adjoint " if adjoint else ""
        self.label += "elevation norm"
        name = self.label.replace(" ", "_")
        u, eta = prob.get_solutions("shallow_water", adjoint=adjoint)[i].split()
        super(ElevationNormCallback, self).__init__(prob, lambda t: norm(eta), i, name)


class TracerNormCallback(TimeseriesCallback):
    """
    Callback for evaluating the L2 norm of the tracer concentration at each timestep/export.
    """
    def __init__(self, prob, i, adjoint=False):
        self.label = "adjoint " if adjoint else ""
        self.label += "tracer norm"
        name = self.label.replace(" ", "_")
        c = prob.get_solutions("tracer", adjoint=adjoint)[i]
        super(TracerNormCallback, self).__init__(prob, lambda t: norm(c), i, name)


class SedimentNormCallback(TimeseriesCallback):
    """
    Callback for evaluating the L2 norm of the sediment at each timestep/export.
    """
    def __init__(self, prob, i, adjoint=False):
        self.label = "adjoint " if adjoint else ""
        self.label += "sediment norm"
        name = self.label.replace(" ", "_")
        s = prob.get_solutions("sediment", adjoint=adjoint)[i]
        super(SedimentNormCallback, self).__init__(prob, lambda t: norm(s), i, name)


class ExnerNormCallback(TimeseriesCallback):
    """
    Callback for evaluating the L2 norm of the modified bathymetry at each timestep/export.
    """
    def __init__(self, prob, i, adjoint=False):
        self.label = "adjoint " if adjoint else ""
        self.label += "bathymetry norm"
        name = self.label.replace(" ", "_")
        b = prob.get_solutions("bathymetry", adjoint=adjoint)[i]
        super(ExnerNormCallback, self).__init__(prob, lambda t: norm(b), i, name)


class VorticityNormCallback(TimeseriesCallback):
    """
    Callback for evaluating the L2 norm of the fluid vorticity at each timestep/export.
    """
    def __init__(self, prob, i, **kwargs):
        self.label = "vorticity norm"
        rec = L2ProjectorVorticity(prob.V[i], op=prob.op)
        prob.vorticity[i] = Function(prob.P1[i], name="Vorticity")

        def vorticity(t):
            prob.vorticity[i].assign(rec.project(prob.fwd_solutions[i]))
            if prob.op.plot_pvd:
                prob.vorticity_file._topology = None
                prob.vorticity_file.write(prob.vorticity[i])
            return norm(prob.vorticity[i])

        super(VorticityNormCallback, self).__init__(prob, vorticity, i, "vorticity_norm")


>>>>>>> origin/master
class QoICallback(TimeseriesCallback):
    r"""
    Callback for evaluating functional quantities of interest of the form

  ..math::
        \int_0^T\int_\Omega\mathbf k\cdot\mathbf q \;\mathrm dx\;\mathrm dt,

    where :math:`\mathbf k=\mathbf k(\mathbf x,t)` is a kernel function and
    :math:`\mathbf q=\mathbf q(\mathbf x,t)` is the prognostic solution
    tuple for the forward problem.
    """
    def __init__(self, prob, i, **kwargs):
        """
        :arg prob: :class:`AdaptiveProblem` object.
        :arg i: mesh index.
        """
        self.name = "QoI"
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
    def __init__(self, prob, i, gauge, **kwargs):
        """
        :arg prob: :class:`AdaptiveProblem` object.
        :arg i: mesh index.
        :arg gauge: name of gauge to be evaluated.
        """
        self.name = gauge
        u, eta = prob.fwd_solutions[i].split()
        gauge_location = prob.op.gauges[gauge]["coords"]

        def extract(t):
            return float(eta.at(gauge_location))

        super(GaugeCallback, self).__init__(prob, extract, i, gauge)
