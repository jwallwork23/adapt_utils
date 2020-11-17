from thetis import *

from adapt_utils.unsteady.callback import TimeseriesCallback


__all__ = ["DesalinationOutfallCallback"]


class DesalinationOutfallCallback(TimeseriesCallback):
    r"""
    Callback for evaluating functional quantities of interest of the form

  ..math::
        \int_0^T\int_R (c - c_b) \;\mathrm dx\;\mathrm dt,

    where :math:`c(\mathbf x,t)` is salinity, :math:`c_b` is the constant
    background salinity and :math:`R` is a region of interest.
    """
    def __init__(self, prob, i, **kwargs):
        """
        :arg prob: :class:`AdaptiveProblem` object.
        :arg i: mesh index.
        """
        self.label = "inlet salinity diff"
        ks = prob.op.set_qoi_kernel(prob.meshes[i])  # Kernel in space
        kt = Constant(0.0)                           # Kernel in time
        c = prob.fwd_solutions_tracer[i]
        c_b = prob.op.background_salinity

        def functional(t):
            kt.assign(1.0 if t >= prob.op.start_time else 0.0)
            return assemble(kt*ks*(c - c_b)*dx)

        label = "inlet_salinity_diff"
        super(DesalinationOutfallCallback, self).__init__(prob, functional, i, label, **kwargs)
