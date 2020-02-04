from thetis import *

from adapt_utils.swe.tsunami.options import heaviside_approx


__all__ = ["InundationCallback"]


class InundationCallback(callback.AccumulatorCallback):
    name = "Inundation QoI"

    def __init__(self, solver_obj, **kwargs):
        alpha = solver_obj.options.wetting_and_drying_alpha
        eta = solver_obj.fields.elev_2d
        b = solver_obj.fields.bathymetry_2d
        dry = conditional(le(b, 0), 0, 1)
        f_init = assemble(heaviside_approx(eta + b, alpha)*dx(degree=12))
        def qoi():
            return assemble(dry*(eta + heaviside_approx(eta + b, alpha))*dx(degree=12)) - f_init
        super(InundationCallback, self).__init__(qoi, solver_obj, **kwargs)

# TODO: Flux over coast. (Needs internal boundary.)
# TODO: Consider other QoIs. (Speak to Branwen.)
