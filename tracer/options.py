from firedrake import File
from thetis.configuration import *

from adapt.options import AdaptOptions


__all__ = ["TracerOptions"]


class TracerOptions(AdaptOptions):

    # Domain
    nx = PositiveInteger(4).tag(config=True)  # TODO: help

    # Timestepping
    dt = PositiveFloat(0.1, help="Timestep").tag(config=True)
    start_time = NonNegativeFloat(0., help="Start of time window of interest").tag(config=True)
    end_time = PositiveFloat(50., help="End of time window of interest (and simulation)").tag(config=True)
    dt_per_export = PositiveFloat(10, help="Number of timesteps per export").tag(config=True)

    # Source
    bell_x0 = Float(2.5, help="x-coordinate of source centre").tag(config=True)
    bell_y0 = Float(5., help="y-coordinate of source centre").tag(config=True)
    bell_r0 = PositiveFloat(0.457, help="Radius of source Gaussian").tag(config=True)

    # Diffusivity and sponge BC
    sponge_scaling = PositiveFloat(0.1, help="Scaling parameter for quadratic sponge to right-hand boundary").tag(config=True)
    sponge_start = PositiveFloat(50., help="x-coordinate where sponge starts").tag(config=True)
    viscosity = PositiveFloat(1e-1).tag(config=True)

    # Indicator function and adjoint
    solve_adjoint = Bool(False).tag(config=True)
    loc_x = Float(40., help="x-coordinate of centre of important region").tag(config=True)
    loc_y = Float(7.5, help="y-coordinate of centre of important region").tag(config=True)
    loc_r = PositiveFloat(0.5, help="Radius of important region").tag(config=True)

    def __init__(self, approach='FixedMesh'):
        super(TracerOptions, self).__init__(approach)

        self.end_time -= 0.5*self.dt

        # Plotting
        self.adjoint_outfile = File(self.directory()+'Tracer2d/Adjoint2d.pvd')
        if self.solve_adjoint:
            self.estimator_outfile = File(self.directory()+'Tracer2d/'+self.approach+'Estimator2d.pvd')
