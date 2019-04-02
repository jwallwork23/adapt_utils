from firedrake import File
from thetis.configuration import *

from adapt_utils.options import Options


__all__ = ["TracerOptions"]


class TracerOptions(Options):
    """
    Default parameter class for TracerProblems.
    """

    # Domain
    nx = PositiveInteger(4, help="Mesh resolution in x- and y-directions.").tag(config=True)
    target_vertices = PositiveFloat(1000., help="Target number of vertices (not an integer!)")

    # Timestepping
    dt = PositiveFloat(0.1, help="Timestep").tag(config=True)
    start_time = NonNegativeFloat(0., help="Start of time window of interest").tag(config=True)
    end_time = PositiveFloat(50., help="End of time window of interest (and simulation)").tag(config=True)
    dt_per_export = PositiveFloat(10, help="Number of timesteps per export").tag(config=True)

    # Source
    #bell_x0 = Float(2.5, help="x-coordinate of source centre").tag(config=True)
    #bell_y0 = Float(5., help="y-coordinate of source centre").tag(config=True)
    #bell_r0 = PositiveFloat(0.457, help="Radius of source Gaussian").tag(config=True)
    source = List(default_value=[(2.5, 5., 0.457)]).tag(config=True)

    # Diffusivity and sponge BC
    sponge_scaling = PositiveFloat(0.1, help="Scaling for quadratic sponge.").tag(config=True)
    sponge_start = PositiveFloat(50., help="x-coordinate where sponge starts").tag(config=True)
    diffusivity = PositiveFloat(1e-1).tag(config=True)

    # Indicator function and adjoint
    solve_adjoint = Bool(False).tag(config=True)
    region_of_interest = List(default_value=[(20., 7.5, 0.5)]).tag(config=True)

    # Adaptivity options
    h_min = PositiveFloat(1e-4, help="Minimum tolerated element size").tag(config=True)
    h_max = PositiveFloat(5., help="Maximum tolerated element size").tag(config=True)

    def __init__(self, approach='fixed_mesh'):
        super(TracerOptions, self).__init__(approach)

        self.end_time -= 0.5*self.dt


class PowerOptions(TracerOptions):
    """
    Parameters for test case in [Power et al 2006].
    """
    def __init__(self, approach='fixed_mesh'):
        super(PowerOptions, self).__init__(approach)
        self.diffusivity = 1.
