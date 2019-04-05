from firedrake import *
from firedrake_adjoint import *
from thetis.configuration import *

from adapt_utils.options import Options


__all__ = ["TracerOptions", "PowerOptions", "TelemacOptions"]


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
    end_time = PositiveFloat(50., help="End of time window of interest").tag(config=True)
    dt_per_export = PositiveFloat(10, help="Number of timesteps per export").tag(config=True)

    # Solver
    params = PETScSolverParameters({'pc_type': 'lu',
                                    'mat_type': 'aij' ,
                                    'ksp_monitor': None,
                                    'ksp_converged_reason': None}).tag(config=True)

    # Physical 
    source_loc = List(default_value=None, allow_none=True).tag(config=True)
    source = FiredrakeScalarExpression(None, allow_none=True).tag(config=True)
    diffusivity = FiredrakeScalarExpression(Constant(1e-1)).tag(config=True)
    fluid_velocity = FiredrakeVectorExpression(None, allow_none=True).tag(config=True)

    # Adjoint
    solve_adjoint = Bool(False).tag(config=True)
    region_of_interest = List(default_value=[(20., 7.5, 0.5)]).tag(config=True)

    # Adaptivity
    h_min = PositiveFloat(1e-4, help="Minimum tolerated element size").tag(config=True)
    h_max = PositiveFloat(5., help="Maximum tolerated element size").tag(config=True)

    boundary_conditions = PETScSolverParameters({}).tag(config=True)

    def __init__(self, approach='fixed_mesh'):
        super(TracerOptions, self).__init__(approach)

        self.end_time -= 0.5*self.dt
        self.restrict = 'anisotropy'

    def set_diffusivity(self, fs):
        pass

    def set_velocity(self, fs):
        pass

    def set_source(self, fs):
        pass

    def set_kernel(self, fs):
        pass


class PowerOptions(TracerOptions):
    """
    Parameters for test case in [Power et al 2006].
    """
    def __init__(self, approach='fixed_mesh'):
        super(PowerOptions, self).__init__(approach)

        # Source / receiver
        self.source_loc = [(1., 2., 0.1)]
        self.region_of_interest = [(3., 2., 0.1)]

        # Boundary conditions
        self.boundary_conditions[1] = 'dirichlet_zero'
        #self.boundary_conditions[2] = 'neumann_zero'  # FIXME
        self.boundary_conditions[3] = 'neumann_zero'
        self.boundary_conditions[4] = 'neumann_zero'

    def set_diffusivity(self, fs):
        self.diffusivity = Constant(1.)
        return self.diffusivity

    def set_velocity(self, fs):
        self.fluid_velocity = Function(fs)
        self.fluid_velocity.interpolate(as_vector((15., 0.)))
        return self.fluid_velocity

    def set_source(self, fs):
        self.source = Function(fs)
        self.source.interpolate(self.bump(fs.mesh(), source=True))
        self.source.rename("Source term")
        return self.source

    def set_objective_kernel(self, fs):
        self.kernel = Function(fs)
        self.kernel.interpolate(self.bump(fs.mesh()))
        return self.kernel


class TelemacOptions(TracerOptions):
    """
    Parameters for the 'Point source with diffusion' TELEMAC-2D test case.
    """
    def __init__(self, approach='fixed_mesh'):
        super(TelemacOptions, self).__init__(approach)

        # Source / receiver
        self.source_loc = [(1., 5., 0.457)]
        self.region_of_interest = [(20., 7.5, 0.5)]

        # Boundary conditions
        self.boundary_conditions[1] = 'dirichlet_zero'
        self.boundary_conditions[3] = 'neumann_zero'
        self.boundary_conditions[4] = 'neumann_zero'

        self.sponge_scaling = 0.1    # scaling parameter
        self.sponge_start = 50.      # x-location of sponge start
        self.base_diffusivity = 0.1  # background diffusivity

    def set_diffusivity(self, fs):
        x, y = SpatialCoordinate(fs.mesh())
        self.diffusivity = Function(fs)
        self.diffusivity.interpolate(self.base_diffusivity
                                     + self.sponge_scaling*pow(max_value(0, x-self.sponge_start), 2))
        return self.diffusivity

    def set_velocity(self, fs):
        self.fluid_velocity = Function(fs)
        self.fluid_velocity.interpolate(as_vector((1., 0.)))
        return self.fluid_velocity

    def set_source(self, fs):
        x, y = SpatialCoordinate(fs.mesh())
        x0, y0, r0 = self.source_loc[0]
        bell = 1 + cos(pi * min_value(sqrt(pow(x - x0, 2) + pow(y - y0, 2)) / r0, 1.0))
        self.source = Function(fs)
        self.source.interpolate(0. + conditional(ge(bell, 0.), bell, 0.))
        return self.source

    def set_objective_kernel(self, fs):
        self.kernel = Function(fs)
        self.kernel.interpolate(self.bump(fs.mesh()))
        return self.kernel
