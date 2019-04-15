from firedrake import *
from firedrake_adjoint import *
from thetis.configuration import *

from adapt_utils.options import Options
from adapt_utils.misc import *


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

    def exact_solution(self, fs):
        pass

    def exact_objective(self):
        return assemble(inner(self.kernel, self.solution)*dx)

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
        #self.source.interpolate(self.bump(fs.mesh(), source=True))
        self.source.interpolate(self.box(fs.mesh(), source=True))
        area = assemble(self.source*dx)
        rescaling = 0.04/area if area != 0. else 1.
        self.source.interpolate(rescaling*self.source)
        self.source.rename("Source term")
        return self.source

    def set_objective_kernel(self, fs):
        self.kernel = Function(fs)
        #self.kernel.interpolate(self.bump(fs.mesh()))
        self.kernel.interpolate(self.box(fs.mesh()))
        area = assemble(self.kernel*dx)
        rescaling = 0.04/area if area != 0. else 1.
        self.kernel.interpolate(rescaling*self.kernel)
        return self.kernel


class TelemacOptions(TracerOptions):
    """
    Parameters for the 'Point source with diffusion' TELEMAC-2D test case.
    """
    def __init__(self, approach='fixed_mesh', offset=0.):
        super(TelemacOptions, self).__init__(approach)
        self.offset = offset

        # Source / receiver
        self.source_loc = [(1.+self.offset, 5., 0.1)]
        self.region_of_interest = [(20., 7.5, 0.5)]
        self.source_value = 100.
        self.source_discharge = 0.1

        # Boundary conditions
        self.boundary_conditions[1] = 'dirichlet_zero'
        self.boundary_conditions[3] = 'neumann_zero'
        self.boundary_conditions[4] = 'neumann_zero'

        self.sponge_scaling = 0.1    # scaling parameter
        self.sponge_start = 50.      # x-location of sponge start
        self.base_diffusivity = 0.1  # background diffusivity

    def set_diffusivity(self, fs):  # TODO: this is redundant
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
        self.source = Function(fs)
        with pyadjoint.stop_annotating():
            nrm=assemble(self.indicator(fs.mesh(), source=True)*dx)
        scaling = pi*r0*r0/nrm if nrm != 0 else 1
        scaling *= 0.5*self.source_value  # TODO: where does factor of half come from?
        self.source.interpolate(self.indicator(fs.mesh(), source=True, scale=scaling))
        return self.source

    def set_objective_kernel(self, fs):
        self.kernel = Function(fs)
        self.kernel.interpolate(self.indicator(fs.mesh()))
        return self.kernel

    def exact_solution(self, fs):
        self.solution = Function(fs)
        mesh = fs.mesh()
        x, y = SpatialCoordinate(mesh)
        x0, y0, r = self.source_loc[0]
        u = self.fluid_velocity
        nu = self.diffusivity
        r = max_value(sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0)), 0.1)  # (Bessel fn explodes at (x0, y0))
        q = 0.01  # sediment discharge of source (kg/s)
        xc = x - self.offset
        self.solution.interpolate(0.5*q/(pi*nu)*exp(0.5*u[0]*xc/nu)*bessk0(0.5*u[0]*r/nu))
        outfile = File(self.directory() + 'analytic.pvd')
        outfile.write(self.solution)  # NOTE: use 25 discretisation levels in ParaView
        return self.solution
