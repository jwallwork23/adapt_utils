from thetis import *
from thetis.configuration import *

import numpy as np


__all__ = ["DefaultOptions"]


class Options(FrozenConfigurable):
    name = 'Common parameters for mesh adaptive simulations'

    # Adapt
    approach = Unicode('fixed_mesh', help="Mesh adaptive approach.").tag(config=True)
    num_adapt = NonNegativeInteger(4, help="Number of mesh adaptations per remesh.").tag(config=True)
    rescaling = PositiveFloat(0.85, help="Scaling parameter for target number of vertices.").tag(config=True)
    convergence_rate = PositiveInteger(6, help="Convergence rate parameter used in approach of [Carpio et al. 2013].").tag(config=True)
    h_min = PositiveFloat(1e-10, help="Minimum tolerated element size.").tag(config=True)
    h_max = PositiveFloat(5., help="Maximum tolerated element size.").tag(config=True)

    # Stabilisation
    stabilisation = Unicode(None, allow_none=True, help="Stabilisation approach.").tag(config=True)

    # Plotting
    plot_pvd = Bool(False, help="Toggle plotting of fields.").tag(config=True)

    # Metric
    max_anisotropy = PositiveFloat(1000., help="Maximum tolerated anisotropy.").tag(config=True)
    normalisation = Unicode('complexity', help="Metric normalisation approach, from {'complexity', 'error'}.").tag(config=True)
    target = PositiveFloat(1e+2, help="Target complexity / inverse desired error for normalisation, as appropriate.").tag(config=True)
    norm_order = NonNegativeInteger(None, allow_none=True, help="Degree p of Lp norm used in 'error' normalisation approach. Use 'None' to specify infinity norm.").tag(config=True)
    intersect_boundary = Bool(False, help="Intersect with initial boundary metric.").tag(config=True)

    # Hessian
    hessian_recovery = Unicode('dL2', help="Hessian recovery technique, from {'dL2', 'parts'}.").tag(config=True)
    hessian_solver_parameters = PETScSolverParameters({'snes_rtol': 1e8,
                                                       'ksp_rtol': 1e-5,
                                                       'ksp_gmres_restart': 20,
                                                       'pc_type': 'sor'}).tag(config=True)

    # Time discretisation
    timestepper = Unicode('CrankNicolson', help="Time integration scheme used.").tag(config=True)
    dt = PositiveFloat(0.1, help="Timestep").tag(config=True)
    start_time = NonNegativeFloat(0., help="Start of time window of interest.").tag(config=True)
    end_time = PositiveFloat(60., help="End of time window of interest.").tag(config=True)
    dt_per_export = PositiveFloat(10, help="Number of timesteps per export.").tag(config=True)
    dt_per_remesh = PositiveFloat(20, help="Number of timesteps per mesh adaptation.").tag(config=True)

    # Finite element space  # TODO: use a notation which is more general
    family = Unicode('dg-dg', help="Mixed finite element family, from {'dg-dg', 'dg-cg'}.").tag(config=True)

    # PDE / optimisation
    boundary_conditions = PETScSolverParameters({}, help="Boundary conditions expressed as a dictionary.").tag(config=True)
    adjoint_boundary_conditions = PETScSolverParameters({}, help="Boundary conditions for adjoint problem expressed as a dictionary.").tag(config=True)
    degree = NonNegativeInteger(1, help="Order of function space").tag(config=True)
    region_of_interest = List(default_value=[], help="Spatial region related to quantity of interest").tag(config=True)
    element_rtol = PositiveFloat(0.005, help="Relative tolerance for convergence in mesh element count").tag(config=True)
    qoi_rtol = PositiveFloat(0.005, help="Relative tolerance for convergence in quantity of interest.").tag(config=True)
    estimator_rtol = PositiveFloat(0.005, help="Relative tolerance for convergence in error estimator.").tag(config=True)

    # Adjoint
    order_increase = Bool(False, help="Interpolate adjoint solution into higher order space.").tag(config=True)

    def __init__(self, approach='fixed_mesh'):
        self.approach = approach
        self.di = 'outputs/' + self.approach + '/'

    def ball(self, fs, scale=1., source=False):
        """Ball indicator function associated with region(s) of interest"""
        mesh = fs.mesh()
        dim = mesh.topological_dimension()
        assert dim in (2, 3)
        if dim == 2:
            x, y = SpatialCoordinate(fs)
        else:
            x, y, z = SpatialCoordinate(fs)
        locs = self.source_loc if source else self.region_of_interest
        eps = 1e-10
        for j in range(len(locs)):
            x0 = locs[j][0]
            y0 = locs[j][1]
            r = locs[j][2] if dim == 2 else locs[j][3]
            if dim == 3:
                z0 = locs[j][2]
                expr = lt((x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0), r*r + eps)
            else:
                expr = lt((x-x0)*(x-x0) + (y-y0)*(y-y0), r*r + eps)
            b = expr if j == 0 else Or(b, expr)
        expr = conditional(b, scale, 0.)
        indi = Function(fs)
        indi.interpolate(expr)
        return indi

    def bump(self, fs, scale=1., source=False):
        """Bump function associated with region(s) of interest"""
        mesh = fs.mesh()
        dim = mesh.topological_dimension()
        assert dim in (2, 3)
        if dim == 2:
            x, y = SpatialCoordinate(fs)
        else:
            x, y, z = SpatialCoordinate(fs)
        locs = self.source_loc if source else self.region_of_interest
        i = 0
        for j in range(len(locs)):
            x0 = locs[j][0]
            y0 = locs[j][1]
            r = locs[j][2] if dim == 2 else locs[j][3]
            expr1 = (x-x0)*(x-x0) + (y-y0)*(y-y0)
            expr2 = scale*exp(1 -1/(1 - (x-x0)*(x-x0)/r**2))*exp(1 - 1/(1 - (y-y0)*(y-y0)/r**2))
            if dim == 3:
                z0 = locs[j][2]
                expr1 += (z-z0)*(z-z0)
                expr2 *= exp(1 - 1/(1 - (z-z0)*(z-z0)/r**2))
            i += conditional(lt(expr1, r*r), expr2, 0)
        bump = Function(fs)
        bump.interpolate(i)
        return bump

    def gaussian(self, fs, scale=1., source=False):
        """Gaussian function associated with region(s) of interest"""
        mesh = fs.mesh()
        dim = mesh.topological_dimension()
        assert dim in (2, 3)
        if dim == 2:
            x, y = SpatialCoordinate(fs)
        else:
            x, y, z = SpatialCoordinate(fs)
        locs = self.source_loc if source else self.region_of_interest
        i = 0
        for j in range(len(locs)):
            x0 = locs[j][0]
            y0 = locs[j][1]
            r = locs[j][2] if dim == 2 else locs[j][3]
            expr = (x-x0)*(x-x0) + (y-y0)*(y-y0)
            if dim == 3:
                z0 = locs[j][2]
                expr += (z-z0)*(z-z0)
            i += conditional(lt(expr, r*r), scale*exp(1 - 1/(1 - expr/r**2)), 0)
        bump = Function(fs)
        bump.interpolate(i)
        return bump

    def box(self, fs, scale=1., source=False):
        """Box function associated with region(s) of interest"""
        mesh = fs.mesh()
        dim = mesh.topological_dimension()
        assert dim in (2, 3)
        if dim == 2:
            x, y = SpatialCoordinate(fs)
        else:
            x, y, z = SpatialCoordinate(fs)
        locs = self.source_loc if source else self.region_of_interest
        for j in range(len(locs)):
            x0 = locs[j][0]
            y0 = locs[j][1]
            r = locs[j][2] if dim == 2 else locs[j][3]
            expr = And(And(gt(x, x0-r), lt(x, x0+r)), And(gt(y, y0-r), lt(y, y0+r)))
            if dim == 3:
                z0 = locs[j][2]
                expr = And(expr, And(gt(z, z0-r), lt(z, z0+r)))
            b = expr if j == 0 else Or(b, expr)
        expr = conditional(b, scale, 0.)
        box = Function(fs)
        box.interpolate(expr)
        return box


class DefaultOptions(Options):
    name = 'Parameters for the case where no mode is selected'
    mode = 'Default'

    def __init__(self, approach='fixed_mesh', dt=0.01):
        super(DefaultOptions, self).__init__(approach)
        self.dt = dt
        self.start_time = 0.
        self.end_time = 10.
        self.dt_per_export = 10
        self.end_time -= 0.5*self.dt
        self.h_min = 1e-6
        self.h_max = 1e3
