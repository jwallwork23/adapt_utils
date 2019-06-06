from thetis import *
from thetis.configuration import *

import numpy as np


__all__ = ["DefaultOptions"]


class Options(FrozenConfigurable):
    name = 'Common parameters for mesh adaptive simulations'

    # Adapt
    approach = Unicode('fixed_mesh', help="Mesh adaptive approach.").tag(config=True)
    dwr_approach = Unicode('error_representation', help="DWR error estimation approach, from {'error_representation', 'dwr', 'cell_facet_split'}. (See [Rognes & Logg, 2010])").tag(config=True)
    num_adapt = NonNegativeInteger(4, help="Number of mesh adaptations per remesh.").tag(config=True)
    rescaling = PositiveFloat(0.85, help="Scaling parameter for target number of vertices.").tag(config=True)
    convergence_rate = PositiveInteger(6).tag(config=True)  # TODO: docs

    # Smooth / intersect
    gradate = Bool(False, help='Toggle metric gradation.').tag(config=True)
    max_element_growth = PositiveFloat(1.4, help="Gradation scaling parameter.").tag(config=True)
    intersect = Bool(False, help='Intersect with previous mesh.').tag(config=True)
    relax = Bool(False, help='Take metric relaxation with previous mesh.').tag(config=True)
    intersect_boundary = Bool(False, help='Intersect with initial boundary metric.').tag(config=True)
    adapt_on_bathymetry = Bool(False, help='Toggle adaptation based on bathymetry.').tag(config=True)

    # Stabilisation
    stabilisation = Unicode(None, allow_none=True, help="Stabilisation approach.").tag(config=True)

    # Plotting
    plot_pvd = Bool(False, help='Toggle plotting of fields.').tag(config=True)
    plot_metric = Bool(False, help='Toggle plotting of metric field.').tag(config=True)

    # Metric
    max_anisotropy = PositiveFloat(100., help="Maximum tolerated anisotropy.").tag(config=True)
    restrict = Unicode('target', help="Hessian restriction approach, from {'num_vertices', 'p_norm', 'error'}.").tag(config=True)
    target = PositiveFloat(1e+2, help="Target number of vertices / inverse desired error for 'target' restriction approach.").tag(config=True)
    norm_order = NonNegativeInteger(2, help="Degree p of Lp norm used in 'p_norm' restriction approach.").tag(config=True)
    min_norm = PositiveFloat(1e-6).tag(config=True)
    max_norm = PositiveFloat(1e9).tag(config=True)

    # Hessian
    hessian_recovery = Unicode('dL2', help="Hessian recovery technique, from {'dL2', 'parts'}.").tag(config=True)
    hessian_solver_parameters = PETScSolverParameters({'snes_rtol': 1e8,
                                                       'ksp_rtol': 1e-5,
                                                       'ksp_gmres_restart': 20,
                                                       'pc_type': 'sor'}).tag(config=True)

    # PDE / optimisation
    timestepper = Unicode('CrankNicolson', help="Time integration scheme used.").tag(config=True)
    family = Unicode('dg-dg', help="Mixed finite element family, from {'dg-dg', 'dg-cg'}.").tag(config=True)
    degree = PositiveInteger(1, help="Order of function space").tag(config=True)
    element_rtol = PositiveFloat(0.01, help="Relative tolerance for convergence in mesh element count").tag(config=True)
    objective_rtol = PositiveFloat(0.00025, help="Relative tolerance for convergence in objective value.").tag(config=True)

    # Adjoint
    adjoint_steps = NonNegativeInteger(1000, help="Number of adjoint steps used").tag(config=True)
    solve_adjoint = Bool(False).tag(config=True)
    order_increase = Bool(False, help="Interpolate adjoint solution into higher order space.").tag(config=True)

    def __init__(self, approach='fixed_mesh'):  # TODO: always initialise approach
        self.approach = approach
        self.solve_adjoint = True if self.approach in ('DWP', 'DWR') else False  # TODO: redundant
        self.di = 'outputs/' + self.approach + '/'

    def final_index(self):
        """Final timestep index"""
        return int(np.ceil(self.end_time / self.dt))

    def first_export(self):
        """First exported timestep of period of interest"""
        return int(self.start_time / (self.dt_per_export * self.dt))

    def final_export(self):
        """Final exported timestep of period of interest"""
        return int(np.ceil(self.end_time / (self.dt_per_export * self.dt)))

    def final_mesh_index(self):
        """Final mesh index"""
        return int(self.final_index() / self.dt_per_remesh)

    def exports_per_remesh(self):
        """Number of exports per mesh adaptation"""
        assert self.dt_per_remesh % self.dt_per_export == 0
        return int(self.dt_per_remesh / self.dt_per_export)

    def disk(self, fs, scale=1., source=False):
        """Disk indicator function associated with region(s) of interest"""
        x, y = SpatialCoordinate(fs.mesh())
        locs = self.source_loc if source else self.region_of_interest
        eps = 1e-10
        for j in range(len(locs)):
            x0 = locs[j][0]
            y0 = locs[j][1]
            r = locs[j][2]
            if j == 0:
                b = lt((x-x0)*(x-x0) + (y-y0)*(y-y0), r*r + eps)
            else:
                b = Or(b, lt((x-x0)*(x-x0) + (y-y0)*(y-y0), r*r + eps))
        expr = conditional(b, scale, 0.)
        indi = Function(fs)
        indi.interpolate(expr)  # NOTE: Pyadjoint can't deal with coordinateless functions
        return indi

    def bump(self, fs, scale=1., source=False):
        """Bump function associated with region(s) of interest"""
        x, y = SpatialCoordinate(fs.mesh())
        locs = self.source_loc if source else self.region_of_interest
        i = 0
        for j in range(len(locs)):
            x0 = locs[j][0]
            y0 = locs[j][1]
            r = locs[j][2]
            i += conditional(lt(((x-x0)*(x-x0) + (y-y0)*(y-y0)), r*r),
                             scale*exp(1.-1./(1.-(x-x0)*(x-x0)/r**2))*exp(1.-1./(1.-(y-y0)*(y-y0)/r**2)),
                             0.)
        bump = Function(fs)
        bump.interpolate(i)  # NOTE: Pyadjoint can't deal with coordinateless functions
        return bump

    def gaussian(self, fs, scale=1., source=False):
        """Gaussian function associated with region(s) of interest"""
        x, y = SpatialCoordinate(fs)
        locs = self.source_loc if source else self.region_of_interest
        i = 0
        for j in range(len(locs)):
            x0 = locs[j][0]
            y0 = locs[j][1]
            r = locs[j][2]
            i += conditional(lt(((x-x0)*(x-x0) + (y-y0)*(y-y0)), r*r),
                             scale*exp(1. - 1. / (1. - ((x-x0)*(x-x0) + (y-y0)*(y-y0)) / r ** 2)),
                             0.)
        bump = Function(fs)
        bump.interpolate(i)  # NOTE: Pyadjoint can't deal with coordinateless functions
        return bump

    def box(self, fs, scale=1., source=False):
        """Box function associated with region(s) of interest"""
        x, y = SpatialCoordinate(fs)
        locs = self.source_loc if source else self.region_of_interest
        for j in range(len(locs)):
            x0 = locs[j][0]
            y0 = locs[j][1]
            r = locs[j][2]
            if j == 0:
                b = And(And(gt(x, x0-r), lt(x, x0+r)), And(gt(y, y0-r), lt(y, y0+r)))
            else:
                b = Or(b, And(And(gt(x, x0-r), lt(x, x0+r)), And(gt(y, y0-r), lt(y, y0+r))))
        expr = conditional(b, scale, 0.)
        box = Function(fs)
        box.interpolate(expr)  # NOTE: Pyadjoint can't deal with coordinateless functions
        return box


class DefaultOptions(Options):
    name = 'Parameters for the case where no mode is selected'
    mode = 'Default'

    # solver
    dt = PositiveFloat(0.01, name="Timestep").tag(config=True)
    start_time = NonNegativeFloat(0., help="Start of time window of interest").tag(config=True)
    end_time = PositiveFloat(10., help="End of time window of interest (and simulation)").tag(config=True)
    dt_per_export = PositiveInteger(10, help="Number of timesteps per export").tag(config=True)

    # adapt
    h_min = PositiveFloat(1e-6, help="Minimum tolerated element size").tag(config=True)
    h_max = PositiveFloat(1e3, help="Maximum tolerated element size").tag(config=True)

    # physical
    viscosity = NonNegativeFloat(1e-3).tag(config=True)
    drag_coefficient = NonNegativeFloat(0.0025).tag(config=True)

    # adjoint
    region_of_interest = List(default_value=[(0.5, 0.5, 0.1)]).tag(config=True)
    # TODO: Surely below is redundant?
    loc_x = Float(0., help="x-coordinate of centre of important region").tag(config=True)
    loc_y = Float(0., help="y-coordinate of centre of important region").tag(config=True)
    loc_r = PositiveFloat(1., help="Radius of important region").tag(config=True)

    def __init__(self, approach='FixedMesh'):
        super(DefaultOptions, self).__init__(approach)

        self.end_time -= 0.5*self.dt
