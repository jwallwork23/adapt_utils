from thetis import *
from thetis.configuration import *

import numpy as np


__all__ = ["DefaultOptions"]


class AdaptOptions(FrozenConfigurable):
    name = 'Common parameters for mesh adaptive simulations'

    # Mesh adaptivity parameters
    approach = Unicode('FixedMesh',
                       help="Mesh adaptive approach, from {'FixedMesh', 'HessianBased', 'DWP', 'DWR'}"
                       ).tag(config=True)
    gradate = Bool(False, help='Toggle metric gradation.').tag(config=True)
    intersect = Bool(False, help='Intersect with previous mesh.').tag(config=True)
    intersect_boundary = Bool(False, help='Intersect with initial boundary metric.').tag(config=True)
    adapt_on_bathymetry = Bool(False, help='Toggle adaptation based on bathymetry.').tag(config=True)
    plot_pvd = Bool(False, help='Toggle plotting of fields.').tag(config=True)
    plot_metric = Bool(False, help='Toggle plotting of metric field.').tag(config=True)
    max_element_growth = PositiveFloat(1.4,
                                       help="Metric gradation scaling parameter.").tag(config=True)
    max_anisotropy = PositiveFloat(100., help="Maximum tolerated anisotropy.").tag(config=True)
    num_adapt = NonNegativeInteger(1, help="Number of mesh adaptations per remesh.").tag(config=True)
    order_increase = Bool(False,
                          help="Interpolate adjoint solution into higher order space."
                          ).tag(config=True)
    normalisation = Unicode('lp',
                            help="Normalisation approach, from {'lp', 'manual'}.").tag(config=True)
    hessian_recovery = Unicode('dL2',
                               help="Hessian recovery technique, from {'dL2', 'parts'}."
                               ).tag(config=True)
    timestepper = Unicode('CrankNicolson', help="Time integration scheme used.").tag(config=True)
    norm_order = NonNegativeInteger(2, help="Degree p of Lp norm used.")
    family = Unicode('dg-dg',
                     help="Mixed finite element family, from {'dg-dg', 'dg-cg'}.").tag(config=True)
    min_norm = PositiveFloat(1e-6).tag(config=True)
    max_norm = PositiveFloat(1e9).tag(config=True)

    # Initialisation for number of adjoint steps (always be changed by a call to `store_adjoint`)
    adjoint_steps = NonNegativeInteger(1000, help="Number of adjoint steps used").tag(config=True)
    solve_adjoint = Bool(False).tag(config=True)

    def __init__(self, approach='FixedMesh'):
        try:
            assert(approach in ('FixedMesh', 'HessianBased', 'DWP', 'DWR'))
        except:
            raise ValueError
        self.approach = approach
        self.solve_adjoint = True if self.approach in ('DWP', 'DWR') else False

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

    def mixed_space(self, mesh, enrich=False):
        """
        :param mesh: mesh upon which to build mixed space.
        :return: mixed VectorFunctionSpace x FunctionSpace as specified by ``self.family``.
        """
        d1 = 1
        d2 = 2 if self.family == 'dg-cg' else 1
        if enrich:
            d1 += self.order_increase
            d2 += self.order_increase
        dgdg = self.family == 'dg-dg'
        return VectorFunctionSpace(mesh, "DG", d1) * FunctionSpace(mesh, "DG" if dgdg else "CG", d2)

    def adaptation_stats(self, mn, adaptTimer, solverTime, nEle, Sn, mM, t):
        """
        :arg mn: mesh number.
        :arg adaptTimer: time taken for mesh adaption.
        :arg solverTime: time taken for solver.
        :arg nEle: current number of elements.
        :arg Sn: sum over #Elements.
        :arg mM: tuple of min and max #Elements.
        :arg t: current simuation time.
        :return: mean element count.
        """
        av = Sn / mn
        print("""\n************************** Adaption step %d ****************************
Percent complete  : %4.1f%%    Adapt time : %4.2fs Solver time : %4.2fs     
#Elements... Current : %d  Mean : %d  Minimum : %s  Maximum : %s\n""" %
              (mn, 100 * t / self.end_time, adaptTimer, solverTime, nEle, av, mM[0], mM[1]))
        return av

    def directory(self):
        return 'outputs/' + self.approach + '/'

    def indicator(self, mesh):
        """Indicator function associated with region of interest"""
        P1DG = FunctionSpace(mesh, "DG", 1)
        iA = Function(P1DG, name="Region of interest")
        x = SpatialCoordinate(mesh)
        eps = 1e-10
        expr = conditional(lt(pow(x[0]-self.loc_x, 2) + pow(x[1]-self.loc_y, 2), pow(self.loc_r, 2) + eps), 1, 0)
        return project(expr, P1DG)


class DefaultOptions(AdaptOptions):
    name = 'Parameters for the case where no mode is selected'
    mode = 'Default'

    # Solver parameters
    dt = PositiveFloat(0.01, name="Timestep").tag(config=True)
    start_time = NonNegativeFloat(0., help="Start of time window of interest").tag(config=True)
    end_time = PositiveFloat(10., help="End of time window of interest (and simulation)").tag(config=True)
    dt_per_export = PositiveInteger(10, help="Number of timesteps per export").tag(config=True)

    # Adaptivity parameters
    h_min = PositiveFloat(1e-6, help="Minimum tolerated element size").tag(config=True)
    h_max = PositiveFloat(1., help="Maximum tolerated element size").tag(config=True)
    target_vertices = PositiveFloat(1000., help="Target number of vertices (not an integer!)")
    rescaling = PositiveFloat(0.85, help="Scaling parameter for target number of vertices.").tag(config=True)

    # Physical parameters
    viscosity = NonNegativeFloat(1e-3).tag(config=True)
    drag_coefficient = NonNegativeFloat(0.0025).tag(config=True)

    # Indicator function and adjoint
    loc_x = Float(0., help="x-coordinate of centre of important region").tag(config=True)
    loc_y = Float(0., help="y-coordinate of centre of important region").tag(config=True)
    loc_r = PositiveFloat(1., help="Radius of important region").tag(config=True)

