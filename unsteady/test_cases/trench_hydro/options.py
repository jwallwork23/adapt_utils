from thetis import *
from thetis.configuration import *

from adapt_utils.unsteady.options import CoupledOptions
from adapt_utils.unsteady.swe.utils import heaviside_approx

import os
import numpy as np
import matplotlib
# import matplotlib.pyplot as plt


matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif')


__all__ = ["TrenchHydroOptions"]


class TrenchHydroOptions(CoupledOptions):

    def __init__(self, friction='nikuradse', plot_timeseries=False, nx=1, ny=1, **kwargs):
        super(TrenchHydroOptions, self).__init__(**kwargs)
        self.plot_timeseries = plot_timeseries
        self.default_mesh = RectangleMesh(np.int(16*5*nx), 5*ny, 16, 1.1)
        self.plot_pvd = True
        self.num_hours = 15

        self.di = os.path.join(self.di, 'bathymetry')

        # Physical
        self.base_viscosity = 1e-6
        self.base_diffusivity = 0.15
        self.wetting_and_drying = False
        try:
            assert friction in ('nikuradse', 'manning', 'nik_solver')
        except AssertionError:
            raise ValueError("Friction parametrisation '{:s}' not recognised.".format(friction))
        self.friction = friction
        self.average_size = 160e-6  # Average sediment size
        self.friction_coeff = 0.025
        self.ksp = None
        # Stabilisation
        self.stabilisation = 'lax_friedrichs'

        # Time integration
        self.dt = 0.25
        self.end_time = 500 #self.num_hours*3600.0
        # self.dt_per_export = 6
        self.dt_per_export = 100
        self.timestepper = 'CrankNicolson'
        self.implicitness_theta = 1.0
        self.family = 'dg-dg'

        # Adaptivity
        self.h_min = 1e-8
        self.h_max = 10.

        # Timeseries
        self.wd_obs = []
        self.trange = np.linspace(0.0, self.end_time, self.num_hours+1)
        tol = 1e-8  # FIXME: Point evaluation hack
        self.xrange = np.linspace(tol, 16-tol, 20)

        self.uv_file = File(os.path.join(self.di, 'uv.pvd'))


    def set_quadratic_drag_coefficient(self, fs):
        self.depth = Function(fs).interpolate(self.set_bathymetry(fs) + Constant(0.397))
        if self.friction == 'nikuradse':
            return interpolate(self.get_cfactor(self.depth), fs)

    def get_cfactor(self, depth):
        try:
            assert hasattr(self, 'depth')
        except AssertionError:
            raise ValueError("Depth is undefined.")
        ksp = Constant(3*self.average_size)
        hc = conditional(depth > Constant(0.001), depth, Constant(0.001))
        aux = max_value(11.036*hc/ksp, 1.001)
        return 2*(0.4**2)/(ln(aux)**2)

    def set_bathymetry(self, fs):

        initial_depth = Constant(0.397)
        depth_riv = Constant(initial_depth - 0.397)
        depth_trench = Constant(depth_riv - 0.15)
        depth_diff = depth_trench - depth_riv
        x, y = SpatialCoordinate(fs.mesh())
        trench = conditional(le(x, 5), depth_riv, conditional(le(x, 6.5), (1/1.5)*depth_diff*(x-6.5) + depth_trench,
                             conditional(le(x, 9.5), depth_trench, conditional(le(x, 11), -(1/1.5)*depth_diff*(x-11) + depth_riv, depth_riv))))
        return interpolate(-trench, fs)

    def set_boundary_conditions(self, prob, i):
        inflow_tag = 1
        outflow_tag = 2
        bottom_wall_tag = 3
        top_wall_tag = 4
        boundary_conditions = {
            'shallow_water': {
                inflow_tag: {'flux': Constant(-0.22)},
                outflow_tag: {'elev': Constant(0.397)},
            }
        }
        return boundary_conditions

    def set_initial_condition(self, prob):
        u, eta = prob.fwd_solutions[0].split()
        u.interpolate(as_vector([0.51, 0.0]))
        eta.assign(0.397)

    def get_update_forcings(self, prob, i):
        u, eta = prob.fwd_solutions[i].split()
        bathymetry_displacement = prob.equations[i].shallow_water.depth.wd_bathymetry_displacement
        depth = prob.depth[i]

        #def update_forcings(t):

            # Update bathymetry and friction
            #if self.friction == 'nikuradse':
            #    if self.wetting_and_drying:
            #        depth.project(eta + bathymetry_displacement(eta) + prob.bathymetry[i])
            #    prob.fields[i].quadratic_drag_coefficient.interpolate(self.get_cfactor(depth))

        return None #update_forcings

    def get_export_func(self, prob, i):
        eta_tilde = Function(prob.P1DG[i], name="Modified elevation")
        #self.eta_tilde_file._topology = None
        if self.plot_timeseries:
            u, eta = prob.fwd_solutions[i].split()
            b = prob.bathymetry[i]
            wd = Function(prob.P1DG[i], name="Heaviside approximation")

        def export_func():
            eta_tilde.project(self.get_eta_tilde(prob, i))
            #self.eta_tilde_file.write(eta_tilde)
            u, eta = prob.fwd_solutions[i].split()
            self.uv_file.write(u)
            #if self.plot_timeseries:

                # Store modified bathymetry timeseries
            #    wd.project(heaviside_approx(-eta-b, self.wetting_and_drying_alpha))
            #    self.wd_obs.append([wd.at([x, 0]) for x in self.xrange])

        return export_func

