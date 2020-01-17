from thetis import *
from thetis.configuration import *

from adapt_utils.swe.options import ShallowWaterOptions
from adapt_utils.misc.misc import heavyside_approx

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)


__all__ = ["BalzanoOptions"]


class BalzanoOptions(ShallowWaterOptions):
    """
    Parameters for test case in [...].
    """ # TODO: cite

    def __init__(self, approach='fixed_mesh', friction='manning', plot_timeseries=False):
        super(BalzanoOptions, self).__init__(approach)
        self.plot_pvd = True
        self.plot_timeseries = plot_timeseries

        # Initial mesh
        try:
            assert os.path.exists('strip.msh')  # TODO: abspath
        except AssertionError:
            raise ValueError("Mesh does not exist or cannot be found. Please build it.")
        self.default_mesh = Mesh('strip.msh')
        self.basin_x = 13800.0  # Length of wet region
        self.num_hours = 24

        # Physical
        self.base_viscosity = 1e-6
        self.base_diffusivity = 0.15
        self.wetting_and_drying = True
        self.wetting_and_drying_alpha = Constant(0.43)
        try:
            assert friction in ('nikuradse', 'manning')
        except AssertionError:
            raise ValueError("Friction parametrisation '{:s}' not recognised.".format(friction))
        self.friction = friction
        self.average_size = 200e-6  # Average sediment size
        self.friction_coeff = 0.025

        # Stabilisation
        self.stabilisation = 'no'

        # Boundary conditions
        h_amp = 0.5  # Ocean boundary forcing amplitude
        h_T = self.num_hours/2*3600  # Ocean boundary forcing period
        self.elev_func = lambda t: h_amp*(-cos(2*pi*(t-(6*3600))/h_T)+1)

        # Time integration
        self.dt = 600.0
        self.end_time = self.num_hours*3600.0
        self.dt_per_export = 6
        self.dt_per_remesh = 20
        self.timestepper = 'CrankNicolson'
        # self.implicitness_theta = 0.5  # TODO

        # Adaptivity
        self.h_min = 1e-8
        self.h_max = 10.

        # Goal-Oriented
        self.qoi_mode = 'inundation_volume'

        # Outputs
        self.wd_bath_file = File(os.path.join(self.di, 'moving_bath.pvd'))
        P1DG = FunctionSpace(self.default_mesh, "DG", 1)  # FIXME
        self.moving_bath = Function(P1DG, name='Moving bathymetry')
        self.eta_tilde_file = File(os.path.join(self.di, 'eta_tilde.pvd'))
        self.eta_tilde = Function(P1DG, name='Modified elevation')

        # Timeseries
        self.wd_obs = []
        self.trange = np.linspace(0.0, self.end_time, self.num_hours+1)
        tol = 1e-8  # FIXME: Point evaluation hack
        self.xrange = np.linspace(tol, 1.5*self.basin_x-tol, 20)
        self.qois = []

    def set_quadratic_drag_coefficient(self, fs):
        if self.friction == 'nikuradse':
            self.quadratic_drag_coefficient = interpolate(self.get_cfactor(), fs)
        return self.quadratic_drag_coefficient

    def get_cfactor(self):
        try:
            assert hasattr(self, 'depth')
        except AssertionError:
            raise ValueError("Depth is undefined.")
        ksp = Constant(3*self.average_size)
        hc = conditional(self.depth > 0.001, self.depth, 0.001)
        aux = max_value(11.036*hc/ksp, 1.001)
        return 2*(0.4**2)/(ln(aux)**2)

    def set_manning_drag_coefficient(self, fs):
        if self.friction == 'manning':
            self.manning_drag_coefficient = Constant(self.friction_coeff or 0.02)
        return self.manning_drag_coefficient

    def set_bathymetry(self, fs):
        max_depth = 5.0
        x, y = SpatialCoordinate(fs.mesh())
        self.bathymetry = Function(fs, name="Bathymetry")
        self.bathymetry.interpolate((1.0 - x/self.basin_x)*max_depth)
        return self.bathymetry

    def set_viscosity(self, fs):
        self.viscosity = Function(fs)
        self.viscosity.assign(self.base_viscosity)
        return self.viscosity

    def set_boundary_conditions(self, fs):
        if not hasattr(self, 'elev_in'):
            self.set_boundary_surface()
        self.elev_in.assign(self.elev_func(0.0))
        inflow_tag = 1
        wall_tag = 2
        outflow_tag = 3
        boundary_conditions = {}
        boundary_conditions[inflow_tag] = {'elev': self.elev_in}
        boundary_conditions[wall_tag] = {'un': Constant(0.0)}
        return boundary_conditions

    def update_boundary_conditions(self, t=0.0):
        self.elev_in.assign(self.elev_func(t) if 6*3600 <= t <= 18*3600 else 0.0)

    def set_initial_condition(self, fs):
        """
        Set initial elevation and velocity using asymptotic solution.

        :arg fs: `FunctionSpace` in which the initial condition should live.
        """
        self.initial_value = Function(fs, name="Initial condition")
        u, eta = self.initial_value.split()
        u.interpolate(as_vector([1.0e-7, 0.0]))
        eta.assign(0.0)
        return self.initial_value

    def get_update_forcings(self, solver_obj):
        eta = solver_obj.fields.elev_2d
        bathymetry_displacement = solver_obj.eq_sw.bathymetry_displacement_mass_term.wd_bathymetry_displacement

        def update_forcings(t):
            self.update_boundary_conditions(t=t)

            # Update bathymetry and friction
            if self.friction == 'nikuradse':
                if self.wetting_and_drying:
                    self.depth.project(eta + bathymetry_displacement(eta) + self.bathymetry)
                self.quadratic_drag_coefficient.interpolate(self.get_cfactor())

        return update_forcings

    def get_export_func(self, solver_obj):
        bathymetry_displacement = solver_obj.eq_sw.bathymetry_displacement_mass_term.wd_bathymetry_displacement
        solution = solver_obj.fields.solution_2d
        eta = solution.split()[1]
        def export_func():
            self.moving_bath.project(self.bathymetry + bathymetry_displacement(eta))
            self.wd_bath_file.write(self.moving_bath)
            self.eta_tilde.project(eta + bathymetry_displacement(eta))
            self.eta_tilde_file.write(self.eta_tilde)

            if self.plot_timeseries:

                # Store modified bathymetry timeseries
                P1DG = solver_obj.function_spaces.P1DG_2d
                wd = project(heavyside_approx(-eta-self.bathymetry, self.wetting_and_drying_alpha), P1DG)
                self.wd_obs.append([wd.at([x, 0]) for x in self.xrange])

                # Store QoI timeseries
                self.evaluate_qoi_form(solver_obj)
                # self.set_qoi_kernel(solver_obj)
                # self.qois.append(assemble(inner(self.kernel, solution)*dx(degree=12)))
                self.qois.append(assemble(self.qoi_form))
        return export_func

    def set_qoi_kernel(self, solver_obj):
        J = self.evaluate_qoi_form(solver_obj)
        eta = solver_obj.fields.solution_2d.split()[1]
        dJdeta = derivative(J, eta, TestFunction(eta.function_space()))  # TODO: test

    def evaluate_qoi_form(self, solver_obj):
        try:
            assert self.qoi_mode in ('inundation_volume', 'maximum_inundation', 'overtopping_volume')
        except AssertionError:
            raise ValueError("QoI mode '{:s}' not recognised.".format(self.qoi_mode))
        bathymetry_displacement = solver_obj.eq_sw.bathymetry_displacement_mass_term.wd_bathymetry_displacement
        eta = solver_obj.fields.elev_2d
        dry = conditional(ge(self.bathymetry, 0), 0, 1)
        if 'inundation' in self.qoi_mode:
            f = heavyside_approx(eta + self.bathymetry, self.wetting_and_drying_alpha)
            eta_init = self.initial_value.split()[1]
            f_init = heavyside_approx(eta_init + self.bathymetry, self.wetting_and_drying_alpha)
            self.qoi_form = dry*(eta + f - f_init)*dx(degree=12)
        elif self.qoi_mode == 'overtopping_volume':
            raise NotImplementedError  # TODO: Flux over coast. (Needs an internal boundary.)
        else:
            raise NotImplementedError  # TODO: Consider others. (Speak to Branwen.)
        return self.qoi_form

    def evaluate_qoi(self):  # TODO: Do time-dep QoI properly
        f = self.qois
        N = len(f)
        assert N > 0
        if 'maximum' in self.qoi_mode:
            qoi = np.max(f)
        else:  # Trapezium rule
            h = self.dt*self.dt_per_export
            qoi = 0.5*h*(f[0] + f[N-1])
            for i in range(1, N-1):
                qoi += h*f[i]
        return qoi

    def plot(self):
        self.plot_heavyside()
        if 'volume' in self.qoi_mode:
            self.plot_qoi()
        print_output("QoI '{:s}' = {:.4e}".format(self.qoi_mode, self.evaluate_qoi()))

    def plot_heavyside(self):
        """Timeseries plot of approximate Heavyside function."""
        scaling = 0.7
        plt.figure(1, figsize=(scaling*7.0, scaling*4.0))
        plt.gcf().subplots_adjust(bottom=0.15)
        T = [[t/3600]*20 for t in self.trange]
        X = [self.xrange for t in T]

        cset1 = plt.contourf(T, X, self.wd_obs, 20, cmap=plt.cm.get_cmap('binary'))
        plt.clim(0.0, 1.2)
        cset2 = plt.contour(T, X, self.wd_obs, 20, cmap=plt.cm.get_cmap('binary'))
        plt.clim(0.0, 1.2)
        cset3 = plt.contour(T, X, self.wd_obs, 1, colors='k', linestyles='dotted', linewidths=5.0, levels = [0.5])
        cb = plt.colorbar(cset1, ticks=np.linspace(0, 1, 6))
        cb.set_label("$\mathcal H(\eta-b)$")
        plt.ylim(min(X[0]), max(X[0]))
        plt.xlabel("Time [$\mathrm h$]")
        plt.ylabel("$x$ [$\mathrm m$]")
        plt.savefig(os.path.join(self.di, "heavyside_timeseries.pdf"))

    def plot_qoi(self):
        """Timeseries plot of instantaneous QoI."""
        plt.figure(2)
        T = self.trange/3600
        qois = [q/1.0e9 for q in self.qois]
        qoi = self.evaluate_qoi()/1.0e9
        plt.plot(T, qois, linestyle='dashed', color='b', marker='x')
        plt.fill_between(T, np.zeros_like(qois), qois)
        plt.xlabel("Time [$\mathrm h$]")
        plt.ylabel("Instantaneous QoI [$\mathrm{km}^3$]")
        plt.title("Time integrated QoI: ${:.1f}\,\mathrm k\mathrm m^3\,\mathrm h$".format(qoi))
        plt.savefig(os.path.join(self.di, "qoi_timeseries_{:s}.pdf".format(self.qoi_mode)))
