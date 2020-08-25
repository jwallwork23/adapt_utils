from thetis import *
from thetis.configuration import *

from adapt_utils.swe.morphological_options import MorphOptions
from adapt_utils.misc import heaviside_approx

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)


__all__ = ["BalzanoOptions"]



class BalzanoOptions(MorphOptions):

    """
    Parameters for test case described in [1].

    [1] A. Balzano, "Evaluation of methods for numerical simulation of wetting and drying in
        shallow water flow models." Coastal Engineering 34.1-2 (1998): 83-107.
    """

    def __init__(self, friction='manning', plot_timeseries=False, nx=1, ny = 1, **kwargs):
        super(BalzanoOptions, self).__init__(**kwargs)
        self.plot_timeseries = plot_timeseries
        self.basin_x = 13800.0  # Length of wet region

        self.default_mesh = RectangleMesh(17*nx, ny, 1.5*self.basin_x, 1200.0)
        P1DG = FunctionSpace(self.default_mesh, "DG", 1)
        self.eta_tilde = Function(P1DG, name='Modified elevation')
        self.V = FunctionSpace(self.default_mesh, "CG", 1)
        self.vector_cg = VectorFunctionSpace(self.default_mesh, "CG", 1)
        
        self.plot_pvd = True        
                
        self.num_hours = 24

        # Physical
        self.base_viscosity = 1e-6
        self.base_diffusivity = 0.15

        self.tracer_init_value = Constant(1e-5)
        self.gravity = Constant(9.81)
        self.porosity = Constant(0.4)
        self.ks = 0.025
        
        self.solve_tracer = True
        self.wetting_and_drying = True
        self.wetting_and_drying_alpha = Constant(0.43)
        try:
            assert friction in ('nikuradse', 'manning')
        except AssertionError:
            raise ValueError("Friction parametrisation '{:s}' not recognised.".format(friction))
        self.friction = friction
        self.average_size = 200e-6  # Average sediment size
        self.friction_coeff = 0.025

        self.morfac = 1


        # Initial
        self.uv_init = as_vector([1.0e-7, 0.0])
        self.eta_init = Constant(0.0)

        self.get_initial_depth(VectorFunctionSpace(self.default_mesh, "CG", 2)*P1DG)       
        
        self.set_up_suspended(self.default_mesh)
        

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
        self.dt_per_remesh = 6
        self.timestepper = 'CrankNicolson'
        # self.implicitness_theta = 0.5  # TODO

        # Adaptivity
        self.h_min = 1e-8
        self.h_max = 10.

        # Goal-Oriented
        self.qoi_mode = 'inundation_volume'


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


    def set_source_tracer(self, fs, solver_obj, init = False):
        self.coeff = Function(self.depth.function_space()).project(self.coeff)
        self.ceq = Function(self.depth.function_space()).project(self.ceq)
        if init:
            self.testtracer = Function(self.depth.function_space()).project(self.testtracer)
            self.source = Function(self.depth.function_space()).project(-(self.settling_velocity*self.coeff*self.testtracer/self.depth)+ (self.settling_velocity*self.ceq/self.depth))
        else:
            self.source = Function(self.depth.function_space()).project(-(self.settling_velocity*self.coeff*solver_obj.fields.tracer_2d/self.depth)+ (self.settling_velocity*self.ceq/self.depth))
                        
        return self.source

    def get_cfactor(self):
        try:
            assert hasattr(self, 'depth')
        except AssertionError:
            raise ValueError("Depth is undefined.")

        self.ksp = Constant(3*self.average_size)
        hc = conditional(self.depth > 0.001, self.depth, 0.001)
        aux = max_value(11.036*hc/self.ksp, 1.001)
        return 2*(0.4**2)/(ln(aux)**2)

    def set_manning_drag_coefficient(self, fs):
        if self.friction == 'manning':
            self.manning_drag_coefficient = Constant(self.friction_coeff or 0.02)
        return self.manning_drag_coefficient

    def set_bathymetry(self, fs, **kwargs):
        max_depth = 5.0
        x, y = SpatialCoordinate(fs.mesh())
        self.bathymetry = Function(fs, name="Bathymetry")
        self.bathymetry.interpolate((1.0 - x/self.basin_x)*max_depth)
        return self.bathymetry

    def set_viscosity(self, fs):
        self.viscosity = Function(fs)
        self.viscosity.assign(self.base_viscosity)
        return self.viscosity


    def set_coriolis(self, fs):
        return

    def set_boundary_conditions(self, fs):
        if not hasattr(self, 'elev_in'):
            self.set_boundary_surface()
        self.elev_in.assign(self.elev_func(0.0))
        inflow_tag = 1
        outflow_tag = 2
        bottom_wall_tag = 3
        top_wall_tag = 4
        boundary_conditions = {}
        boundary_conditions[inflow_tag] = {'elev': self.elev_in}
        boundary_conditions[outflow_tag] = {'un': Constant(0.0)}
        boundary_conditions[bottom_wall_tag] = {'un': Constant(0.0)}
        boundary_conditions[top_wall_tag] = {'un': Constant(0.0)}
        return boundary_conditions
    
    def set_boundary_conditions_tracer(self, fs):
        inflow_tag = 1
        outflow_tag = 2
        bottom_wall_tag = 3
        top_wall_tag = 4
        boundary_conditions = {}
        boundary_conditions[inflow_tag] = {'value': self.tracer_init_value}
        print(self.tracer_init_value.dat.data[:])

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

        u.interpolate(self.uv_init)
        eta.assign(self.eta_init)
        self.tracer_init = Function(eta.function_space(), name="Tracer Initial condition").project(self.tracer_init_value)

        
        return self.initial_value#, self.tracer_init_value

    def get_update_forcings(self, solver_obj):


        def update_forcings(t):
            self.uv1, self.eta = solver_obj.fields.solution_2d.split()
            self.u_cg.project(self.uv1)
            self.elev_cg.project(self.eta)
            bathymetry_displacement =   solver_obj.eq_sw.bathymetry_displacement_mass_term.wd_bathymetry_displacement
            
            # Update depth
            if self.wetting_and_drying:
                self.depth.project(self.eta + bathymetry_displacement(self.eta) + self.bathymetry)
            else:
                self.depth.project(self.eta + self.bathymetry)
                
            self.update_boundary_conditions(t=t)
            self.qfc.interpolate(self.get_cfactor())
            
            # calculate skin friction coefficient
            self.hclip.interpolate(conditional(self.ksp > self.depth, self.ksp, self.depth))
            self.cfactor.interpolate(conditional(self.depth > self.ksp, 2*((2.5*ln(11.036*self.hclip/self.ksp))**(-2)), Constant(0.0)))
            
            self.update_suspended(solver_obj)


        return update_forcings

    def get_export_func(self, solver_obj):
        bathymetry_displacement = solver_obj.eq_sw.bathymetry_displacement_mass_term.wd_bathymetry_displacement
        eta = solver_obj.fields.elev_2d
        b = solver_obj.fields.bathymetry_2d
        def export_func():
            self.eta_tilde.project(eta + bathymetry_displacement(eta))
            self.eta_tilde_file.write(self.eta_tilde)

            if self.plot_timeseries:

                # Store modified bathymetry timeseries
                P1DG = solver_obj.function_spaces.P1DG_2d
                wd = project(heaviside_approx(-eta-b, self.wetting_and_drying_alpha), P1DG)
                self.wd_obs.append([wd.at([x, 0]) for x in self.xrange])

                # Store QoI timeseries
                self.evaluate_qoi_form(solver_obj)
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
        b = solver_obj.fields.bathymetry_2d
        dry = conditional(ge(b, 0), 0, 1)
        if 'inundation' in self.qoi_mode:
            f = heaviside_approx(eta + b, self.wetting_and_drying_alpha)
            eta_init = project(self.initial_value.split()[1], eta.function_space())
            f_init = heaviside_approx(eta_init + b, self.wetting_and_drying_alpha)
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
        self.plot_heaviside()
        if 'volume' in self.qoi_mode:
            self.plot_qoi()
        print_output("QoI '{:s}' = {:.4e}".format(self.qoi_mode, self.evaluate_qoi()))

    def plot_heaviside(self):
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
        plt.savefig(os.path.join(self.di, "heaviside_timeseries.pdf"))

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

