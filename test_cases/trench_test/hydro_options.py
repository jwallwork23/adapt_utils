from thetis import *
from thetis.configuration import *

from adapt_utils.swe.morphological_options import TracerOptions

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)


__all__ = ["TrenchHydroOptions"]


class TrenchHydroOptions(TracerOptions):
    """
    Parameters for test case described in [1].

    [1] Clare, Mariana, et al. “Hydro-morphodynamics 2D Modelling Using a Discontinuous Galerkin Discretisation.” 
    EarthArXiv, 9 Jan. 2020. Web.
    """

    def __init__(self, friction='manning', plot_timeseries=False, nx=1, ny = 1, **kwargs):
        self.plot_timeseries = plot_timeseries
        

        self.default_mesh = RectangleMesh(16*5*nx, 5*ny, 16, 1.1)
        self.P1DG = FunctionSpace(self.default_mesh, "DG", 1)  # FIXME
        self.V = FunctionSpace(self.default_mesh, "CG", 1)
        self.vector_cg = VectorFunctionSpace(self.default_mesh, "CG", 1)
        
        super(TrenchHydroOptions, self).__init__(**kwargs)
        self.plot_pvd = True        

        # Physical
        self.base_viscosity = 1e-6
                
        self.gravity = Constant(9.81)
        
        self.solve_tracer = False
        self.wetting_and_drying = False
        #self.wetting_and_drying_alpha = Constant(0.43)
        try:
            assert friction in ('nikuradse', 'manning')
        except AssertionError:
            raise ValueError("Friction parametrisation '{:s}' not recognised.".format(friction))
        self.friction = friction
        self.average_size = 160e-6  # Average sediment size
        

        # Initial
        self.uv_init = as_vector([0.51, 0.0])
        self.eta_init = Constant(0.4)

        self.get_initial_depth(VectorFunctionSpace(self.default_mesh, "CG", 2)*self.P1DG)       
                
        # Stabilisation
        # self.stabilisation = 'no'
        self.grad_depth_viscosity = True


        # Time integration
        self.dt = 0.25
        self.end_time = 500
        #self.dt_per_export = self.end_time/(40*self.dt)
        #self.dt_per_remesh = self.end_time/(40*self.dt)
        self.timestepper = 'CrankNicolson'
        self.implicitness_theta = 1.0

        # Goal-Oriented
        self.qoi_mode = 'inundation_volume'


        # Timeseries
        self.wd_obs = []
        #self.trange = np.linspace(0.0, self.end_time, self.num_hours+1)
        #tol = 1e-8  # FIXME: Point evaluation hack
        #self.xrange = np.linspace(tol, 16-tol, 20)
        self.qois = []    

    def set_quadratic_drag_coefficient(self, fs):
        if self.friction == 'nikuradse':
            self.quadratic_drag_coefficient = project(self.get_cfactor(), self.depth.function_space())
        return self.quadratic_drag_coefficient

    def get_cfactor(self):
        try:
            assert hasattr(self, 'depth')
        except AssertionError:
            raise ValueError("Depth is undefined.")
        
        self.ksp = Constant(3*self.average_size)
        hclip = Function(self.P1DG).interpolate(conditional(self.ksp > self.depth, self.ksp, self.depth))
        aux = 11.036*hclip/self.ksp
        return conditional(self.depth>self.ksp, 2*(0.4**2)/(ln(aux)**2), 0.0)

    def set_manning_drag_coefficient(self, fs):
        if self.friction == 'manning':
            self.manning_drag_coefficient = Constant(self.friction_coeff or 0.02)
        return self.manning_drag_coefficient

    def set_bathymetry(self, fs, **kwargs):
        initial_depth = Constant(0.397)
        depth_riv = Constant(initial_depth - 0.397)
        depth_trench = Constant(depth_riv - 0.15)
        depth_diff = depth_trench - depth_riv
        x, y = SpatialCoordinate(fs.mesh())
        trench = conditional(le(x, 5), depth_riv, conditional(le(x,6.5), (1/1.5)*depth_diff*(x-6.5) + depth_trench,\
                conditional(le(x, 9.5), depth_trench, conditional(le(x, 11), -(1/1.5)*depth_diff*(x-11) + depth_riv, depth_riv))))
        self.bathymetry = Function(fs, name="Bathymetry")
        self.bathymetry.interpolate(-trench)
        return self.bathymetry

    def set_viscosity(self, fs):
        self.viscosity = Function(fs)
        self.viscosity.assign(self.base_viscosity)
        return self.viscosity

    def set_coriolis(self, fs):
        return

    def set_boundary_conditions(self, fs):
        inflow_tag = 1
        outflow_tag = 2
        bottom_wall_tag = 3
        top_wall_tag = 4
        boundary_conditions = {}
        boundary_conditions[inflow_tag] = {'flux': Constant(-0.22)}
        boundary_conditions[outflow_tag] = {'elev': Constant(0.397)}
        return boundary_conditions

    def update_boundary_conditions(self, t=0.0):
        return None

    def set_initial_condition(self, fs):
        """
        Set initial elevation and velocity using asymptotic solution.

        :arg fs: `FunctionSpace` in which the initial condition should live.
        """
        self.initial_value = Function(fs, name="Initial condition")
        u, eta = self.initial_value.split()
        u.interpolate(self.uv_init)
        eta.assign(self.eta_init)
        
        return self.initial_value

    def get_update_forcings(self, solver_obj):

        def update_forcings(t):
            self.uv1, self.eta = solver_obj.fields.solution_2d.split()
            
            # Update depth
            if self.wetting_and_drying:
                bathymetry_displacement = solver_obj.eq_sw.bathymetry_displacement_mass_term.wd_bathymetry_displacement
                self.depth.project(self.eta + bathymetry_displacement(self.eta) + self.bathymetry)
            else:
                self.depth.project(self.eta + self.bathymetry)
                
            self.quadratic_drag_coefficient.interpolate(self.get_cfactor())
            
                        

        return update_forcings

    def get_export_func(self, solver_obj):
        return None

    def set_qoi_kernel(self, solver_obj):
        #J = self.evaluate_qoi_form(solver_obj)
        #eta = solver_obj.fields.solution_2d.split()[1]
        #dJdeta = derivative(J, eta, TestFunction(eta.function_space()))  # TODO: test
        return None

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
