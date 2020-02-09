from thetis import *
from thetis.configuration import *

#from adapt_utils.test_cases.trench_test.hydro_options import TrenchHydroOptions
from adapt_utils.swe.morphological_options import TracerOptions

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)


__all__ = ["TrenchOptions"]


class TrenchOptions(TracerOptions):
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
        self.vector_dg = VectorFunctionSpace(self.default_mesh, "DG", 1)
        
        super(TrenchOptions, self).__init__(**kwargs)
        self.plot_pvd = True  
        self.di = "morph_output"

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
                

        self.grad_depth_viscosity = True        

        
        self.bathymetry_file = File(self.di + "/bathy.pvd")
                
        self.num_hours = 5

        # Physical
        self.base_diffusivity = 0.15


        self.porosity = Constant(0.4)
        self.ks = 0.025

        self.solve_tracer = True

        try:
            assert friction in ('nikuradse', 'manning')
        except AssertionError:
            raise ValueError("Friction parametrisation '{:s}' not recognised.".format(friction))
        self.friction = friction 
        self.morfac = 100


        # Initial
        input_dir = 'hydrodynamics_trench'

        # Initial
        self.uv_init = as_vector([0.51, 0.0])
        self.eta_init = Constant(0.4)

        #self.eta_init, self.uv_init = self.initialise_fields(input_dir, self.di)        

        self.get_initial_depth(VectorFunctionSpace(self.default_mesh, "CG", 2)*self.P1DG)       
        
        self.set_up_suspended()
        
        self.tracer_init_value = Constant(self.ceq.at([0,0])/self.coeff.at([0,0]))
        self.tracer_init = Function(self.P1DG, name="Tracer Initial condition").project(self.tracer_init_value)        
        
        # Stabilisation
        self.stabilisation = 'no'

        # Time integration
        self.dt = 0.3
        self.end_time = self.num_hours*3600.0/self.morfac
        self.dt_per_export = 40
        self.dt_per_remesh = 100
        self.timestepper = 'CrankNicolson'
        self.implicitness_theta = 1.0

        # Adaptivity
        self.h_min = 1e-8
        self.h_max = 10.

        # Goal-Oriented
        self.qoi_mode = 'inundation_volume'


        # Timeseries
        self.wd_obs = []
        self.trange = np.linspace(0.0, self.end_time, self.num_hours+1)
        tol = 1e-8  # FIXME: Point evaluation hack
        self.xrange = np.linspace(tol, 16-tol, 20)
        self.qois = []
        

    def set_source_tracer(self, fs, solver_obj, init = False):
        self.coeff = Function(self.depth.function_space()).project(self.coeff)
        self.ceq = Function(self.depth.function_space()).project(self.ceq)
        if init:
            self.testtracer = Function(self.depth.function_space()).project(self.testtracer)
            self.source = Function(self.depth.function_space()).project(-(self.settling_velocity*self.coeff*self.testtracer/self.depth)+ (self.settling_velocity*self.ceq/self.depth))
        else:
            self.source = Function(self.depth.function_space()).project(-(self.settling_velocity*self.coeff*solver_obj.fields.tracer_2d/self.depth)+ (self.settling_velocity*self.ceq/self.depth))
                        
        return self.source

    
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
        import ipdb; ipdb.set_trace()
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


    def set_boundary_conditions_tracer(self, fs):
        inflow_tag = 1
        outflow_tag = 2
        bottom_wall_tag = 3
        top_wall_tag = 4
        boundary_conditions = {}
        boundary_conditions[inflow_tag] = {'value': self.tracer_init_value}
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
        u.project(self.uv_init)
        eta.project(self.eta_init)
        
        return self.initial_value

    def get_update_forcings(self, solver_obj):

        def update_forcings(t):
            self.uv1, self.eta = solver_obj.fields.solution_2d.split()
            self.u_cg.project(self.uv1)
            self.elev_cg.project(self.eta)
            
            # Update depth
            if self.wetting_and_drying:
                bathymetry_displacement =   solver_obj.eq_sw.bathymetry_displacement_mass_term.wd_bathymetry_displacement
                self.depth.project(self.eta + bathymetry_displacement(self.eta) + self.bathymetry)
            else:
                self.depth.project(self.eta + self.bathymetry)
            
            self.hc.interpolate(conditional(self.depth > 0.001, self.depth, 0.001))
            self.aux.interpolate(conditional(11.036*self.hc/self.ks > 1.001, 11.036*self.hc/self.ks, 1.001))
            self.qfc.interpolate(2/(ln(self.aux)/0.4)**2)
    
            # calculate skin friction coefficient
            self.cfactor.interpolate(self.get_cfactor())

            self.quadratic_drag_coefficient.interpolate(self.get_cfactor())
            
            self.update_suspended(solver_obj)
            
            self.bathymetry_file.write(self.bathymetry)
                        

        return update_forcings

    def initialise_fields(self, inputdir, outputdir):
        """
        Initialise simulation with results from a previous simulation
        """     

        # elevation
        with timed_stage('initialising elevation'):
            chk = DumbCheckpoint(inputdir + "/elevation", mode=FILE_READ)
            elev_init = Function(self.P1DG, name="elevation")
            chk.load(elev_init)
            File(outputdir + "/elevation_imported.pvd").write(elev_init)
            chk.close()
        # velocity
        with timed_stage('initialising velocity'):
            chk = DumbCheckpoint(inputdir + "/velocity" , mode=FILE_READ)
            uv_init = Function(self.vector_dg, name="velocity")
            chk.load(uv_init)
            File(outputdir + "/velocity_imported.pvd").write(uv_init)
            chk.close()
        return  elev_init, uv_init,
