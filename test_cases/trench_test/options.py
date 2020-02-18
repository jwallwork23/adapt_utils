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
        self.P1 = FunctionSpace(self.default_mesh, "CG", 1)
        self.P1_vec = VectorFunctionSpace(self.default_mesh, "CG", 1)
        self.P1_vec_dg = VectorFunctionSpace(self.default_mesh, "DG", 1)
        
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

        
        self.grad_depth_viscosity = True        

        self.tracer_list = []
        
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


        self.eta_init, self.uv_init = self.initialise_fields(input_dir, self.di)   
        self.uv_d = Function(self.P1_vec_dg).project(self.uv_init)
        self.eta_d = Function(self.P1DG).project(self.eta_init)        
        
        self.convective_vel_flag = True
        
        self.t_old = Constant(0.0)        
        
        self.slope_eff = False
        self.set_up_suspended(self.default_mesh)
        self.set_up_bedload(self.default_mesh)
        
        # Stabilisation
        self.stabilisation = 'lax_friedrichs'

        # Time integration
        
        self.dt = 0.3
        self.end_time = self.num_hours*3600.0/self.morfac
        self.dt_per_export = 60
        self.dt_per_remesh = 60
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
        

    def set_source_tracer(self, fs, solver_obj = None, init = False, t_old = Constant(100)):
        if init:
            if t_old.dat.data[:] == 0.0:
                self.source = Function(fs).project(-(self.settling_velocity*self.coeff*self.tracer_init_value/self.depth)+ (self.settling_velocity*self.ceq/self.depth))
            else:
                self.source = Function(fs).project(-(self.settling_velocity*self.coeff*self.tracer_interp/self.depth)+ (self.settling_velocity*self.ceq/self.depth))
        else:
            self.source.interpolate(-(self.settling_velocity*self.coeff*solver_obj.fields.tracer_2d/self.depth)+(self.settling_velocity*self.ceq/self.depth))
        return self.source

    
    def set_quadratic_drag_coefficient(self, fs):
        if self.friction == 'nikuradse':
            self.quadratic_drag_coefficient = project(self.get_cfactor(), fs)
        return self.quadratic_drag_coefficient

    def get_cfactor(self):
        try:
            assert hasattr(self, 'depth')
        except AssertionError:
            raise ValueError("Depth is undefined.")
        self.ksp = Constant(3*self.average_size)
        hclip = Function(self.P1DG).interpolate(conditional(self.ksp > self.depth, self.ksp, self.depth))
        return Function(self.P1DG).interpolate(conditional(self.depth>self.ksp, 2*((2.5*ln(11.036*hclip/self.ksp))**(-2)), Constant(0.0)))

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
        #boundary_conditions[bottom_wall_tag] = {'un': Constant(0.0)}
        #boundary_conditions[top_wall_tag] = {'un': Constant(0.0)}        
        return boundary_conditions


    def set_boundary_conditions_tracer(self, fs):
        inflow_tag = 1
        outflow_tag = 2
        bottom_wall_tag = 3
        top_wall_tag = 4
        boundary_conditions = {}
        boundary_conditions[inflow_tag] = {'value': self.tracer_init_value}
        return boundary_conditions


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
            
            if round(t, 2)%18.0 == 0:
                if self.t_old.dat.data[:] == t:
                    bath_file = File(self.di + '/bath_timestep.pvd')
                    bath_file.write(solver_obj.fields.bathymetry_2d)  
                    import ipdb; ipdb.set_trace()

            self.tracer_list.append(min(solver_obj.fields.tracer_2d.dat.data[:]))

            self.update_key_hydro(solver_obj)

            if self.t_old.dat.data[:] == t:
                print(t)
                self.update_suspended(solver_obj)
                #self.update_bedload(solver_obj)
                solve(self.f==0, self.z_n1)
        
                self.bathymetry.assign(self.z_n1)
                solver_obj.fields.bathymetry_2d.assign(self.z_n1)
                print(max(self.bathymetry.dat.data[:]))                
            
            self.t_old.assign(t)        

        return update_forcings

    def initialise_fields(self, inputdir, outputdir):
        """
        Initialise simulation with results from a previous simulation
        """     
        from firedrake.petsc import PETSc

        try:
            import firedrake.cython.dmplex as dmplex
        except:
            import firedrake.dmplex as dmplex  # Older version        

        # mesh
        with timed_stage('mesh'):
            # Load
            newplex = PETSc.DMPlex().create()
            newplex.createFromFile(inputdir + '/myplex.h5')
            mesh = Mesh(newplex)
            
        DG_2d = FunctionSpace(mesh, 'DG', 1)  
        vector_dg = VectorFunctionSpace(mesh, 'DG', 1)          
        # elevation
        with timed_stage('initialising elevation'):
            chk = DumbCheckpoint(inputdir + "/elevation", mode=FILE_READ)
            elev_init = Function(DG_2d, name="elevation")
            chk.load(elev_init)
            File(outputdir + "/elevation_imported.pvd").write(elev_init)
            chk.close()
        # velocity
        with timed_stage('initialising velocity'):
            chk = DumbCheckpoint(inputdir + "/velocity" , mode=FILE_READ)
            uv_init = Function(vector_dg, name="velocity")
            chk.load(uv_init)
            File(outputdir + "/velocity_imported.pvd").write(uv_init)
            chk.close()
        return  elev_init, uv_init,
