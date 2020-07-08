from thetis import *
from thetis.configuration import *

from adapt_utils.unsteady.options import CoupledOptions
from thetis.options import ModelOptions2d
from thetis.sediments_adjoint import SedimentModel

import os
import time
import datetime
import numpy as np
from matplotlib import rc

rc('text', usetex=True)


__all__ = ["TrenchOptions"]


# TODO: NEEDS UPDATING
class TrenchOptions(CoupledOptions):
    """
    Parameters for test case described in [1].

    [1] Clare, Mariana, et al. “Hydro-morphodynamics 2D Modelling Using a Discontinuous Galerkin Discretisation.”
    EarthArXiv, 9 Jan. 2020. Web.
    """

    def __init__(self, friction='manning', plot_timeseries=False, nx=1, ny=1, mesh = None, input_dir = None, output_dir = None, **kwargs):
        super(TrenchOptions, self).__init__(**kwargs)
        self.slope_eff = False
        self.angle_correction = False
        self.convective_vel_flag = True
        self.wetting_and_drying = False
        self.conservative = False
        self.depth_integrated = False
        self.suspended = True
        self.bedload = True
        self.implicit_source = False
        self.fixed_tracer = None        
        
        #self.solve_swe = True
        self.solve_tracer = False
        self.plot_timeseries = plot_timeseries
        #self.default_mesh = RectangleMesh(np.int(16*5*nx), 5*ny, 16, 1.1)
        self.plot_pvd = True

        if mesh is None:
            self.t_old = Constant(0.0)
            self.default_mesh = RectangleMesh(np.int(16*5*nx), 5*ny, 16, 1.1)
            self.P1DG = FunctionSpace(self.default_mesh, "DG", 1)
            self.P1 = FunctionSpace(self.default_mesh, "CG", 1)
            self.P1_vec = VectorFunctionSpace(self.default_mesh, "CG", 1)
            self.P1_vec_dg = VectorFunctionSpace(self.default_mesh, "DG", 1)
        else:
            self.P1DG = FunctionSpace(mesh, "DG", 1)
            self.P1 = FunctionSpace(mesh, "CG", 1)
            self.P1_vec = VectorFunctionSpace(mesh, "CG", 1)
            self.P1_vec_dg = VectorFunctionSpace(mesh, "DG", 1)   

        # Physics
        self.base_viscosity = 1e-6
        self.base_diffusivity = 0.15
        self.gravity = Constant(9.81)
        self.porosity = Constant(0.4)
        self.ks = 0.025
        try:
            assert friction in ('nikuradse', 'manning')
        except AssertionError:
            raise ValueError("Friction parametrisation '{:s}' not recognised.".format(friction))
        self.friction = friction
        self.average_size = 160e-6  # Average sediment size
        self.ksp = 3*self.average_size
        self.morfac = 100

        # Model
        self.wetting_and_drying = False
        self.grad_depth_viscosity = True

        # I/O
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        self.input_dir = 'hydrodynamics_trench'
        outputdir = 'outputs' + st
        self.di = outputdir  # "morph_output"
        self.tracer_list = []
        self.bathymetry_file = File(self.di + "/bathy.pvd")

        self.convective_vel_flag = True

        self.t_old = Constant(0.0)

        self.slope_eff = True
        self.angle_correction = True

        # Stabilisation
        self.stabilisation = None #'lax_friedrichs'

        self.num_hours = 15

        self.t_old = Constant(0.0)

        self.morfac = Constant(100)
        self.norm_smoother_constant = Constant(0.1)


        if mesh is None:
            self.set_up_morph_model(self.input_dir, self.default_mesh)
        else:
            self.set_up_morph_model(self.input_dir, mesh)

        # Time integration
        self.dt = 0.3
        self.end_time = float(self.num_hours*3600.0/self.morfac)
        self.dt_per_export = 40
        self.dt_per_remesh = 40
        self.timestepper = 'CrankNicolson'
        self.implicitness_theta = 1.0

        # Adaptivity
        self.h_min = 1e-8
        self.h_max = 10.

        # Timeseries
        self.wd_obs = []
        self.trange = np.linspace(0.0, self.end_time, self.num_hours+1)
        tol = 1e-8  # FIXME: Point evaluation hack
        self.xrange = np.linspace(tol, 16-tol, 20)
        self.qois = []

        # Outputs  (NOTE: self.di has changed)
        self.bath_file = File(os.path.join(self.di, 'bath_export.pvd'))
        
    def set_up_morph_model(self, input_dir, mesh = None):

        # Outputs
        self.bath_file = File(os.path.join(self.di, 'bath_export.pvd'))     

        # Physical
        self.base_viscosity = 1e-6        
        self.base_diffusivity = 0.18161630470135287

        self.porosity = Constant(0.4)
        self.ks = Constant(0.025)
        self.average_size = 160*(10**(-6))  # Average sediment size        

        self.wetting_and_drying = False
        self.conservative = False
        self.implicit_source = False
        self.slope_eff = True
        self.angle_correction = True
        self.solve_tracer = False
        self.suspended = True
        self.convectivevel_flag = True
        self.bedload = True

        # Initial
        self.elev_init, self.uv_init = self.initialise_fields(mesh, input_dir, self.di)

        self.uv_d = Function(self.P1_vec_dg).project(self.uv_init)

        self.eta_d = Function(self.P1DG).project(self.elev_init)

        if not hasattr(self, 'bathymetry') or self.bathymetry is None:
            self.bathymetry = self.set_bathymetry(self.P1)


        if self.suspended:
            self.tracer_init = None
            
        self.sediment_model = SedimentModel(ModelOptions2d, suspendedload=self.suspended, convectivevel=self.convective_vel_flag,
                            bedload=self.bedload, angle_correction=self.angle_correction, slope_eff=self.slope_eff, seccurrent=False,
                            mesh2d=mesh, bathymetry_2d=self.bathymetry,
                            uv_init = self.uv_d, elev_init = self.eta_d, ks=self.ks, average_size=self.average_size, 
                            cons_tracer = self.conservative, wetting_and_drying = self.wetting_and_drying, wetting_alpha = self.wetting_and_drying_alpha)            

    def set_bathymetry(self, fs, **kwargs):

        initial_depth = Constant(0.397)
        depth_riv = Constant(initial_depth - 0.397)
        depth_trench = Constant(depth_riv - 0.15)
        depth_diff = depth_trench - depth_riv
        x, y = SpatialCoordinate(fs.mesh())
        trench = conditional(le(x, 5), depth_riv, conditional(le(x, 6.5), (1/1.5)*depth_diff*(x-6.5) + depth_trench,
                             conditional(le(x, 9.5), depth_trench, conditional(le(x, 11), -(1/1.5)*depth_diff*(x-11) + depth_riv, depth_riv))))
        self.bathymetry = Function(fs, name="Bathymetry")
        self.bathymetry.interpolate(-trench)
        return self.bathymetry

    def set_boundary_conditions(self, prob, i):
        inflow_tag = 1
        outflow_tag = 2
        bottom_wall_tag = 3
        top_wall_tag = 4
        boundary_conditions = {
            'shallow_water': {
                inflow_tag: {'flux': Constant(-0.22)},
                outflow_tag: {'elev': Constant(0.397)},
            },
	   'tracer': {
                inflow_tag: {'value': self.sediment_model.sediment_rate}
            }
        }
        return boundary_conditions


    def set_initial_condition(self, prob):
        """
        Set initial elevation and velocity using asymptotic solution.

        :arg fs: `FunctionSpace` in which the initial condition should live.
        """
        fs = prob.fwd_solutions[0].function_space()      
        self.initial_value = Function(fs, name="Initial condition")
        u, eta = self.initial_value.split()
        u.project(self.uv_init)
        eta.project(self.elev_init)
        return self.initial_value
    

    def get_update_forcings(self, prob, i):
        return None


    def initialise_fields(self, mesh2d, inputdir, outputdir):
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

        DG_2d = FunctionSpace(mesh, "DG", 1)
        # elevation
        with timed_stage('initialising elevation'):
            chk = DumbCheckpoint(inputdir + "/elevation", mode=FILE_READ)
            elev_init = Function(DG_2d, name="elevation")
            chk.load(elev_init)
            File(outputdir + "/elevation_imported.pvd").write(elev_init)
            chk.close()
        # velocity
        with timed_stage('initialising velocity'):
            chk = DumbCheckpoint(inputdir + "/velocity", mode=FILE_READ)
            V = VectorFunctionSpace(mesh, "DG", 1)
            uv_init = Function(V, name="velocity")
            chk.load(uv_init)
            File(outputdir + "/velocity_imported.pvd").write(uv_init)
            chk.close()
        return elev_init, uv_init,


    def get_export_func(self, prob, i):
        self.bath_export = prob.bathymetry[0]

        def export_func():
            self.bath_file.write(self.bath_export)
        return export_func
