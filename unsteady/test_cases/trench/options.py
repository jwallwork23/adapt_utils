from thetis import *
from thetis.configuration import *

from adapt_utils.swe.morphological_options import MorphOptions  # TODO: Will change

import os
import time
import datetime
import numpy as np
from matplotlib import rc

rc('text', usetex=True)


__all__ = ["TrenchOptions"]


# TODO: Hookup with AdaptiveMorphologicalProblem
class TrenchOptions(MorphOptions):
    """
    Parameters for test case described in [1].

    [1] Clare, Mariana, et al. “Hydro-morphodynamics 2D Modelling Using a Discontinuous Galerkin Discretisation.”
    EarthArXiv, 9 Jan. 2020. Web.
    """

    def __init__(self, friction='manning', plot_timeseries=False, nx=1, ny=1, **kwargs):
        super(TrenchOptions, self).__init__(**kwargs)
        self.solve_swe = True
        self.solve_tracer = True
        self.plot_timeseries = plot_timeseries
        self.default_mesh = RectangleMesh(np.int(16*5*nx), 5*ny, 16, 1.1)
        self.plot_pvd = True

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
        self.set_up_suspended(self.default_mesh)  # TODO: Update
        self.set_up_bedload(self.default_mesh)  # TODO: Update

        # Stabilisation
        self.stabilisation = 'lax_friedrichs'

        # Time integration
        self.dt = 0.3
        self.num_hours = 15
        self.end_time = self.num_hours*3600.0/self.morfac
        self.dt_per_export = 60
        self.dt_per_remesh = 60
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

    def set_source_tracer(self, fs, solver_obj=None):  # TODO: Hook up
        depo = project(self.settling_velocity * self.coeff, fs)
        ero = project(self.settling_velocity * self.ceq, fs)
        return depo, ero

    def get_cfactor(self, depth):
        ksp = Constant(3*self.average_size)
        hclip = conditional(ksp > self.depth, ksp, depth)
        return conditional(depth > self.ksp, 2*((2.5*ln(11.036*hclip/self.ksp))**(-2)), 0)

    def set_quadratic_drag_coefficient(self, fs):
        if self.friction == 'nikuradse':
            return project(self.get_cfactor(), fs)

    def set_bathymetry(self, fs, **kwargs):

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
        boundary_conditions = {
            'shallow_water': {
                inflow_tag: {'flux': Constant(-0.22)},
                outflow_tag: {'elev': Constant(0.397)},
            },
            'tracer': {
                inflow_tag: {'value': self.tracer_init_value},  # TODO: set_initial_condition_tracer
            },
        }
        return boundary_conditions

    def set_initial_condition(self, prob):
        """
        Set initial elevation and velocity using asymptotic solution.

        :arg fs: `FunctionSpace` in which the initial condition should live.
        """
        eta_init, uv_init = self.initialise_fields(self.input_dir, self.di)
        u, eta = prob.fwd_solutions[0].split()
        u.project(uv_init)
        eta.project(eta_init)

    def get_update_forcings(self, solver_obj):

        def update_forcings(t):
            """
            if round(t, 2)%18.0 == 0:
                if self.t_old.dat.data[:] == t:
                    bath_file = File(self.di + '/bath_timestep.pvd')
                    bath_file.write(solver_obj.fields.bathymetry_2d)
                    #import ipdb; ipdb.set_trace()
            """
            self.tracer_list.append(min(solver_obj.fields.tracer_2d.dat.data[:]))

            self.update_key_hydro(solver_obj)

            if self.t_old.dat.data[:] == t:
                self.update_suspended(solver_obj)
                self.update_bedload(solver_obj)

                solve(self.f == 0, self.z_n1)

                self.bathymetry.assign(self.z_n1)
                solver_obj.fields.bathymetry_2d.assign(self.z_n1)

            self.t_old.assign(t)

        return update_forcings

    def initialise_fields(self, inputdir, outputdir):
        """
        Initialise simulation with results from a previous simulation
        """
        from firedrake.petsc import PETSc

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
            chk = DumbCheckpoint(inputdir + "/velocity", mode=FILE_READ)
            uv_init = Function(vector_dg, name="velocity")
            chk.load(uv_init)
            File(outputdir + "/velocity_imported.pvd").write(uv_init)
            chk.close()
        return elev_init, uv_init,

    def get_export_func(self, prob, i):

        def export_func():
            self.bath_file.write(prob.bathymetry[i])

        return export_func
