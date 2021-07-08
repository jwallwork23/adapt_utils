from thetis import *
from thetis.configuration import *
from thetis.options import ModelOptions2d

from adapt_utils.io import initialise_hydrodynamics
from adapt_utils.unsteady.options import CoupledOptions
from adapt_utils.unsteady.sediment.sediments_model import SedimentModel

import numpy as np

__all__ = ["BeachOptions"]


class BeachOptions(CoupledOptions):
    """
    Parameters for test case adapted from [1].

    [1] Karna et al. "A fully implicit wetting–drying method for DG-FEM shallow water models, with an application to the Scheldt Estuary." Computer Methods in Applied Mechanics and Engineering 200.5-8 (2011): 509-524.
    """

    def __init__(self, dt_exp=72, friction='manning', nx=1, ny=1, mesh = None, input_dir = None, output_dir = None, **kwargs):
        super(BeachOptions, self).__init__(**kwargs)

        try:
            assert friction in ('nikuradse', 'manning')
        except AssertionError:
            raise ValueError("Friction parametrisation '{:s}' not recognised.".format(friction))
        self.friction = friction

        self.lx = 220
        self.ly = 10

        if output_dir is not None:
            self.di = output_dir

        self.default_mesh = RectangleMesh(np.int(220*nx), np.int(10*ny), self.lx, self.ly)

        self.friction_coeff = 0.02

        self.set_up_morph_model(self.default_mesh)

        # Initial
        self.elev_init, self.uv_init = self.initialise_fields(input_dir, self.di)

        self.plot_pvd = True
        self.hessian_recovery = 'dL2'

        self.grad_depth_viscosity = True

        self.num_hours = 3600

        # Stabilisation
        self.stabilisation = 'lax_friedrichs'

        self.morphological_acceleration_factor = Constant(10000)

        # Boundary conditions
        h_amp = 0.25  # Ocean boundary forcing amplitude
        v_amp = 0.5  # Ocean boundary foring velocity
        omega = 0.5  # Ocean boundary forcing frequency
        self.ocean_elev_func = lambda t: (h_amp * np.cos(-omega*(t+(100.0))))
        self.ocean_vel_func = lambda t: (v_amp * np.cos(-omega*(t+(100.0))))

        self.tracer_init = Constant(0.0)

        # Time integration

        self.dt = 0.5
        self.end_time = float(self.num_hours*3600.0/self.morphological_acceleration_factor)
        self.dt_per_mesh_movement = dt_exp
        self.dt_per_export = dt_exp
        self.implicitness_theta = 1.0

    def set_up_morph_model(self, mesh=None):

        # Physical
        self.base_viscosity = 0.5
        self.base_diffusivity = 100
        self.gravity = Constant(9.81)
        self.porosity = Constant(0.4)
        self.ks = Constant(0.025)
        self.average_size = 0.0002  # Average sediment size

        self.wetting_and_drying = True
        self.depth_integrated = True
        self.use_tracer_conservative_form = True
        self.slope_eff = True
        self.angle_correction = False
        self.suspended = True
        self.convective_vel_flag = True
        self.bedload = True
        self.solve_sediment = True
        self.solve_exner = True

        self.norm_smoother = Constant(10/25)

        P1 = FunctionSpace(mesh, "CG", 1)
        bath = self.set_bathymetry(P1)
        self.wetting_and_drying_alpha = Constant(1/40)

    def create_sediment_model(self, mesh, bathymetry):
        self.P1DG = FunctionSpace(mesh, "DG", 1)
        self.P1_vec_dg = VectorFunctionSpace(mesh, "DG", 1)

        self.uv_d = Function(self.P1_vec_dg).project(self.uv_init)
        self.eta_d = Function(self.P1DG).project(self.elev_init)

        self.sediment_model = SedimentModel(ModelOptions2d, suspendedload=self.suspended, convectivevel=self.convective_vel_flag,
                                            bedload=self.bedload, angle_correction=self.angle_correction, slope_eff=self.slope_eff, seccurrent=False,
                                            mesh2d=mesh, bathymetry_2d=bathymetry,
                                            uv_init=self.uv_d, elev_init=self.eta_d, ks=self.ks, average_size=self.average_size,
                                            cons_tracer=self.use_tracer_conservative_form, wetting_and_drying=self.wetting_and_drying, wetting_alpha=self.wetting_and_drying_alpha)

    def set_manning_drag_coefficient(self, fs):
        if self.friction == 'manning':
            if hasattr(self, 'friction_coeff'):
                self.manning_drag_coefficient = Constant(self.friction_coeff)
            else:
                self.manning_drag_coefficient = Constant(0.02)
        return self.manning_drag_coefficient

    def set_bathymetry(self, fs, **kwargs):
        x, y = SpatialCoordinate(fs.mesh())
        self.bathymetry = Function(fs, name="Bathymetry")
        self.bathymetry.interpolate(Constant(180/40) - x/40)
        return self.bathymetry

    def set_viscosity(self, fs):
        x, y = SpatialCoordinate(fs.mesh())
        self.viscosity = Function(fs)
        sponge_viscosity = Function(fs).interpolate(conditional(x >= 100, -399 + 4*x, Constant(1.0)))
        self.viscosity.interpolate(sponge_viscosity*self.base_viscosity)
        return self.viscosity

    def set_boundary_conditions(self, prob, i):
        if not hasattr(self, 'elev_in'):
            self.elev_in = Constant(0.0)
        if not hasattr(self, 'vel_in'):
            self.vel_in = Constant(as_vector((0.0, 0.0)))
        self.elev_in.assign(self.ocean_elev_func(0.0))
        vel_const = Constant(self.ocean_vel_func(0.0))
        self.vel_in.assign(as_vector((vel_const, 0.0)))

        inflow_tag = 1
        boundary_conditions = {
            'shallow_water': {
                inflow_tag: {'elev': self.elev_in, 'uv': self.vel_in},
            },
            'sediment': {}
        }
        return boundary_conditions

    def update_boundary_conditions(self, solver_obj, t=0.0):
        self.elev_in.assign(self.ocean_elev_func(t))
        vel_const = Constant(self.ocean_vel_func(t))
        self.vel_in.assign(as_vector((vel_const, 0.0)))

    def set_initial_condition(self, prob):
        u, eta = prob.fwd_solutions[0].split()
        u.project(self.uv_init)
        eta.project(self.elev_init)

    def set_sediment_source(self, fs):
        if self.suspended and not self.depth_integrated:
            return self.sediment_model.ero_term
        else:
            return None

    def set_sediment_sink(self, fs):
        if self.suspended and not self.depth_integrated:
            return self.sediment_model.depo_term
        else:
            return None

    def set_sediment_depth_integ_sink(self, fs):
        if self.suspended and self.depth_integrated:
            return self.sediment_model.depo_term
        else:
            return None

    def set_sediment_depth_integ_source(self, fs):
        if self.suspended and self.depth_integrated:
            return self.sediment_model.ero
        else:
            return None

    def set_advective_velocity_factor(self, fs):
        if self.convective_vel_flag:
            return self.sediment_model.corr_factor_model.corr_vel_factor
        else:
            return Constant(1.0)

    def set_initial_condition_sediment(self, prob):
        prob.fwd_solutions_sediment[0].interpolate(Constant(0.0))

    def set_initial_condition_bathymetry(self, prob):
        prob.fwd_solutions_bathymetry[0].interpolate(self.set_bathymetry(prob.fwd_solutions_bathymetry[0].function_space()))

    def get_update_forcings(self, prob, i, adjoint):

        def update_forcings(t):
            uv, elev = prob.fwd_solutions[0].split()
            self.update_boundary_conditions(prob, t=t)

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
            #File(outputdir + "/elevation_imported.pvd").write(elev_init)
            chk.close()
        # velocity
        with timed_stage('initialising velocity'):
            chk = DumbCheckpoint(inputdir + "/velocity" , mode=FILE_READ)
            uv_init = Function(vector_dg, name="velocity")
            chk.load(uv_init)
            #File(outputdir + "/velocity_imported.pvd").write(uv_init)
            chk.close()

        return  elev_init, uv_init,

    def get_export_func(self, prob, i):
        return None

    def set_boundary_surface(self):
        """Set the initial displacement of the boundary elevation."""
        pass
