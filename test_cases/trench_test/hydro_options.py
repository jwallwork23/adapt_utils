from thetis import *
from thetis.configuration import *

from adapt_utils.swe.morphological_options import MorphOptions

import os
from matplotlib import rc

rc('text', usetex=True)


__all__ = ["TrenchHydroOptions"]


class TrenchHydroOptions(MorphOptions):
    """
    Parameters for test case described in [1].

    [1] Clare, Mariana, et al. “Hydro-morphodynamics 2D Modelling Using a Discontinuous Galerkin Discretisation.”
    EarthArXiv, 9 Jan. 2020. Web.
    """

    def __init__(self, friction='manning', plot_timeseries=False, nx=1, ny=1, **kwargs):
        super(TrenchHydroOptions, self).__init__(**kwargs)
        self.plot_timeseries = plot_timeseries

        self.default_mesh = RectangleMesh(16*5*nx, 5*ny, 16, 1.1)
        self.P1DG = FunctionSpace(self.default_mesh, "DG", 1)
        self.V = FunctionSpace(self.default_mesh, "CG", 1)
        self.vector_cg = VectorFunctionSpace(self.default_mesh, "CG", 1)
        self.vector_dg = VectorFunctionSpace(self.default_mesh, "DG", 1)

        self.plot_pvd = True
        self.di = "hydro_output"

        # Physical
        self.base_viscosity = 1e-6

        self.gravity = Constant(9.81)

        self.solve_tracer = False
        self.wetting_and_drying = False

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
        self.dt_per_export = 500
        self.dt_per_remesh = 500
        self.timestepper = 'CrankNicolson'
        self.implicitness_theta = 1.0

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
        return conditional(self.depth > self.ksp, 2*(0.4**2)/(ln(aux)**2), 0.0)

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
        trench = conditional(le(x, 5), depth_riv, conditional(le(x, 6.5), (1/1.5)*depth_diff*(x-6.5) + depth_trench,
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
        boundary_conditions = {}
        boundary_conditions[inflow_tag] = {'flux': Constant(-0.22)}
        boundary_conditions[outflow_tag] = {'elev': Constant(0.397)}
        return boundary_conditions

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
        return None

    def plot(self):
        return None


def export_final_state(inputdir, uv, elev,):
    """
    Export fields to be used in a subsequent simulation
    """
    if not os.path.exists(inputdir):
        os.makedirs(inputdir)
    print_output("Exporting fields for subsequent simulation")
    chk = DumbCheckpoint(inputdir + "/velocity", mode=FILE_CREATE)
    chk.store(uv, name="velocity")
    File(inputdir + '/velocityout.pvd').write(uv)
    chk.close()
    chk = DumbCheckpoint(inputdir + "/elevation", mode=FILE_CREATE)
    chk.store(elev, name="elevation")
    File(inputdir + '/elevationout.pvd').write(elev)
    chk.close()

    plex = elev.function_space().mesh()._plex
    viewer = PETSc.Viewer().createHDF5(inputdir + '/myplex.h5', 'w')
    viewer(plex)
