from firedrake import *

import numpy as np
import os

from adapt_utils.maths import bessk0
from adapt_utils.options import CoupledOptions


__all__ = ["PointDischarge2dOptions"]


class PointDischarge2dOptions(CoupledOptions):
    r"""
    Parameters for the 'Point source with diffusion' test case from TELEMAC-2D validation document
    version 7.0.

    We consider a quantity of interest (QoI) :math:`J` of the form

  ..math:: J(\phi) = \int_A \phi \;\mathrm{d}x,

    where :math:`A` is a circular 'receiver' region.

    :kwarg approach: Mesh adaptation strategy.
    :kwarg shift: Shift in x-direction for source location.
    :kwarg aligned: Toggle whether receiver is positioned in the centre of the flow or not.
    """
    def __init__(self, level=0, shift=1.0, aligned=True, **kwargs):
        super(PointDischarge2dOptions, self).__init__(**kwargs)
        self.solve_swe = False
        self.solve_tracer = True

        # Steady state
        self.timestepper = 'SteadyState'
        self.start_time = 0.0
        self.end_time = 20.0
        self.dt = 20.0
        self.dt_per_export = 1

        # Domain
        n = 2**level
        self.default_mesh = RectangleMesh(100*n, 20*n, 50, 10)
        self.shift = shift

        # FEM
        self.degree_tracer = 1
        self.tracer_family = 'cg'
        self.stabilisation_tracer = 'SUPG'
        self.use_automatic_sipg_parameter = True
        self.use_limiter_for_tracers = False

        # Hydrodynamics
        self.base_velocity = [1.0, 0.0]
        self.base_diffusivity = 0.1
        self.characteristic_speed = Constant(1.0)
        self.characteristic_diffusion = Constant(0.1)

        # Source / receiver
        self.source_value = 100.0
        self.region_of_interest = [(20.0, 5.0, 0.5)] if aligned else [(20.0, 7.5, 0.5)]

        # Goal-oriented error estimation
        self.degree_increase_tracer = 0
        self.adapt_field = 'tracer'

        # Mesh adaptation
        rtol = 0.005
        self.element_rtol = rtol
        self.estimator_rtol = rtol
        self.qoi_rtol = rtol
        self.h_min = 1.0e-10
        self.h_max = 1.0e+02

        # Solver parameters
        self.solver_parameters['tracer'] = {
            'mat_type': 'aij',
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
        }
        # self.solver_parameters['tracer'] = {
        #     'ksp_type': 'gmres',
        #     'pc_type': 'sor',
        # }

    def set_boundary_conditions(self, prob, i):
        zero = Constant(0.0)
        boundary_conditions = {
            'tracer': {
                1: {'value': zero},
                2: {},
                3: {'diff_flux': zero},
                4: {'diff_flux': zero},
            }
        }
        return boundary_conditions

    def set_calibrated_radius(self):
        """
        It isn't obvious how to represent a delta function in a finite element model. The approach
        here is to approximate it using a Gaussian with a narrow radius. This radius is calibrated
        using gradient-based optimisation for the square L2 error against the analytical solution,
        using the `calibrate_radius.py` script on a fine mesh.

        Note that the calibrated radii are computed on mesh level 4 in the hierarchy.
        """
        stabilisation = self.stabilisation_tracer
        calibration_results = {
            'cg': {
                None: 0.05606309,
                'su': 0.05606563,
                'supg': 0.05606535,
                'su_anisotropic': 0.05606395,
                'supg_anisotropic': 0.05606388,
            },
            'dg': {
                None: 0.05606298,
                'lax_friedrichs': 0.05606298,
                'lax_friedrichs_anisotropic': 0.05606303,
            },
        }
        if self.anisotropic_stabilisation:
            stabilisation += '_anisotropic'
        stabilisation = None if stabilisation is None else stabilisation.lower()
        calibrated_r = calibration_results[self.tracer_family][stabilisation]
        return [(1.0 + self.shift, 5.0, calibrated_r)]

    @property
    def source_loc(self):
        return self.set_calibrated_radius()

    def set_tracer_source(self, fs):
        return self.gaussian(fs.mesh(), source=True, scale=self.source_value)

    def get_velocity(self, t):
        return as_vector(self.base_velocity)

    def set_qoi_kernel_tracer(self, prob, i):
        return self.set_qoi_kernel(prob.meshes[i])

    def set_qoi_kernel(self, mesh):
        b = self.ball(mesh, source=False)
        area = assemble(b*dx)
        area_analytical = pi*self.region_of_interest[0][2]**2
        rescaling = 1.0 if np.isclose(area, 0.0) else area_analytical/area
        return rescaling*b

    def analytical_solution(self, fs):
        solution = Function(fs, name="Analytical tracer concentration")
        mesh = fs.mesh()
        x, y = SpatialCoordinate(mesh)
        x0, y0, r = self.source_loc[0]
        u = Constant(as_vector(self.base_velocity))
        D = Constant(self.base_diffusivity)
        Pe = 0.5*u[0]/D
        # q = 0.01  # sediment discharge of source (kg/s)
        q = 1
        rr = sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0))
        rr = max_value(rr, r)  # Bessel function explodes at (x0, y0)
        solution.interpolate(0.5*q/(pi*D)*exp(Pe*(x-x0))*bessk0(Pe*rr))
        if self.plot_pvd:
            outfile = File(os.path.join(self.di, 'analytical.pvd'))
            outfile.write(solution)  # NOTE: use 40 discretisation levels in ParaView
        return solution

    def analytical_qoi(self, mesh=None):
        mesh = mesh or self.default_mesh
        x, y = SpatialCoordinate(mesh)
        x0, y0, r = self.source_loc[0]
        u = Constant(as_vector(self.base_velocity))
        D = Constant(self.base_diffusivity)
        Pe = 0.5*u[0]/D
        # q = 0.01  # sediment discharge of source (kg/s)
        q = 1
        rr = sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0))
        rr = max_value(rr, r)  # (Bessel fn explodes at (x0, y0))
        sol = 0.5*q/(pi*D)*exp(Pe*(x-x0))*bessk0(Pe*rr)
        kernel = self.set_qoi_kernel(mesh)
        return assemble(kernel*sol*dx(degree=self.qoi_quadrature_degree))
