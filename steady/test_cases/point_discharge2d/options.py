from firedrake import *
from thetis.configuration import *

import numpy as np
import os

from adapt_utils.steady.tracer.options import bessk0
from adapt_utils.unsteady.options import CoupledOptions


__all__ = ["TelemacOptions"]


class PointDischarge2dOptions(CoupledOptions):
    r"""
    Parameters for the 'Point source with diffusion' test case from TELEMAC-2D validation document
    version 7.0.

    We consider a quantity of interest (QoI) :math:`J` of the form

  ..math:: J(\phi) = \int_A \phi \;\mathrm{d}x,

    where :math:`A` is a circular 'receiver' region.

    :kwarg approach: Mesh adaptation strategy.
    :kwarg offset: Shift in x-direction for source location.
    :kwarg centred: Toggle whether receiver is positioned in the centre of the flow or not.
    """
    def __init__(self, level=0, offset=1.0, centred=False, **kwargs):
        super(PointDischarge2dOptions, self).__init__(**kwargs)
        self.solve_swe = False
        self.solve_tracer = True
        self.timestepper = 'SteadyState'
        self.end_time = 18.0
        self.dt = 20.0
        self.dt_per_export = 1

        # Domain
        self.default_mesh = RectangleMesh(100*2**level, 20*2**level, 50, 10)
        self.offset = offset

        # FEM
        self.degree_tracer = 1
        # self.family = 'cg'
        # self.stabilisation = 'SUPG'
        self.tracer_family = 'dg'
        self.stabilisation = 'lax_friedrichs'
        self.use_automatic_sipg_parameter = True
        self.use_limiter_for_tracers = False
        self.lax_friedrichs_tracer_scaling_factor = Constant(1.0)

        # Physics
        self.base_velocity = [1.0, 0.0]
        self.base_diffusivity = 0.1

        # Source / receiver
        # NOTE: It isn't obvious how to represent a delta function on a finite element mesh. The
        #       idea here is to use a disc with a very small radius. In the context of desalination
        #       outfall, this makes sense, because the source is from a pipe. However, in the context
        #       of analytical solutions, it is not quite right. As such, we have calibrated the
        #       radius so that solving on a sequence of increasingly refined uniform meshes leads to
        #       convergence of the uniform mesh solution to the analytical solution.
        # calibrated_r = 0.06245
        calibrated_r = 0.07980 if centred else 0.07972
        self.source_loc = [(1.0 + self.offset, 5.0, calibrated_r)]
        self.region_of_interest = [(20.0, 5.0, 0.5)] if centred else [(20.0, 7.5, 0.5)]
        self.source_value = 100.0
        self.source_discharge = 0.1

        # Metric normalisation
        self.normalisation = 'error'
        self.norm_order = 1

        # Goal-oriented error estimation
        self.degree_increase_tracer = 1

        # Mesh adaptation
        self.max_adapt = 35
        self.element_rtol = 0.002

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

    def set_tracer_source(self, fs):
        x0, y0, r0 = self.source_loc[0]
        nrm = assemble(self.ball(fs.mesh(), source=True)*dx)
        scaling = 1.0 if np.allclose(nrm, 0.0) else pi*r0*r0/nrm
        scaling *= 0.5*self.source_value
        # scaling *= self.source_value
        return self.ball(fs.mesh(), source=True, scale=scaling)

    def set_qoi_kernel_tracer(self, prob, i):
        return self.set_qoi_kernel(prob.meshes[i])

    def set_qoi_kernel(self, mesh):
        b = self.ball(mesh, source=False)
        area = assemble(b*dx)
        area_exact = pi*self.region_of_interest[0][2]**2
        rescaling = 1.0 if np.allclose(area, 0.0) else area_exact/area
        return rescaling*b

    def exact_solution(self, fs):
        solution = Function(fs)
        mesh = fs.mesh()
        x, y = SpatialCoordinate(mesh)
        x0, y0, r = self.source_loc[0]
        u = Constant(as_vector(self.base_velocity))
        nu = Constant(self.base_diffusivity)
        # q = 0.01  # sediment discharge of source (kg/s)
        q = 1
        r = max_value(sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0)), r)  # (Bessel fn explodes at (x0, y0))
        solution.interpolate(0.5*q/(pi*nu)*exp(0.5*u[0]*(x-x0)/nu)*bessk0(0.5*u[0]*r/nu))
        solution.rename('Analytic tracer concentration')
        outfile = File(os.path.join(self.di, 'analytic.pvd'))
        outfile.write(solution)  # NOTE: use 40 discretisation levels in ParaView
        return solution

    def exact_qoi(self, fs):
        mesh = fs.mesh()
        x, y = SpatialCoordinate(mesh)
        x0, y0, r = self.source_loc[0]
        u = Constant(as_vector(self.base_velocity))
        nu = Constant(self.base_diffusivity)
        # q = 0.01  # sediment discharge of source (kg/s)
        q = 1
        r = max_value(sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0)), r)  # (Bessel fn explodes at (x0, y0))
        sol = 0.5*q/(pi*nu)*exp(0.5*u[0]*(x-x0)/nu)*bessk0(0.5*u[0]*r/nu)
        kernel = self.set_qoi_kernel(mesh)
        return assemble(kernel*sol*dx(degree=12))
