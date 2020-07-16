from firedrake import *
from thetis.configuration import *
# from scipy.special import kn

import numpy as np

from adapt_utils.steady.tracer.options import *


__all__ = ["Telemac3dOptions"]


# TODO: 3D tracer solver
class Telemac3dOptions(TracerOptions):
    r"""
    Parameters for a 3D extension of the 'Point source with diffusion' test case from TELEMAC-2D
    validation document version 7.0.

    We consider a quantity of interest (QoI) :math:`J` of the form

..  math:: J(\phi) = \int_A \phi \;\mathrm{d}x,

    where :math:`A` is a spherical 'receiver' region.

    :kwarg approach: Mesh adaptation strategy,
    :kwarg offset: Shift in x-direction for source location.
    :kwarg centred: Toggle whether receiver is positioned in the centre of the flow or not.
    """
    def __init__(self, offset=0., centred=False, **kwargs):
        super(Telemac3dOptions, self).__init__(**kwargs)
        self.solve_swe = False
        self.solve_tracer = True
        self.default_mesh = BoxMesh(100, 20, 20, 50, 10, 10)
        self.offset = offset
        self.family = 'cg'
        self.stabilisation = 'SUPG'

        # Source / receiver
        calibrated_r = 0.07980 if centred else 0.07972  # TODO: calibrate for 3d case
        self.source_loc = [(1.+self.offset, 5., 5., calibrated_r)]
        self.region_of_interest = [(20., 5., 5., 0.5)] if centred else [(20., 7.5, 7.5, 0.5)]
        self.source_value = 100.
        self.source_discharge = 0.1
        self.base_diffusivity = 0.1
        self.base_velocity = [1.0, 0.0, 0.0]

        # Metric normalisation
        self.normalisation = 'error'
        self.norm_order = 1

    def set_boundary_conditions(self, prob, i):
        zero = Constant(0.0)
        boundary_conditions = {
            1: {'value': zero},
            2: {},
            3: {'diff_flux': zero},
            4: {'diff_flux': zero},
            5: {'diff_flux': zero},
            6: {'diff_flux': zero},
        }
        return boundary_conditions

    def set_source(self, fs):  # TODO
        x0, y0, z0, r0 = self.source_loc[0]
        nrm = assemble(self.ball(fs.mesh(), source=True)*dx)
        scaling = 1.0 if np.allclose(nrm, 0.0) else pi*r0*r0/nrm
        scaling *= 0.5*self.source_value
        return self.ball(fs.mesh(), source=True, scale=scaling)

    def set_qoi_kernel(self, fs):
        b = self.ball(fs.mesh(), source=False)
        area = assemble(b*dx)
        area_exact = pi*self.region_of_interest[0][2]**2
        rescaling = 1.0 if np.allclose(area, 0.0) else area_exact/area
        return rescaling*b

    def exact_solution(self, fs):
        solution = Function(fs)
        mesh = fs.mesh()
        x, y, z = SpatialCoordinate(mesh)
        x0, y0, z0, r = self.source_loc[0]
        u = self.set_velocity(VectorFunctionSpace(fs.mesh(), fs.ufl_element()))
        nu = self.set_diffusivity(fs)
        # q = 0.01  # sediment discharge of source (kg/s)
        q = 1
        r = max_value(sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0)), r)  # (Bessel fn explodes at (x0, y0, z0))
        solution.interpolate(0.5*q/(pi*nu)*exp(0.5*u[0]*(x-x0)/nu)*bessk0(0.5*u[0]*r/nu))
        solution.rename('Analytic tracer concentration')
        outfile = File(self.di + 'analytic.pvd')
        outfile.write(solution)
        return solution

    def exact_qoi(self, fs1, fs2):
        mesh = fs1.mesh()
        x, y, z = SpatialCoordinate(mesh)
        x0, y0, z0, r = self.source_loc[0]
        u = self.set_velocity(VectorFunctionSpace(fs1.mesh(), fs1.ufl_element()))
        nu = self.set_diffusivity(fs1)
        # q = 0.01  # sediment discharge of source (kg/s)
        q = 1
        r = max_value(sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0)), r)  # (Bessel fn explodes at (x0, y0, z0))
        sol = 0.5*q/(pi*nu)*exp(0.5*u[0]*(x-x0)/nu)*bessk0(0.5*u[0]*r/nu)
        kernel = self.set_qoi_kernel(fs2)
        return assemble(kernel*sol*dx(degree=12))
