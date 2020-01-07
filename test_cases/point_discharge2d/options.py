from firedrake import *
from thetis.configuration import *
from scipy.special import kn

from adapt_utils.tracer.options import TracerOptions
from adapt_utils.misc.misc import *


__all__ = ["TelemacOptions"]


class TelemacOptions(TracerOptions):
    r"""
    Parameters for the 'Point source with diffusion' test case from TELEMAC-2D validation document
    version 7.0.

    We consider a quantity of interest (QoI) :math:`J` of the form

..  math:: J(\phi) = \int_A \phi \;\mathrm{d}x,

    where :math:`A` is a circular 'receiver' region.

    :kwarg approach: Mesh adaptation strategy.
    :kwarg offset: Shift in x-direction for source location.
    :kwarg centred: Toggle whether receiver is positioned in the centre of the flow or not.
    """
    def __init__(self, approach='fixed_mesh', offset=0., centred=False):
        super(TelemacOptions, self).__init__(approach)
        self.default_mesh = RectangleMesh(100, 20, 50, 10)
        self.offset = offset

        # Source / receiver
        # NOTE: It isn't obvious how to represent a delta function on a finite element mesh. The
        #       idea here is to use a disc with a very small radius. In the context of desalination
        #       outfall, this makes sense, because the source is from a pipe. However, in the context
        #       of analytical solutions, it is not quite right. As such, we have calibrated the
        #       radius so that solving on a sequence of increasingly refined uniform meshes leads to
        #       convergence of the uniform mesh solution to the analytical solution.
        calibrated_r = 0.07980 if centred else 0.07972
        self.source_loc = [(1.+self.offset, 5., calibrated_r)]
        self.region_of_interest = [(20., 5., 0.5)] if centred else [(20., 7.5, 0.5)]
        self.source_value = 100.
        self.source_discharge = 0.1
        self.base_diffusivity = 0.1

        # Boundary conditions
        self.boundary_conditions[1] = {'value': Constant(0.0)}
        self.boundary_conditions[2] = {}
        self.boundary_conditions[3] = {'diff_flux': Constant(0.0)}
        self.boundary_conditions[4] = {'diff_flux': Constant(0.0)}

    def set_diffusivity(self, fs):
        self.diffusivity = Constant(self.base_diffusivity)
        return self.diffusivity

    def set_velocity(self, fs):
        self.fluid_velocity = Function(fs)
        self.fluid_velocity.interpolate(as_vector((1., 0.)))
        return self.fluid_velocity

    def set_source(self, fs):
        x0, y0, r0 = self.source_loc[0]
        self.source = Function(fs)
        nrm=assemble(self.ball(fs, source=True)*dx)
        scaling = pi*r0*r0/nrm if nrm != 0 else 1
        scaling *= 0.5*self.source_value
        self.source.interpolate(self.ball(fs, source=True, scale=scaling))
        return self.source

    def set_qoi_kernel(self, fs):
        b = self.ball(fs, source=False)
        area = assemble(b*dx)
        area_exact = pi*self.region_of_interest[0][2]**2
        rescaling = area_exact/area if area != 0. else 1
        self.kernel = rescaling*b
        return self.kernel

    def exact_solution(self, fs):
        self.solution = Function(fs)
        mesh = fs.mesh()
        x, y = SpatialCoordinate(mesh)
        x0, y0, r = self.source_loc[0]
        u = self.set_velocity(VectorFunctionSpace(fs.mesh(), fs.ufl_element()))
        nu = self.set_diffusivity(fs)
        #q = 0.01  # sediment discharge of source (kg/s)
        q = 1
        r = max_value(sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0)), r)  # (Bessel fn explodes at (x0, y0))
        self.solution.interpolate(0.5*q/(pi*nu)*exp(0.5*u[0]*(x-x0)/nu)*bessk0(0.5*u[0]*r/nu))
        self.solution.rename('Analytic tracer concentration')
        outfile = File(self.di + 'analytic.pvd')
        outfile.write(self.solution)  # NOTE: use 40 discretisation levels in ParaView
        return self.solution

    def exact_qoi(self, fs1, fs2):
        mesh = fs1.mesh()
        x, y = SpatialCoordinate(mesh)
        x0, y0, r = self.source_loc[0]
        u = self.set_velocity(VectorFunctionSpace(fs1.mesh(), fs1.ufl_element()))
        nu = self.set_diffusivity(fs1)
        #q = 0.01  # sediment discharge of source (kg/s)
        q = 1
        r = max_value(sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0)), r)  # (Bessel fn explodes at (x0, y0))
        sol = 0.5*q/(pi*nu)*exp(0.5*u[0]*(x-x0)/nu)*bessk0(0.5*u[0]*r/nu)
        self.set_qoi_kernel(fs2)
        return assemble(self.kernel*sol*dx(degree=12))
