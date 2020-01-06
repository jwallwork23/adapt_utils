from thetis import *
from thetis.configuration import *

from adapt_utils.tracer.options import TracerOptions


__all__ = ["PowerOptions"]


class PowerOptions(TracerOptions):
    """
    Parameters for test case in [Power et al. 2006].

    We consider a quantity of interest (QoI) :math:`J` of the form

..  math:: J(\phi) = \int_A \phi \;\mathrm{d}x,

    where :math:`A` is a square 'receiver' region.

    :kwarg centred: Toggle whether receiver is positioned in the centre of the flow or not.
    """
    def __init__(self, approach='fixed_mesh', centred=True):
        super(PowerOptions, self).__init__(approach)
        self.default_mesh = SquareMesh(40, 40, 4, 4)

        # Source / receiver
        self.source_loc = [(1., 2., 0.1)] if centred else [(1., 1.5, 0.1)]
        self.region_of_interest = [(3., 2., 0.1)] if centred else [(3., 2.5, 0.1)]
        self.base_diffusivity = 1.

        # Boundary conditions  # TODO: make Thetis-conforming
        self.boundary_conditions[1] = {'value': Constant(0.0)}
        #self.boundary_conditions[2] = {'diff_flux': Constant(0.0)}
        self.boundary_conditions[2] = {}
        self.boundary_conditions[3] = {'diff_flux': Constant(0.0)}
        self.boundary_conditions[4] = {'diff_flux': Constant(0.0)}

    def set_diffusivity(self, fs):
        self.diffusivity = Constant(self.base_diffusivity)
        return self.diffusivity

    def set_velocity(self, fs):
        self.fluid_velocity = Function(fs)
        self.fluid_velocity.interpolate(as_vector((15., 0.)))
        return self.fluid_velocity

    def set_source(self, fs):
        self.source = Function(fs)
        #self.source.interpolate(self.bump(fs, source=True))
        self.source.interpolate(self.box(fs, source=True))
        area = assemble(self.source*dx)
        rescaling = 0.04/area if area != 0. else 1.
        self.source.interpolate(rescaling*self.source)
        self.source.rename("Source term")
        return self.source

    def set_qoi_kernel(self, fs):  # FIXME: update
        self.kernel = Function(fs)
        #self.kernel.interpolate(self.bump(fs))
        self.kernel.interpolate(self.box(fs))
        area = assemble(self.kernel*dx)
        rescaling = 0.04/area if area != 0. else 1.
        self.kernel.interpolate(rescaling*self.kernel)
        return self.kernel
