from thetis import *
from thetis.configuration import *

import numpy as np

<<<<<<< HEAD
from adapt_utils.params import *
from adapt_utils.options import CoupledOptions


__all__ = ["BoxDischarge2dOptions"]


class BoxDischarge2dOptions(CoupledOptions):
=======
from adapt_utils.steady.tracer.options import TracerOptions


__all__ = ["PowerOptions"]


class PowerOptions(TracerOptions):
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
    r"""
    Parameters for test case in [Power et al. 2006].

    We consider a quantity of interest (QoI) :math:`J` of the form

..  math:: J(\phi) = \int_A \phi \;\mathrm{d}x,

    where :math:`A` is a square 'receiver' region.

    :kwarg centred: Toggle whether receiver is positioned in the centre of the flow or not.
    """
<<<<<<< HEAD
    def __init__(self, centred=True, direct=True, **kwargs):
        self.timestepper = 'SteadyState'
        super(BoxDischarge2dOptions, self).__init__(**kwargs)
=======
    def __init__(self, centred=True, **kwargs):
        super(PowerOptions, self).__init__(**kwargs)
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
        self.solve_swe = False
        self.solve_tracer = True
        self.default_mesh = SquareMesh(40, 40, 4, 4)

        # Source / receiver
        self.source_loc = [(1., 2., 0.1)] if centred else [(1., 1.5, 0.1)]
        self.region_of_interest = [(3., 2., 0.1)] if centred else [(3., 2.5, 0.1)]
        self.base_diffusivity = 1.0
        self.base_velocity = [15.0, 0.0]
<<<<<<< HEAD
        self.characteristic_speed = Constant(15.0)
        self.characteristic_diffusion = Constant(1.0)

        # Solver parameters
        self.solver_parameters['tracer'] = direct_tracer_params if direct else iterative_tracer_params
        self.adjoint_solver_parameters['tracer'] = self.solver_parameters['tracer']
=======
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a

    def set_boundary_conditions(self, prob, i):
        zero = Constant(0.0)
        boundary_conditions = {
            1: {'value': zero},
            # 2: {'diff_flux': zero},
            2: {},
            3: {'diff_flux': zero},
            4: {'diff_flux': zero},
        }
        return boundary_conditions

<<<<<<< HEAD
    def get_velocity(self, t):
        return as_vector(self.base_velocity)

=======
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
    def set_initial_condition(self, prob):
        u, eta = prob.fwd_solutions[0].split()
        u.interpolate(as_vector(self.base_velocity))

    def set_source(self, fs):  # TODO
        source = self.bump(fs.mesh(), source=True)
        area = assemble(source*dx)
<<<<<<< HEAD
        rescaling = 1.0 if np.isclose(area, 0.0) else 0.04/area
=======
        rescaling = 1.0 if np.allclose(area, 0.0) else 0.04/area
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
        return interpolate(rescaling*source)

    def set_qoi_kernel(self, fs):  # FIXME: update
        kernel = self.bump(fs.mesh())
        area = assemble(kernel*dx)
<<<<<<< HEAD
        rescaling = 1.0 if np.isclose(area, 0.0) else 0.04/area
=======
        rescaling = 1.0 if np.allclose(area, 0.0) else 0.04/area
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
        return interpolate(rescaling*kernel, fs)
