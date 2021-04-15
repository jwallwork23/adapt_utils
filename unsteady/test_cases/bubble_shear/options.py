from thetis import *
<<<<<<< HEAD
from thetis.configuration import *

from adapt_utils.unsteady.options import CoupledOptions
=======

from adapt_utils.options import CoupledOptions
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a


__all__ = ["BubbleOptions"]


# TODO: 3d version
class BubbleOptions(CoupledOptions):
<<<<<<< HEAD
    # TODO: doc
=======
    r"""
    Parameters for a 2D version of the pure advection test case in [Barral et al. 2016].

    An initial circular bubble is advected in a non-uniform velocity field. The velocity
    field reverses halfway through the time period, meaning that the final solution should
    match the initial condition.
    """
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
    def __init__(self, n=1, **kwargs):
        super(BubbleOptions, self).__init__(**kwargs)
        self.solve_swe = False
        self.solve_tracer = True
<<<<<<< HEAD
        self.default_mesh = UnitSquareMesh(40*2**n, 40*2**n)
        if self.tracer_family == 'cg':
            self.stabilisation = 'supg'
        elif self.tracer_family == 'dg':
            self.stabilisation = 'lax_friedrichs'
=======
        self.adapt_field = 'tracer'
        self.default_mesh = UnitSquareMesh(40*2**n, 40*2**n)
        if self.tracer_family == 'cg':
            self.stabilisation_tracer = 'supg'
        elif self.tracer_family == 'dg':
            self.stabilisation_tracer = 'lax_friedrichs'
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
        else:
            raise NotImplementedError

        # Source / receiver
        self.source_loc = [(0.5, 0.85, 0.1)]
        self.base_diffusivity = 0.0
<<<<<<< HEAD

        # Time integration
        self.period = 6.0
        self.dt = 0.015
        # self.end_time = self.period/4
        self.end_time = self.period/2
        # self.dt_per_export = 10
        self.dt_per_export = 1
=======
        self.characteristic_speed = Constant(2.0)  # TODO: check

        # Time integration
        self.period = 6.0
        self.dt = 0.005*0.5**n
        self.end_time = self.period/2
        self.dt_per_export = 1  # Required for Lagrangian mesh movement
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a

    def set_boundary_conditions(self, prob, i):
        boundary_conditions = {
            'tracer': {
<<<<<<< HEAD
                1: {'value': Constant(0.0)},
                2: {'value': Constant(0.0)},
                3: {'value': Constant(0.0)},
                4: {'value': Constant(0.0)},
=======
                'on_boundary': {'value': Constant(0.0)}
                # 1: {'value': Constant(0.0)},
                # 2: {'value': Constant(0.0)},
                # 3: {'value': Constant(0.0)},
                # 4: {'value': Constant(0.0)},
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
            },
        }
        return boundary_conditions

    def get_velocity(self, coords, t):
        x, y = coords
        return as_vector([
            2*sin(pi*x)*sin(pi*x)*sin(2*pi*y)*cos(2*pi*t/self.period),
            -2*sin(2*pi*x)*sin(pi*y)*sin(pi*y)*cos(2*pi*t/self.period),
        ])

    def update_velocity(self, prob, i, t):
        u, eta = prob.fwd_solutions[i].split()
        u.interpolate(self.get_velocity(prob.meshes[i].coordinates, t))

<<<<<<< HEAD
    def set_initial_condition(self, prob):
        self.update_velocity(prob, 0, 0.0)

    def set_initial_condition_tracer(self, prob):
        prob.fwd_solutions_tracer[0].interpolate(self.ball(prob.meshes[0], source=True))
=======
    def set_initial_condition(self, prob, i=0):
        self.update_velocity(prob, i, 0.0)

    def set_initial_condition_tracer(self, prob, i=0):
        prob.fwd_solutions_tracer[i].interpolate(self.ball(prob.meshes[i], source=True))
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a

    def get_update_forcings(self, prob, i, adjoint=False):

        def update_forcings(t):
            self.update_velocity(prob, i, t)

        return update_forcings
