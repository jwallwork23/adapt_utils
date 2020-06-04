from thetis import *
from thetis.configuration import *

import math
import os

from adapt_utils.tracer.options import TracerOptions


__all__ = ["LeVequeOptions"]


# NOTE: Could set three different tracers in Thetis implementation
class LeVequeOptions(TracerOptions):
    r"""
    Parameters for test case in [LeVeque 1996]. The analytical final time solution is the initial
    condition, since there is no diffusivity.

    We consider a quantity of interest (QoI) :math:`J` of the form

..  math:: J(\phi) = \int_A \phi(T) \;\mathrm{d}x,

    where :math:`A` is a circular region surrounding the slotted cylinder. That is, we seek to
    accurately resolve the slotted cylinder at the final time.

    The discontinuity of both source and QoI motivates utilising DG methods.

    The QoI considered in this test case may be viewed as an extension of the QoI considered in the
    [Power et al. 2006] and TELEMAC-2D test cases to time-dependent problems.
    """
    def __init__(self, shape=0, mesh_type='circle', refinement_level=0, background_concentration=0.0, **kwargs):
        n = refinement_level

        # Temporal discretisation
        self.dt = pi/300.0
        self.end_time = 2*pi + self.dt
        self.dt_per_export = 10
        self.dt_per_remesh = 10

        super(LeVequeOptions, self).__init__(**kwargs)
        self.bg = background_concentration

        # Domain
        if mesh_type == 'circle':
            mesh_file = os.path.join(os.path.dirname(__file__), 'circle.msh')
            if os.path.exists(mesh_file):
                self.default_mesh = Mesh(mesh_file)
            else:
                raise IOError("Mesh file {:s} does not exist.".format(mesh_file))
            if n > 0:
                self.default_mesh = MeshHierarchy(self.default_mesh, n)[-1]
        elif mesh_type == 'square':
            self.default_mesh = UnitSquareMesh(40*2**n, 40*2**n)
        else:
            raise ValueError("Mesh type {:s} not recognised.".format(mesh_type))

        # Source / receiver
        self.source_loc = [(0.25, 0.5, 0.15),     # Bell     (shape 0)
                           (0.5, 0.25, 0.15),     # Cone     (shape 1)
                           (0.5, 0.75, 0.15),     # Cylinder \ (shape 2)
                           (0.475, 0.525, 0.85)]  # Slot     /
        self.set_region_of_interest(shape=shape)

        # Physics
        self.base_diffusivity = 0.

        # Spatial discretisation
        if self.family in ('CG', 'cg', 'Lagrange'):
            self.stabilisation = 'SUPG'
        elif self.family in ('DG', 'dg', 'Discontinuous Lagrange'):
            self.stabilisation = 'no'
        else:
            raise NotImplementedError

        # Solver
        self.params = self.iterative_params
        # self.params["ksp_monitor"] = None
        # self.params["ksp_converged_reason"] =  None

    def set_region_of_interest(self, shape=0):
        assert shape in (0, 1, 2)
        self.shape = shape
        self.region_of_interest = [(self.source_loc[shape][0], self.source_loc[shape][1], 0.175)]

    def set_boundary_conditions(self, fs):
        zero = Constant(self.bg, domain=fs.mesh())
        self.boundary_conditions = {'tracer': {}}
        self.adjoint_boundary_conditions = {'tracer': {}}
        for i in range(1, 5):
            self.boundary_conditions['tracer'][i] = {i: {'value': zero}}
            self.adjoint_boundary_conditions['tracer'][i] = {i: {'diff_flux': zero}}
        return self.boundary_conditions

    def set_velocity(self, fs):
        x, y = SpatialCoordinate(fs.mesh())
        self.fluid_velocity = interpolate(as_vector((0.5 - y, x - 0.5)), fs)
        return self.fluid_velocity

    def set_initial_condition(self, fs):
        x, y = SpatialCoordinate(fs.mesh())
        bell_x0, bell_y0, bell_r0 = self.source_loc[0]
        cone_x0, cone_y0, cone_r0 = self.source_loc[1]
        cyl_x0, cyl_y0, cyl_r0 = self.source_loc[2]
        slot_left, slot_right, slot_top = self.source_loc[3]

        bell = 0.25*(1+cos(pi*min_value(sqrt(pow(x-bell_x0, 2) + pow(y-bell_y0, 2))/bell_r0, 1.0)))
        cone = 1.0 - min_value(sqrt(pow(x-cone_x0, 2) + pow(y-cone_y0, 2))/cone_r0, 1.0)
        slot_cyl = conditional(
            sqrt(pow(x-cyl_x0, 2) + pow(y-cyl_y0, 2)) < cyl_r0, conditional(
                And(And(x > slot_left, x < slot_right), y < slot_top), 0.0, 1.0), 0.0)

        self.initial_value = interpolate(self.bg + bell + cone + slot_cyl, fs)
        return self.initial_value

    def set_qoi_kernel(self, fs):
        b = self.ball(fs.mesh(), source=False)
        area = assemble(b*dx)
        area_exact = pi*self.region_of_interest[0][2]**2
        rescaling = area_exact/area if area != 0. else 1
        self.kernel = rescaling*b
        return self.kernel

    def set_final_condition(self, fs):
        b = self.ball(fs.mesh(), source=False)
        area = assemble(b*dx)
        area_exact = pi*self.region_of_interest[0][2]**2
        rescaling = area_exact/area if area != 0. else 1
        self.final_value = interpolate(rescaling*b, fs)
        return self.final_value

    def exact_solution(self, fs):
        if not hasattr(self, 'initial_value'):
            self.set_initial_condition(fs)
        return self.initial_value

    def exact_qoi(self):
        h = 1  # Height of cone and cylinder

        # Gaussian
        if self.shape == 0:
            r = self.source_loc[0][2]
            return (pi/4 - 1/pi)*r*r

        # Cone
        elif self.shape == 1:
            r = self.source_loc[1][2]
            return h*pi*r*r/3

        # Slotted cylinder
        elif self.shape == 2:
            l = self.source_loc[3][1] - self.source_loc[2][0]  # Width of slot to left
            t = self.source_loc[3][2] - self.source_loc[2][1]  # Height of slot to top
            r = self.source_loc[2][2]                          # Cylinder radius
            return h*(pi*r*r - 2*t*l - r*r*math.asin(l/r) - l*sqrt(r*r - l*l))

    def quadrature_qoi(self, fs):
        x, y = SpatialCoordinate(fs.mesh())
        bell_x, bell_y, bell_r = self.source_loc[0]
        cone_x, cone_y, cone_r = self.source_loc[1]
        cyl_x, cyl_y, cyl_r = self.source_loc[2]
        slot_left, slot_right, slot_top = self.source_loc[3]

        bell = 0.25*(1.0 + cos(pi*min_value(sqrt(pow(x - bell_x, 2) + pow(y - bell_y, 2))/bell_r, 1.0)))
        cone = 1.0 - min_value(sqrt(pow(x - cone_x, 2) + pow(y - cone_y, 2))/cyl_r, 1.0)
        slot_cyl = conditional(sqrt(pow(x - cyl_x, 2) + pow(y - cyl_y, 2)) < cyl_r,
                               conditional(And(And(x > slot_left, x < slot_right), y < slot_top),
                                           0.0, 1.0), 0.0)

        sol = self.bg + bell + cone + slot_cyl
        self.set_qoi_kernel(fs)
        return assemble(self.kernel*sol*dx(degree=12))
