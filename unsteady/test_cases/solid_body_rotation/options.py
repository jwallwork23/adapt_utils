from thetis import *
from thetis.configuration import *

import math
import os

from adapt_utils.options import CoupledOptions


__all__ = ["LeVequeOptions"]


# NOTE: Could set three different tracers in Thetis implementation
class LeVequeOptions(CoupledOptions):
    r"""
    Parameters for test case in [LeVeque 1996]. The analytical terminal time solution is the initial
    condition, since there is no diffusivity.

    We consider a quantity of interest (QoI) :math:`J` of the form

..  math:: J(\phi) = \int_A \phi(T) \;\mathrm{d}x,

    where :math:`A` is a circular region surrounding the slotted cylinder. That is, we seek to
    accurately resolve the slotted cylinder at the terminal time.

    The discontinuity of both source and QoI motivates utilising DG methods.

    The QoI considered in this test case may be viewed as an extension of the QoI considered in the
    [Power et al. 2006] and TELEMAC-2D test cases to time-dependent problems.
    """
    def __init__(self, shape=0, geometry='circle', level=0, background_concentration=0.0, **kwargs):
        self.solve_swe = False
        self.solve_tracer = True
        self.shape = shape

        # Temporal discretisation
        self.dt = pi/300.0
        self.start_time = self.end_time
        self.end_time = 2*pi
        self.dt_per_export = 10

        super(LeVequeOptions, self).__init__(**kwargs)
        self.bg = background_concentration

        # Domain
        if geometry == 'circle':
            mesh_file = os.path.join(os.path.dirname(__file__), 'circle.msh')
            if os.path.exists(mesh_file):
                self.default_mesh = Mesh(mesh_file)
            else:
                raise IOError("Mesh file {:s} does not exist.".format(mesh_file))
            if level > 0:
                self.default_mesh = MeshHierarchy(self.default_mesh, level)[-1]
        elif geometry == 'square':
            self.default_mesh = UnitSquareMesh(40*2**level, 40*2**level)
        else:
            raise ValueError("Geometry {:s} not recognised.".format(geometry))
        self.default_mesh.coordinates.dat.data[:] -= 0.5

        # Source / receiver
        self.source_loc = [(-0.25, 0, 0.15),       # Bell     (shape 0)
                           (0, -0.25, 0.15),       # Cone     (shape 1)
                           (0, 0.25, 0.15),        # Cylinder \ (shape 2)
                           (-0.025, 0.025, 0.35)]  # Slot     /
        self.set_region_of_interest(shape=shape)

        # Physics
        self.base_diffusivity = 0.0

        # Spatial discretisation
        if self.tracer_family == 'cg':
            self.stabilisation_tracer = 'SUPG'
        elif self.tracer_family == 'dg':
            self.stabilisation_tracer = 'lax_friedrichs'
            self.lax_friedrichs_tracer_scaling_factor = Constant(1.0)

        # Solver
        self.solver_parameters = {
            'tracer': {
                'ksp_type': 'gmres',
                'pc_type': 'sor',
                # 'ksp_converged_reason': None,
            }
        }
        self.adjoint_solver_parameters = {
            'tracer': {
                'ksp_type': 'gmres',
                'pc_type': 'sor',
                # 'ksp_converged_reason': None,
            }
        }

    def set_region_of_interest(self, shape=0):
        assert shape in (0, 1, 2)
        self.shape = shape
        self.region_of_interest = [(self.source_loc[shape][0], self.source_loc[shape][1], 0.175)]

    def set_boundary_conditions(self, prob, i):
        boundary_conditions = {'tracer': {}}
        for i in range(1, 5):
            boundary_conditions['tracer'][i] = {i: {'value': Constant(self.bg)}}
        return boundary_conditions

    def get_velocity(self, coords, t):
        return as_vector([-coords[1], coords[0]])

    def set_initial_condition(self, prob, i=0):
        q = prob.fwd_solutions[i]
        u, eta = q.split()
        u.interpolate(self.get_velocity(prob.meshes[i].coordinates, 0.0))

    def set_initial_condition_tracer(self, prob):
        x, y = SpatialCoordinate(prob.meshes[0])
        bell_x0, bell_y0, bell_r0 = self.source_loc[0]
        cone_x0, cone_y0, cone_r0 = self.source_loc[1]
        cyl_x0, cyl_y0, cyl_r0 = self.source_loc[2]
        slot_left, slot_right, slot_top = self.source_loc[3]

        bell = 0.25*(1+cos(pi*min_value(sqrt(pow(x-bell_x0, 2) + pow(y-bell_y0, 2))/bell_r0, 1.0)))
        cone = 1.0 - min_value(sqrt(pow(x-cone_x0, 2) + pow(y-cone_y0, 2))/cone_r0, 1.0)
        slot_cyl = conditional(
            sqrt(pow(x-cyl_x0, 2) + pow(y-cyl_y0, 2)) < cyl_r0, conditional(
                And(And(x > slot_left, x < slot_right), y < slot_top), 0.0, 1.0), 0.0)

        prob.fwd_solutions_tracer[0].interpolate(self.bg + bell + cone + slot_cyl)

    def set_qoi_kernel_tracer(self, prob, i):
        b = self.ball(prob.meshes[i], source=False)
        area = assemble(b*dx)
        area_exact = pi*self.region_of_interest[0][2]**2
        rescaling = area_exact/area if np.allclose(area, 0.0) else 1
        return rescaling*b

    def set_terminal_condition_tracer(self, prob):
        prob.adj_solutions_tracer[-1].interpolate(self.set_qoi_kernel_tracer(prob, -1))

    def exact_solution(self, fs):
        raise NotImplementedError  # TODO
        # return self.set_initial_condition(fs)

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

    def quadrature_qoi(self, prob, i):
        x, y = SpatialCoordinate(prob.meshes[i])
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
        kernel = self.set_qoi_kernel_tracer(prob, i)
        return assemble(kernel*sol*dx(degree=12))

    def get_update_forcings(self, prob, i, adjoint=False):

        def update_forcings(t):
            self.set_initial_condition(prob, i=i)

        return update_forcings
