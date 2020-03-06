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
    def __init__(self, shape=0, n=0, **kwargs):
        super(LeVequeOptions, self).__init__(**kwargs)
        if self.family in ('CG', 'cg', 'Lagrange'):
            self.stabilisation = 'SUPG'
        elif self.family in ('DG', 'dg', 'Discontinuous Lagrange'):
            self.stabilisation = 'no'
        else:
            raise NotImplementedError

        # self.default_mesh = UnitSquareMesh(40*2**n, 40*2**n)
        mesh_file = os.path.join(os.path.dirname(__file__), 'circle.msh')
        if os.path.exists(mesh_file):
            self.default_mesh = Mesh(mesh_file)
        if n > 0:
            mh = MeshHierarchy(self.default_mesh, n)
            self.default_mesh = mh[-1]

        # Source / receiver
        self.source_loc = [(0.25, 0.5, 0.15), (0.5, 0.25, 0.15), (0.5, 0.75, 0.15), (0.475, 0.525, 0.85)]
        assert shape in (0, 1, 2)
        self.shape = shape
        if shape == 0:
            self.region_of_interest = [(0.25, 0.5, 0.175)]
        elif shape == 1:
            self.region_of_interest = [(0.5, 0.25, 0.175)]
        else:
            self.region_of_interest = [(0.5, 0.75, 0.175)]
        self.base_diffusivity = 0.

        # Time integration
        self.dt = pi/300.0
        self.end_time = 2*pi + self.dt
        self.dt_per_export = 10
        # self.dt_per_export = 1
        self.dt_per_remesh = 10

        self.params = {
            "ksp_type": "gmres",
            "pc_type": "sor",
            # "ksp_monitor": None,
            # "ksp_converged_reason": None,
        }

    def set_boundary_conditions(self, fs):
        # zero = Constant(1.0, domain=fs.mesh())
        zero = Constant(0.0, domain=fs.mesh())
        for i in range(1, 5):
            self.boundary_conditions[i] = {i: {'value': zero}}
            self.adjoint_boundary_conditions[i] = {i: {'diff_flux': zero}}
        return self.boundary_conditions

    def set_diffusivity(self, fs):
        self.diffusivity = Constant(self.base_diffusivity)
        return self.diffusivity

    def set_velocity(self, fs):
        x, y = SpatialCoordinate(fs.mesh())
        self.fluid_velocity = interpolate(as_vector((0.5 - y, x - 0.5)), fs)
        return self.fluid_velocity

    def set_source(self, fs):
        self.source = Constant(0.)
        return self.source

    def set_initial_condition(self, fs):
        x, y = SpatialCoordinate(fs.mesh())
        bell_x0, bell_y0, bell_r0 = self.source_loc[0]
        cone_x0, cone_y0, cone_r0 = self.source_loc[1]
        cyl_x0, cyl_y0, cyl_r0 = self.source_loc[2]
        slot_left, slot_right, slot_top = self.source_loc[3]
        bell = 0.25*(1+cos(pi*min_value(sqrt(pow(x-bell_x0, 2) + pow(y-bell_y0, 2))/bell_r0, 1.0)))
        cone = 1.0 - min_value(sqrt(pow(x-cone_x0, 2) + pow(y-cone_y0, 2))/cyl_r0, 1.0)
        slot_cyl = conditional(sqrt(pow(x-cyl_x0, 2) + pow(y-cyl_y0, 2)) < cyl_r0,
                     conditional(And(And(x > slot_left, x < slot_right), y < slot_top),
                       0.0, 1.0), 0.0)

        self.initial_value = Function(fs)
        #self.initial_value.interpolate(1.0 + bell + cone + slot_cyl)
        self.initial_value.interpolate(bell + cone + slot_cyl)
        return self.initial_value

    def set_qoi_kernel(self, fs):
        b = self.ball(fs, source=False)
        area = assemble(b*dx)
        area_exact = pi*self.region_of_interest[0][2]**2
        rescaling = area_exact/area if area != 0. else 1
        self.kernel = rescaling*b
        return self.kernel

    def set_final_condition(self, fs):
        b = self.ball(fs, source=False)
        area = assemble(b*dx)
        area_exact = pi*self.region_of_interest[0][2]**2
        rescaling = area_exact/area if area != 0. else 1
        self.final_value = Function(fs)
        self.final_value.interpolate(rescaling*b)
        return self.final_value

    def exact_solution(self, fs):
        if not hasattr(self, 'initial_value'):
            self.set_initial_condition(fs)
        return self.initial_value

    def exact_qoi(self):
        h = 1
        # Gaussian
        if self.shape == 0:
            r = self.source_loc[0][2]
            return (pi/4-1/pi)*r*r
        # Cone
        elif self.shape == 1:
            r = self.source_loc[1][2]
            return h*pi*r*r/3
        # Slotted cylinder
        else:
            l = self.source_loc[3][1] - self.source_loc[2][0]  # width of slot to left
            t = self.source_loc[3][2] - self.source_loc[2][1]  # height of slot to top
            r = self.source_loc[2][2]                          # cylinder radius
            return h*(pi*r*r - 2*t*l - r*r*math.asin(l/r) - l*sqrt(r*r-l*l))

    def quadrature_qoi(self, fs):
        x, y = SpatialCoordinate(fs.mesh())
        bell_x0, bell_y0, bell_r0 = self.source_loc[0]
        cone_x0, cone_y0, cone_r0 = self.source_loc[1]
        cyl_x0, cyl_y0, cyl_r0 = self.source_loc[2]
        slot_left, slot_right, slot_top = self.source_loc[3]

        bell = 0.25*(1+cos(pi*min_value(sqrt(pow(x-bell_x0, 2) + pow(y-bell_y0, 2))/bell_r0, 1.0)))
        cone = 1.0 - min_value(sqrt(pow(x-cone_x0, 2) + pow(y-cone_y0, 2))/cyl_r0, 1.0)
        slot_cyl = conditional(sqrt(pow(x-cyl_x0, 2) + pow(y-cyl_y0, 2)) < cyl_r0,
                     conditional(And(And(x > slot_left, x < slot_right), y < slot_top),
                       0.0, 1.0), 0.0)

        sol = bell + cone + slot_cyl
        #sol = 1.0 + bell + cone + slot_cyl
        self.set_qoi_kernel(fs)
        return assemble(self.kernel*sol*dx(degree=12))

    def lp_errors(self, sol):
        if not hasattr(self, 'initial_value'):
            self.set_initial_condition(fs)
        exact = self.initial_value.copy()

        L1_err = assemble(abs(sol - exact)*dx)/assemble(abs(exact)*dx)
        L2_err = sqrt(assemble((sol - exact)*(sol - exact)*dx))/sqrt(assemble(exact*exact*dx))
        with exact.dat.vec_ro as v_exact:
            L_inf_exact = v_exact.max()[1]
        exact -= sol
        domain = '{[i]: 0 <= i < diff.dofs}'
        instructions = '''
        for i
            diff[i] = abs(diff[i])
        end
        '''
        par_loop((domain, instructions), dx, {'diff': (exact, RW)}, is_loopy_kernel=True)
        with exact.dat.vec_ro as v_diff:
            L_inf_err = v_diff.max()[1]/L_inf_exact

        return L1_err, L2_err, L_inf_err
