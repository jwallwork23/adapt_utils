from firedrake import *
from thetis.configuration import *
import math
from scipy.special import kn

from adapt_utils.options import Options
from adapt_utils.misc.misc import *


__all__ = ["TracerOptions", "PowerOptions", "TelemacOptions", "Telemac3dOptions", "LeVequeOptions"]


class TracerOptions(Options):
    """
    Default parameter class for `TracerProblem`s.

    This class specifies parameters relating to:
      * the tracer transport PDE and associated initial and boundary conditions;
      * the initial spatial discretisation used;
      * underlying linear solver and preconditioning;
      * mesh adaptation;
      * the quantity of interest;
      * the time integration scheme used (in the unsteady case).

    For some problems, particular class instances additionally define analytical solutions.
    """

    # Domain
    nx = PositiveInteger(4, help="Mesh resolution in x- and y-directions.").tag(config=True)

    # Solver
    params = PETScSolverParameters({'pc_type': 'lu',
                                    'mat_type': 'aij' ,
                                    'pc_factor_mat_solver_type': 'mumps',
                                    'ksp_monitor': None,
                                    'ksp_converged_reason': None}).tag(config=True)
    # TODO: For problems bigger than ~1e6 dofs in 2d, we want to use a scalable iterative solver

    # Physical 
    source_loc = List(default_value=None, allow_none=True, help="Location of source term (if any).").tag(config=True)
    source = FiredrakeScalarExpression(None, allow_none=True, help="Scalar source term for tracer problem.").tag(config=True)
    diffusivity = FiredrakeScalarExpression(Constant(1e-1), help="(Scalar) diffusivity field for tracer problem.").tag(config=True)
    fluid_velocity = FiredrakeVectorExpression(None, allow_none=True, help="Vector fluid velocity field for tracer problem.").tag(config=True)

    def __init__(self, approach='fixed_mesh', dt=0.1):
        super(TracerOptions, self).__init__(approach)
        self.dt = dt
        self.start_time = 0.
        self.end_time = 60. - 0.5*self.dt
        self.dt_per_export = 10
        self.dt_per_remesh = 20
        self.stabilisation = 'SUPG'

    def set_diffusivity(self, fs):
        pass

    def set_velocity(self, fs):
        pass

    def set_source(self, fs):
        pass

    def set_initial_condition(self, fs):
        pass

    def set_kernel(self, fs):
        pass

    def exact_solution(self, fs):
        pass

    def exact_qoi(self):
        return assemble(inner(self.kernel, self.solution)*dx)

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
        self.boundary_conditions[1] = 'dirichlet_zero'
        #self.boundary_conditions[2] = 'neumann_zero'
        self.boundary_conditions[2] = 'none'
        self.boundary_conditions[3] = 'neumann_zero'
        self.boundary_conditions[4] = 'neumann_zero'

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

        # Boundary conditions  # TODO: make Thetis-conforming
        self.boundary_conditions[1] = 'dirichlet_zero'
        self.boundary_conditions[2] = 'none'
        self.boundary_conditions[3] = 'neumann_zero'
        self.boundary_conditions[4] = 'neumann_zero'

        # Time integration
        self.dt = 0.1
        self.end_time = 60.
        self.dt_per_export = 10
        self.dt_per_remesh = 20

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

    def set_initial_condition(self, fs):
        self.initial_value = Function(fs)
        return self.initial_value

    def set_qoi_kernel(self, fs):
        b = self.ball(fs, source=False)
        area = assemble(b*dx)
        area_exact = math.pi*self.region_of_interest[0][2]**2
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


class Telemac3dOptions(TracerOptions):
    """
    Parameters for a 3D extension of the 'Point source with diffusion' test case from TELEMAC-2D
    validation document version 7.0.

    We consider a quantity of interest (QoI) :math:`J` of the form

..  math:: J(\phi) = \int_A \phi \;\mathrm{d}x,

    where :math:`A` is a spherical 'receiver' region.

    :kwarg approach: Mesh adaptation strategy,
    :kwarg offset: Shift in x-direction for source location.
    :kwarg centred: Toggle whether receiver is positioned in the centre of the flow or not.
    """
    def __init__(self, approach='fixed_mesh', offset=0., centred=False):
        super(Telemac3dOptions, self).__init__(approach)
        self.default_mesh = BoxMesh(100, 20, 20, 50, 10, 10)
        self.offset = offset

        # Source / receiver
        calibrated_r = 0.07980 if centred else 0.07972  # TODO: calibrate for 3d case
        self.source_loc = [(1.+self.offset, 5., 5., calibrated_r)]
        self.region_of_interest = [(20., 5., 5., 0.5)] if centred else [(20., 7.5, 7.5, 0.5)]
        self.source_value = 100.
        self.source_discharge = 0.1
        self.base_diffusivity = 0.1

        # Boundary conditions  # TODO: make Thetis-conforming
        self.boundary_conditions[1] = 'dirichlet_zero'
        self.boundary_conditions[2] = 'none'
        self.boundary_conditions[3] = 'neumann_zero'
        self.boundary_conditions[4] = 'neumann_zero'
        self.boundary_conditions[5] = 'neumann_zero'
        self.boundary_conditions[6] = 'neumann_zero'

        # Time integration
        self.dt = 0.1
        self.end_time = 60.
        self.dt_per_export = 10
        self.dt_per_remesh = 20

    def set_diffusivity(self, fs):
        self.diffusivity = Constant(self.base_diffusivity)
        return self.diffusivity

    def set_velocity(self, fs):
        self.fluid_velocity = Function(fs)
        self.fluid_velocity.interpolate(as_vector((1., 0., 0.)))
        return self.fluid_velocity

    def set_source(self, fs):
        x0, y0, z0, r0 = self.source_loc[0]
        self.source = Function(fs)
        nrm=assemble(self.ball(fs, source=True)*dx)
        scaling = pi*r0*r0/nrm if nrm != 0 else 1
        scaling *= 0.5*self.source_value
        self.source.interpolate(self.ball(fs, source=True, scale=scaling))
        return self.source

    def set_initial_condition(self, fs):
        self.initial_value = Function(fs)
        return self.initial_value

    def set_qoi_kernel(self, fs):
        b = self.ball(fs, source=False)
        area = assemble(b*dx)
        area_exact = math.pi*self.region_of_interest[0][2]**2
        rescaling = area_exact/area if area != 0. else 1
        self.kernel = rescaling*b
        return self.kernel

    def exact_solution(self, fs):
        self.solution = Function(fs)
        mesh = fs.mesh()
        x, y, z = SpatialCoordinate(mesh)
        x0, y0, z0, r = self.source_loc[0]
        u = self.set_velocity(VectorFunctionSpace(fs.mesh(), fs.ufl_element()))
        nu = self.set_diffusivity(fs)
        #q = 0.01  # sediment discharge of source (kg/s)
        q = 1
        r = max_value(sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0)), r)  # (Bessel fn explodes at (x0, y0, z0))
        self.solution.interpolate(0.5*q/(pi*nu)*exp(0.5*u[0]*(x-x0)/nu)*bessk0(0.5*u[0]*r/nu))
        self.solution.rename('Analytic tracer concentration')
        outfile = File(self.di + 'analytic.pvd')
        outfile.write(self.solution)
        return self.solution

    def exact_qoi(self, fs1, fs2):
        mesh = fs1.mesh()
        x, y, z = SpatialCoordinate(mesh)
        x0, y0, z0, r = self.source_loc[0]
        u = self.set_velocity(VectorFunctionSpace(fs1.mesh(), fs1.ufl_element()))
        nu = self.set_diffusivity(fs1)
        #q = 0.01  # sediment discharge of source (kg/s)
        q = 1
        r = max_value(sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0)), r)  # (Bessel fn explodes at (x0, y0, z0))
        sol = 0.5*q/(pi*nu)*exp(0.5*u[0]*(x-x0)/nu)*bessk0(0.5*u[0]*r/nu)
        self.set_qoi_kernel(fs2)
        return assemble(self.kernel*sol*dx(degree=12))


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
    def __init__(self, approach='fixed_mesh', shape=0):
        super(LeVequeOptions, self).__init__(approach)
        self.default_mesh = UnitSquareMesh(40, 40)

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

        # Boundary conditions
        q_in = Constant(1.0)
        for i in range(4):
            self.boundary_conditions[i] = {i: {'value': q_in}}

        # Time integration
        self.dt = math.pi/300.0
        self.end_time = 2*math.pi + self.dt
        self.dt_per_export = 10
        self.dt_per_remesh = 10

    def set_diffusivity(self, fs):
        self.diffusivity = Constant(self.base_diffusivity)
        return self.diffusivity

    def set_velocity(self, fs):
        x, y = SpatialCoordinate(fs.mesh())
        self.fluid_velocity = Function(fs)
        self.fluid_velocity.interpolate(as_vector((0.5 - y, x - 0.5)))
        return self.fluid_velocity

    def set_initial_condition(self, fs):
        x, y = SpatialCoordinate(fs.mesh())
        bell_x0, bell_y0, bell_r0 = self.source_loc[0]
        cone_x0, cone_y0, cone_r0 = self.source_loc[1]
        cyl_x0, cyl_y0, cyl_r0 = self.source_loc[2]
        slot_left, slot_right, slot_top = self.source_loc[3]
        bell = 0.25*(1+cos(math.pi*min_value(sqrt(pow(x-bell_x0, 2) + pow(y-bell_y0, 2))/bell_r0, 1.0)))
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
        area_exact = math.pi*self.region_of_interest[0][2]**2
        rescaling = area_exact/area if area != 0. else 1
        self.kernel = rescaling*b
        return self.kernel

    def exact_solution(self, fs):
        if not hasattr(self, 'initial_value'):
            self.set_initial_condition(fs)
        return self.initial_value

    def exact_qoi(self):
        h = 1
        # Gaussian
        if self.shape == 0:
            r = self.source_loc[0][2]
            return (math.pi/4-1/math.pi)*r*r
        # Cone
        elif self.shape == 1:
            r = self.source_loc[1][2]
            return h*math.pi*r*r/3
        # Slotted cylinder
        else:
            l = self.source_loc[3][1] - self.source_loc[2][0]  # width of slot to left
            t = self.source_loc[3][2] - self.source_loc[2][1]  # height of slot to top
            r = self.source_loc[2][2]                          # cylinder radius
            return h*(math.pi*r*r - 2*t*l - r*r*math.asin(l/r) - r*r*math.sin(2*math.asin(l/r))/2)

    def quadrature_qoi(self, fs):
        x, y = SpatialCoordinate(fs.mesh())
        bell_x0, bell_y0, bell_r0 = self.source_loc[0]
        cone_x0, cone_y0, cone_r0 = self.source_loc[1]
        cyl_x0, cyl_y0, cyl_r0 = self.source_loc[2]
        slot_left, slot_right, slot_top = self.source_loc[3]

        bell = 0.25*(1+cos(math.pi*min_value(sqrt(pow(x-bell_x0, 2) + pow(y-bell_y0, 2))/bell_r0, 1.0)))
        cone = 1.0 - min_value(sqrt(pow(x-cone_x0, 2) + pow(y-cone_y0, 2))/cyl_r0, 1.0)
        slot_cyl = conditional(sqrt(pow(x-cyl_x0, 2) + pow(y-cyl_y0, 2)) < cyl_r0,
                     conditional(And(And(x > slot_left, x < slot_right), y < slot_top),
                       0.0, 1.0), 0.0)

        sol = bell + cone + slot_cyl
        #sol = 1.0 + bell + cone + slot_cyl
        self.set_qoi_kernel(fs)
        return assemble(self.kernel*sol*dx(degree=12))
