from firedrake import *
from thetis.configuration import *
# from scipy.special import kn

from adapt_utils.options import Options
from adapt_utils.misc import *


__all__ = ["TracerOptions", "bessk0"]


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
    n = PositiveInteger(4, help="Mesh resolution in x- and y-directions.").tag(config=True)

    # Solver
    params = PETScSolverParameters({
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'mat_type': 'aij',
        'pc_factor_mat_solver_type': 'mumps',
    }).tag(config=True)

    # For problems bigger than ~1e6 dofs in 2d, we want to use a scalable iterative solver
    iterative_params = PETScSolverParameters({
        'ksp_type': 'gmres',
        'pc_type': 'sor',
    }).tag(config=True)

    # Physical
    source_loc = List(default_value=None, allow_none=True, help="Location of source term (if any).").tag(config=True)
    source = FiredrakeScalarExpression(None, allow_none=True, help="Scalar source term for tracer problem.").tag(config=True)
    diffusivity = FiredrakeScalarExpression(Constant(1e-1), help="(Scalar) diffusivity field for tracer problem.").tag(config=True)
    fluid_velocity = FiredrakeVectorExpression(None, allow_none=True, help="Vector fluid velocity field for tracer problem.").tag(config=True)

    def __init__(self, **kwargs):
        super(TracerOptions, self).__init__(**kwargs)

    def set_diffusivity(self, fs):
        """Should be implemented in derived class."""
        self.diffusivity = Constant(self.base_diffusivity)
        return self.diffusivity

    def set_velocity(self, fs):
        """Should be implemented in derived class."""
        self.fluid_velocity = Constant(self.base_velocity)
        return self.fluid_velocity

    def set_source(self, fs):
        """Should be implemented in derived class."""
        self.source = None
        return self.source


def bessi0(x):
    """Modified Bessel function of the first kind. Code taken from 'Numerical recipes in C'."""
    ax = abs(x)
    y1 = x/3.75
    y1 *= y1
    expr1 = 1.0 + y1*(3.5156229 + y1*(3.0899424 + y1*(1.2067492 + y1*(0.2659732 + y1*(0.360768e-1 + y1*0.45813e-2)))))
    y2 = 3.75/ax
    expr2 = (exp(ax)/sqrt(ax))*(0.39894228 + y2*(0.1328592e-1 + y2*(0.225319e-2 + y2*(-0.157565e-2 + y2*(0.916281e-2 + y2*(-0.2057706e-1 + y2*(0.2635537e-1 + y2*(-0.1647633e-1 + y2*0.392377e-2))))))))
    return conditional(le(ax, 3.75), expr1, expr2)


def bessk0(x):
    """Modified Bessel function of the second kind. Code taken from 'Numerical recipes in C'."""
    y1 = x*x/4.0
    expr1 = (-ln(x/2.0)*bessi0(x)) + (-0.57721566 + y1*(0.42278420 + y1*(0.23069756 + y1*(0.3488590e-1 + y1*(0.262698e-2 + y1*(0.10750e-3 + y1*0.74e-5))))))
    y2 = 2.0/x
    expr2 = (exp(-x)/sqrt(x))*(1.25331414 + y2*(-0.7832358e-1 + y2*(0.2189568e-1 + y2*(-0.1062446e-1 + y2*(0.587872e-2 + y2*(-0.251540e-2 + y2*0.53208e-3))))))
    return conditional(ge(x, 2), expr2, expr1)
