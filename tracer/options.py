from firedrake import *
from thetis.configuration import *
from scipy.special import kn

from adapt_utils.options import Options
from adapt_utils.misc.misc import *


__all__ = ["TracerOptions"]


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
    params = PETScSolverParameters({'ksp_type': 'preonly',
                                    'pc_type': 'lu',
                                    'mat_type': 'aij' ,
                                    'pc_factor_mat_solver_type': 'mumps',
                                    }).tag(config=True)
    # TODO: For problems bigger than ~1e6 dofs in 2d, we want to use a scalable iterative solver

    # Physical 
    source_loc = List(default_value=None, allow_none=True, help="Location of source term (if any).").tag(config=True)
    source = FiredrakeScalarExpression(None, allow_none=True, help="Scalar source term for tracer problem.").tag(config=True)
    diffusivity = FiredrakeScalarExpression(Constant(1e-1), help="(Scalar) diffusivity field for tracer problem.").tag(config=True)
    fluid_velocity = FiredrakeVectorExpression(None, allow_none=True, help="Vector fluid velocity field for tracer problem.").tag(config=True)

    def __init__(self, dt=0.1, **kwargs):
        super(TracerOptions, self).__init__(**kwargs)
        self.dt = dt
        self.start_time = 0.
        self.end_time = 60. - 0.5*self.dt
        self.dt_per_export = 10
        self.dt_per_remesh = 20
        self.stabilisation = 'SUPG'

    def set_diffusivity(self, fs):
        raise NotImplementedError

    def set_velocity(self, fs):
        raise NotImplementedError
