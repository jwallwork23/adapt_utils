r"""
Test annotation of the relaxation method used for mesh movement under an analytically defined
monitor function from [Weller et al. 2016].

[Weller et al. 2016] H. Weller, P. Browne, C. Budd, and M. Cullen, Mesh adaptation on the
    sphere using optimal transport and the numerical solution of a Monge-Amp\ère type
    equation, J. Comput. Phys., 308 (2016), pp. 102--123,
    https://doi.org/10.1016/j.jcp.2015.12.018.
"""
from firedrake import *
from firedrake_adjoint import *
from adapt_utils.adapt.r import MeshMover
import pytest


def qoi1(x):
    """
    Some nonlinear functional.
    """
    return inner(x, x)*dx


def qoi2(x):
    """
    Error of coordinates against some
    other coordinates, `x0`, with the
    same boundary values.
    """
    x0 = Function(x)
    x0.interpolate(as_vector([
        x[0]*(1-x[0]),
        x[1]*(1-x[1]),
    ]))
    return inner(x-x0, x-x0)*dx


# ---------------------------
# standard tests for pytest
# ---------------------------

@pytest.fixture(params=['ring', 'bell'])
def monitor(request):
    return request.param


@pytest.fixture(params=['relaxation'])
def nonlinear_method(request):
    return request.param


@pytest.fixture(params=['alpha', 'beta'])
def ctrl(request):
    return request.param


# @pytest.fixture(params=['dirichlet', 'freeslip', 'equationbc'])  # FIXME
@pytest.fixture(params=['dirichlet', 'freeslip'])
def bnd(request):
    return request.param


@pytest.fixture(params=[qoi1, qoi2])
def qoi(request):
    return request.param


def test_adjoint(monitor, nonlinear_method, ctrl, bnd, qoi):
    """
    Test replay and gradient for control `ctrl`, boundary condition `bnd` and quantity of
    interest `qoi`.
    """
    rtol = 1.0e-01
    maxiter = 100
    alpha = Constant(10.0)
    beta = Constant(200.0)
    gamma = Constant(0.15)
    if ctrl == 'alpha':
        init = 10.0
        control = Control(alpha)
    elif ctrl == 'beta':
        init = 200.0
        control = Control(beta)

    def ring(x=None, **kwargs):
        """
        An analytically defined monitor function which concentrates mesh density in
        a narrow ring within the unit square domain.
        """
        r = dot(x, x)
        return Constant(1.0) + alpha*pow(cosh(beta*(r - gamma)), -2)

    def bell(mesh=None, x=None):
        """
        An analytically defined monitor function which concentrates mesh density in
        a bell region within the unit square domain.
        """
        r = dot(x, x)
        return 1.0 + alpha*pow(cosh(0.5*beta*r), -2)

    # Setup mesh
    with stop_annotating():
        mesh = UnitSquareMesh(20, 20)
        coords = mesh.coordinates.copy(deepcopy=True)
        coords.interpolate(coords - as_vector([0.5, 0.5]))
    mesh = Mesh(coords)

    # Setup boundary conditions
    if bnd == 'dirichlet':
        P1_vec = VectorFunctionSpace(mesh, "CG", 1)
        bc = DirichletBC(P1_vec, 0, 'on_boundary')
        bbc = []
    elif bnd == 'freeslip':
        bc, bbc = [], []
    else:
        bc, bbc = None, None

    # Move mesh
    mm = MeshMover(
        mesh, ring if monitor == 'ring' else bell,
        bc=bc, bbc=bbc, nonlinear_method=nonlinear_method,
    )
    mm.adapt(rtol=rtol, maxiter=maxiter)

    # Evaluate some functional
    J = assemble(qoi(mm.x))
    stop_annotating()

    # Test replay given same input
    Jhat = ReducedFunctional(J, control)
    c = Constant(init)
    JJ = Jhat(c)
    assert np.isclose(J, JJ), f"{J} != {JJ}"

    # Test replay given different input
    c = Constant(1.1*init)
    JJ = Jhat(c)
    JJJ = Jhat(c)
    assert np.isclose(JJ, JJJ), f"{JJ} != {JJJ}"

    # Taylor test
    c = Constant(init)
    dc = Constant(1.0)
    minconv = taylor_test(Jhat, c, dc)
    assert minconv > 1.90, "Taylor test failed"
