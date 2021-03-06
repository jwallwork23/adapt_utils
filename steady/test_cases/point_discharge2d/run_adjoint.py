from thetis import *
from firedrake_adjoint import *
from firedrake.adjoint.blocks import GenericSolveBlock, ProjectBlock

import argparse
import os

from adapt_utils.io import export_field
from adapt_utils.steady.solver import AdaptiveSteadyProblem
from adapt_utils.steady.test_cases.point_discharge2d.options import PointDischarge2dOptions


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument('-level', help="Number of uniform refinements to apply to the initial mesh.")
parser.add_argument('-family', help="Finite element family")
parser.add_argument('-stabilisation', help="Stabilisation method to use")
parser.add_argument('-anisotropic_stabilisation', help="Use anisotropic cell size measure?")
parser.add_argument('-offset', help="Toggle between aligned or offset region of interest.")
parser.add_argument('-taylor_test', help="Run a Taylor test.")
parser.add_argument('-debug', help="Toggle debugging mode.")
args = parser.parse_args()


# --- Set parameters

family = args.family or 'cg'
assert family in ('cg', 'dg')
offset = bool(args.offset or False)
kwargs = {
    'level': int(args.level or 0),
    'aligned': not offset,
    'plot_pvd': True,
    'debug': bool(args.debug or False),
}
op = PointDischarge2dOptions(approach='fixed_mesh', **kwargs)
op.tracer_family = family
stabilisation = args.stabilisation or 'supg'
op.stabilisation_tracer = None if stabilisation == 'none' else stabilisation
op.anisotropic_stabilisation = False if args.anisotropic_stabilisation == '0' else True
alignment = 'offset' if offset else 'aligned'
op.di = create_directory(os.path.join(op.di, op.stabilisation_tracer or family, alignment))
op.use_automatic_sipg_parameter = op.tracer_family == 'dg'

# TODO: Limiters?


# --- Solve forward

tp = AdaptiveSteadyProblem(op, print_progress=False)
n = FacetNormal(tp.mesh)
tp.solve_forward()
J = tp.quantity_of_interest()

D = tp.fields[0].horizontal_diffusivity
m = Control(D)
h = D.copy(deepcopy=True)
h.assign(0.1)
dJdm = compute_gradient(J, m)  # in R space
print("Discrete adjoint gradient = {:.4e}".format(dJdm.dat.data[0]))
stop_annotating()


# --- Solve discrete adjoint

tape = get_working_tape()
solve_blocks = [block for block in tape.get_blocks() if isinstance(block, GenericSolveBlock)]
solve_blocks = [block for block in solve_blocks if not isinstance(block, ProjectBlock)]
adj = solve_blocks[-1].adj_sol.copy(deepcopy=True)
adj *= -1  # FIXME: Why do we need this?
export_field(adj, "Adjoint tracer", "discrete_adjoint", fpath=op.di, plexname=None, op=op)
solutions = [adj]


def reduced_functional(m):
    """
    Evaluate the reduced functional for diffusivity `m`.
    """
    op.base_diffusivity = m.dat.data[0]
    tp.__init__(op, print_progress=False)
    tp.solve_forward()
    return tp.quantity_of_interest()


def gradient_discrete(m):
    """
    Evaluate the gradient of the reduced functional using the discrete adjoint solution.
    """
    c_star = adj
    # op.base_diffusivity = m.dat.data[0]
    # tp.__init__(op, print_progress=False)
    # tp.solve_forward()
    # tp._solve_discrete_adjoint()
    c = tp.fwd_solution_tracer
    # c_star = tp.adj_solution_tracer
    return assemble(-h*inner(grad(c_star), grad(c))*dx)


# Taylor test discrete adjoint
if bool(args.taylor_test or False):
    Jhat = reduced_functional
    dJdm = gradient_discrete(m)
    # dJdm = dJdm.dat.data[0]*h.dat.data[0]
    print("Discrete adjoint gradient = {:.4e}".format(dJdm))
    # Jhat = ReducedFunctional(J, m)
    # dJdm = None
    minconv = taylor_test(Jhat, D, h, dJdm=dJdm)
    assert minconv >= 1.94


# --- Solve continuous adjoint


def gradient_continuous(m):
    """
    Evaluate the gradient of the reduced functional for diffusivity `m` using the continuous
    adjoint method.
    """
    op.base_diffusivity = m.dat.data[0]
    tp.__init__(op, print_progress=False)
    tp.solve_forward()
    tp.solve_adjoint()
    c = tp.fwd_solution_tracer
    c_star = tp.adj_solution_tracer
    return assemble(-h*inner(grad(c_star), grad(c))*dx)


# Taylor test continuous adjoint
if bool(args.taylor_test or False):
    Jhat = reduced_functional
    dJdm = gradient_continuous(D)
    print("Continuous adjoint gradient = {:.4e}".format(dJdm))
    minconv = taylor_test(Jhat, D, h, dJdm=dJdm)
    assert minconv >= 1.94

tp.solve_adjoint()
adj = tp.adj_solution_tracer
op.plot_pvd = False
export_field(adj, "Adjoint tracer", "continuous_adjoint", fpath=op.di, plexname=None, op=op)
solutions.append(adj)


# --- Compute L2 error against discrete adjoint

print_output("L2 'error': {:.4f}%".format(100*errornorm(*solutions)/norm(solutions[0])))
