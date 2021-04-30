from thetis import *

import argparse
import os
from time import perf_counter

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
parser.add_argument('-time', help="Toggle timing mode.")
parser.add_argument('-debug', help="Toggle debugging mode.")
args = parser.parse_args()


def myprint(msg, flag):
    if flag:
        print(msg)


# --- Set parameters

family = args.family or 'cg'
assert family in ('cg', 'dg')
offset = bool(args.offset or False)
time = bool(args.time or False)
taylor = bool(args.taylor_test or False) and not time
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

# TODO: Limiters?


# --- Solve forward

tp = AdaptiveSteadyProblem(op, print_progress=False)
n = FacetNormal(tp.mesh)
D = tp.fields[0].horizontal_diffusivity
h = D.copy(deepcopy=True)
h.assign(0.1)

timestamp = perf_counter()
tp.solve_forward()
myprint("{:.4f}".format(perf_counter() - timestamp), time)


J = tp.quantity_of_interest()


def reduced_functional(m):
    """
    Evaluate the reduced functional for diffusivity `m`.
    """
    op.base_diffusivity = m.dat.data[0]
    tp.__init__(op, print_progress=False)
    tp.solve_forward()
    return tp.quantity_of_interest()


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
if taylor:
    Jhat = reduced_functional
    dJdm = gradient_continuous(D)
    print("Continuous adjoint gradient = {:.4e}".format(dJdm))
    minconv = taylor_test(Jhat, D, h, dJdm=dJdm)
    assert minconv >= 1.94

timestamp = perf_counter()
tp.solve_adjoint()
myprint("{:.4f}".format(perf_counter() - timestamp), time)
adj = tp.adj_solution_tracer
op.plot_pvd = False
export_field(adj, "Adjoint tracer", "continuous_adjoint", fpath=op.di, plexname=None, op=op)
