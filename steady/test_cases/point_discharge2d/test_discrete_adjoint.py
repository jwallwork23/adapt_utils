from thetis import *
import pyadjoint

import argparse
import os

from adapt_utils.steady.solver_adjoint import AdaptiveDiscreteAdjointSteadyProblem
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


# --- Taylor test

tp = AdaptiveDiscreteAdjointSteadyProblem(op, print_progress=False)
tp.solve_forward()


def reduced_functional(m):
    with pyadjoint.stop_annotating():
        _op = op.copy()
        _op.base_diffusivity = m
        _tp = tp.__class__(_op, print_progress=False)
        _tp.control_field.assign(m)
        _tp.solve_forward()
        return _tp.quantity_of_interest()


def gradient():
    tp.solve_forward()
    c = tp.fwd_solution_tracer
    tp.solve_adjoint()
    c_star = tp.adj_solution_tracer
    return assemble(-tp.step*inner(grad(c_star), grad(c))*dx)


tp.step.assign(0.1)
minconv = pyadjoint.taylor_test(reduced_functional, tp.initial_control, tp.step, dJdm=gradient())
assert minconv > 1.95
