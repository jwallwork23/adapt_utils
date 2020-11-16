from thetis import *

import argparse

from adapt_utils.adapt.r import MeshMover
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.test_cases.bubble_shear.options import BubbleOptions


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("-n", help="Resolution of initial mesh.")
parser.add_argument("-conservative", help="Toggle conservative tracer equation")
parser.add_argument("-limiters", help="Toggle limiters for tracer equation")
parser.add_argument("-stabilisation", help="Stabilisation method")
parser.add_argument("-family", help="Choose finite element from 'cg' and 'dg'")
parser.add_argument("-debug", help="Toggle debugging mode.")
args = parser.parse_args()


# --- Set parameters

kwargs = {

    # Spatial discretisation
    'tracer_family': args.family or 'dg',
    'stabilisation_tracer': args.stabilisation or 'lax_friedrichs',
    'use_automatic_sipg_parameter': False,  # We have an inviscid problem
    'use_limiter_for_tracers': bool(args.limiters or True),
    'use_tracer_conservative_form': bool(args.conservative or False),

    # Mesh movement
    'nonlinear_method': 'relaxation',
    'r_adapt_rtol': 5.0e-2,

    # Misc
    'debug': bool(args.debug or False),
}
if os.getenv('REGRESSION_TEST') is not None:
    kwargs['end_time'] = 1.5
op = BubbleOptions(approach='lagrangian', n=int(args.n or 1))
op.update(kwargs)


# --- Initialise the mesh

tp = AdaptiveProblem(op)

# NOTE: We use Monge-Ampere with a monitor function indicating the initial condition

alpha = 10.0   # Parameter controlling significance of refined region
eps = 1.0e-03  # Parameter controlling width of refined region


def monitor(mesh):
    x, y = SpatialCoordinate(mesh)
    x0, y0, r = op.source_loc[0]
    return conditional(le(abs((x-x0)**2 + (y-y0)**2 - r**2), eps), alpha, 1.0)


mesh_mover = MeshMover(tp.meshes[0], monitor, method='monge_ampere', op=op)
mesh_mover.adapt()
tp.__init__(op, meshes=[Mesh(mesh_mover.x), ])


# --- Solve the tracer transport problem

# Note:
#  * Pure Lagrangian leads to tangled elements after only a few iterations
#  * This motivates applying monitor based methods throughout the simulation

tp.set_initial_condition()
init_norm = norm(tp.fwd_solutions_tracer[0])
init_sol = tp.fwd_solutions_tracer[0].copy(deepcopy=True)
tp.solve_forward()

# TODO: Plot inverted elements

# Compare initial and final tracer concentrations
final_norm = norm(tp.fwd_solutions_tracer[0])
final_sol = tp.fwd_solutions_tracer[0].copy(deepcopy=True)
print_output("Initial norm:   {:.4e}".format(init_norm))
print_output("Final norm:     {:.4e}".format(final_norm))
print_output("Relative error: {:.2f}%".format(100*abs(1.0-errornorm(init_sol, final_sol)/init_norm)))
