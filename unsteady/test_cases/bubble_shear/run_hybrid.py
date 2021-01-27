from thetis import *

import argparse
import numpy as np

from adapt_utils.adapt.r import MeshMover
# from adapt_utils.adapt.metric import *
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.test_cases.bubble_shear.options import BubbleOptions


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("-n", help="Resolution of initial mesh.")
parser.add_argument("-conservative", help="Toggle conservative tracer equation")
parser.add_argument("-limiters", help="Toggle limiters for tracer equation")
parser.add_argument("-stabilisation", help="Stabilisation method")
parser.add_argument("-family", help="Choose finite element from 'cg' and 'dg'")
parser.add_argument("-mode", help="Choose from 'h' and 'm'")
parser.add_argument("-debug", help="Toggle debugging mode.")
args, unknown = parser.parse_known_args()


# --- Set parameters

kwargs = {

    # Spatial discretisation
    'tracer_family': args.family or 'dg',
    'stabilisation_tracer': args.stabilisation or 'lax_friedrichs',
    'use_automatic_sipg_parameter': False,  # We have an inviscid problem
    'use_limiter_for_tracers': False if args.limiters == "0" else True,
    # 'use_limiter_for_tracers': bool(args.limiters or False),
    'use_tracer_conservative_form': bool(args.conservative or False),  # FIXME?

    # Mesh movement
    'nonlinear_method': 'relaxation',
    'r_adapt_rtol': 5.0e-2,
    'scaled_jacobian_tol': 0.01,
    'target': 2000,
    'normalisation': 'complexity',
    'norm_order': 1,
    'max_adapt': 4,  # TODO
    'hybrid_mode': args.mode or 'm',

    # Misc
    'debug': bool(args.debug or False),
}
op = BubbleOptions(approach='hybrid', n=int(args.n or 1))
op.update(kwargs)
op.end_time += op.dt  # FIXME


# --- Initialise the mesh

tp = AdaptiveProblem(op)

# NOTE: We use Monge-Ampere with a monitor function indicating the initial condition

alpha = 10.0   # Parameter controlling prominance of refined region
eps = 1.0e-03  # Parameter controlling width of refined region


def monitor(mesh):
    x, y = SpatialCoordinate(mesh)
    x0, y0, r = op.source_loc[0]
    return conditional(le(abs((x-x0)**2 + (y-y0)**2 - r**2), eps), alpha, 1.0)


# Get a suitable initial mesh
mesh_mover = MeshMover(tp.meshes[0], monitor, method='monge_ampere', op=op)
mesh_mover.adapt()
tp.__init__(op, meshes=[Mesh(mesh_mover.x)])
# for i in range(4):
#     tp.set_initial_condition()
#     M = tp.get_static_hessian_metric('tracer')
#     space_normalise(M, op=op)
#     enforce_element_constraints(M, op=op)
#     tp.__init__(op, meshes=[adapt(tp.meshes[0], M)])


# --- Solve the tracer transport problem

# Note:
#  * Pure Lagrangian leads to tangled elements after only a few iterations
#  * This motivates applying monitor based methods throughout the simulation

tp.set_initial_condition()
init_vol = assemble(Constant(1.0)*dx(domain=tp.mesh))
init_l1_norm = norm(tp.fwd_solutions_tracer[0], norm_type='L1')
init_l2_norm = norm(tp.fwd_solutions_tracer[0], norm_type='L2')
init_sol = tp.fwd_solutions_tracer[0].copy(deepcopy=True)
tp.solve_forward()

# TODO: Plot inverted elements

# Compare initial and final tracer concentrations
final_vol = assemble(Constant(1.0)*dx(domain=tp.mesh))
final_l1_norm = norm(tp.fwd_solutions_tracer[0], norm_type='L1')
final_l2_norm = norm(tp.fwd_solutions_tracer[0], norm_type='L2')
final_sol = tp.fwd_solutions_tracer[0].copy(deepcopy=True)
print_output("Volume error:       {:.2f}%".format(100*abs(init_vol-final_vol)/init_vol))
print_output("Conservation error: {:.2f}%".format(100*abs(init_l1_norm-final_l1_norm)/init_l1_norm))
init_sol = init_sol.dat.data
final_sol = final_sol.dat.data
assert np.isclose(init_sol.min(), final_sol.min())
assert np.isclose(init_sol.max(), final_sol.max())
assert np.isclose(np.mean(init_sol), np.mean(final_sol))
