from thetis import *

import os
import argparse

from adapt_utils.case_studies.tohoku.options import TohokuOptions
from adapt_utils.swe.tsunami.solver import AdaptiveTsunamiProblem
from adapt_utils.adapt.metric import isotropic_metric, space_time_normalise


parser = argparse.ArgumentParser(prog="plot_dwp_indicators")

# Space-time domain
parser.add_argument("-end_time", help="End time of simulation")
parser.add_argument("-level", help="(Integer) mesh resolution")
parser.add_argument("-num_meshes", help="Number of meshes to consider")

# Solver
parser.add_argument("-family", help="Element family for mixed FE space")

# Mesh adaptation
parser.add_argument("-norm_order", help="p for Lp normalisation (default 1)")
parser.add_argument("-normalisation", help="Normalisation method (default 'complexity')")
parser.add_argument("-target", help="Target space-time complexity (default 1.0e+03)")
parser.add_argument("-h_min", help="Minimum tolerated element size (default 100m)")
parser.add_argument("-h_max", help="Maximum tolerated element size (default 1000km)")

# Outer loop
parser.add_argument("-num_adapt", help="Maximum number of adaptation loop iterations (default 35)")
parser.add_argument("-element_rtol", help="Relative tolerance for element count (default 0.005)")
parser.add_argument("-qoi_rtol", help="Relative tolerance for quantity of interest (default 0.005)")

# QoI
parser.add_argument("-start_time", help="Start time of period of interest")
parser.add_argument("-locations", help="""
Locations of interest, separated by commas. Choose from {'Fukushima Daiichi', 'Onagawa',
'Fukushima Daini', 'Tokai', 'Hamaoka', 'Tohoku', 'Tokyo'}.
""")
parser.add_argument("-radii", help="Radii of interest, separated by commas")

# Misc
parser.add_argument("-debug", help="Print all debugging statements")
args = parser.parse_args()
p = args.norm_order

# Collect locations and radii
if args.locations is None:
    locations = ['Fukushima Daiichi', ]
else:
    locations = args.locations.split(',')
if args.radii is None:
    radii = [100.0e+03 for l in locations]
else:
    radii = [float(r) for r in args.radii.split(',')]
if len(locations) != len(radii):
    msg = "Number of locations ({:d}) and radii ({:d}) do not match."
    raise ValueError(msg.format(len(locations), len(radii)))

# Set parameters for fixed mesh run
kwargs = {

    # Timestepping
    'end_time': float(args.end_time or 1440.0),

    # Space-time domain
    'level': int(args.level or 2),
    'num_meshes': int(args.num_meshes or 6),

    # Solver
    'family': args.family or 'dg-cg',

    # QoI
    'start_time': float(args.start_time or 0.0),
    'radii': radii,
    'locations': locations,

    # Mesh adaptation
    'normalisation': args.normalisation or 'complexity',
    'norm_order': 1 if p is None else None if p == 'inf' else float(p),
    'target': float(args.target or 5.0e+03),
    'h_min': float(args.h_min or 1.0e+02),
    'h_max': float(args.h_max or 1.0e+06),

    # Outer loop
    'element_rtol': float(args.element_rtol or 0.005),
    'qoi_rtol': float(args.qoi_rtol or 0.005),
    'num_adapt': int(args.num_adapt or 1),  # TODO: temp

    # Misc
    'debug': bool(args.debug or False),
    'plot_pvd': True,
}
assert 0.0 <= kwargs['start_time'] <= kwargs['end_time']
op = TohokuOptions(approach='dwp')
op.update(kwargs)

# Setup problem object
swp = AdaptiveTsunamiProblem(op)

for n in range(op.num_adapt):

    # Solve forward to get checkpoints
    swp.solve_forward()

    # --- Convergence criteria

    # Check QoI convergence
    qoi = swp.quantity_of_interest()
    print_output("Quantity of interest {:d}: {:.4e}".format(n+1, qoi))
    swp.qois.append(qoi)
    if len(swp.qois) > 1:
        if np.abs(swp.qois[-1] - swp.qois[-2]) < op.qoi_rtol*swp.qois[-2]:
            print_output("Converged quantity of interest!")
            break

    # # Check maximum number of iterations
    if n == op.num_adapt - 1:
        break

    # Loop over mesh windows *in reverse*
    indicators = [Function(P1, name="DWP indicator") for P1 in swp.P1]
    metrics = [Function(P1_ten, name="Metric") for P1_ten in swp.P1_ten]
    for i in range(swp.num_meshes-1, -1, -1):

        # --- Solve adjoint on current window

        adj_solutions_step = []

        def export_func():
            adj_solutions_step.append(swp.adj_solutions[i].copy(deepcopy=True))

        swp.transfer_adjoint_solution(i)
        swp.setup_solver_adjoint(i)
        swp.solve_adjoint_step(i, export_func=export_func)

        # --- Solve forward on current window

        fwd_solutions_step = []

        def export_func():
            fwd_solutions_step.append(swp.fwd_solvers[i].fields.solution_2d.copy(deepcopy=True))

        swp.transfer_forward_solution(i)
        swp.setup_solver_forward(i)
        swp.solve_forward_step(i, export_func=export_func)

        # --- Assemble indicators and metrics

        n_fwd = len(fwd_solutions_step)
        n_adj = len(adj_solutions_step)
        if n_fwd != n_adj:
            raise ValueError("Mismatching number of indicators ({:d} vs {:d})".format(n_fwd, n_adj))
        I = 0
        op.print_debug("DWP indicators on mesh {:2d}".format(i))
        for j, solutions in enumerate(zip(fwd_solutions_step, reversed(adj_solutions_step))):
            scaling = 0.5 if j in (0, n_fwd-1) else 1.0  # Trapezium rule  # TODO: Other integrators
            fwd_dot_adj = abs(inner(*solutions))
            op.print_debug("    ||<q, q*>||_L2 = {:.4e}".format(assemble(fwd_dot_adj*fwd_dot_adj*dx)))
            I += op.dt*swp.dt_per_mesh*scaling*fwd_dot_adj
        indicators[i].interpolate(I)
        metrics[i].assign(isotropic_metric(indicators[i], noscale=True, normalise=False))

    # Normalise metrics
    space_time_normalise(metrics, op=op)

    # Output to .pvd and .vtu
    metric_file = File(os.path.join(swp.di, 'metric.pvd'))
    for indicator, metric in zip(indicators, metrics):
        swp.indicator_file._topology = None
        swp.indicator_file.write(indicator)
        metric_file._topology = None
        metric_file.write(metric)
